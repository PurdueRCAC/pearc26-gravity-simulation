#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "jax[cpu]",
# ]
# ///
"""
N-body gravitational simulation using JAX on CPU.

Direct O(N²) pairwise force summation with symplectic velocity-Verlet
integration, several initial-condition scenarios, and live in-terminal
rendering. Designed to exercise all available CPU cores through JAX/XLA
parallelism — ideal for demonstrating single-node HPC jobs.

Usage:
  ./sim.py --list                  show available scenarios
  ./sim.py galaxy                  spiral galaxy (default)
  ./sim.py collision -n 60000      two galaxies, tidal tails
  ./sim.py solar -s 6000           Sun + planets + asteroid belt
  ./sim.py plummer --frames 0      classic cluster, no rendering
"""


# Type annotations
from __future__ import annotations
from typing import Callable, Final

# Standard libs
import os
import sys
import math
import time
import argparse
import platform
import subprocess
from dataclasses import dataclass
from functools import partial
from socket import gethostname
from datetime import datetime

# Pre-config necessary for Jax
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

# External libs
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, lax


WIDTH: Final[int] = 72


# ---------------------------------------------------------------------------
# Terminal formatting
# ---------------------------------------------------------------------------

BOLD = '\033[1m'
DIM = '\033[2m'
RESET = '\033[0m'
CYAN = '\033[36m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
RED = '\033[31m'
COLOR = True

# 256-color ramp (black → purple → red → orange → yellow → white)
INFERNO: Final[tuple[int, ...]] = (16, 53, 90, 126, 161, 197, 203, 209, 214, 220, 226, 230, 231)
ASCII_RAMP: Final[str] = ' .:-=+*#%@'


def disable_color() -> None:
    """Strip all ANSI styling (--plain or NO_COLOR)."""
    global BOLD, DIM, RESET, CYAN, GREEN, YELLOW, RED, COLOR
    BOLD = DIM = RESET = CYAN = GREEN = YELLOW = RED = ''
    COLOR = False


def header(title: str) -> None:
    """Print a section header with a decorative rule."""
    rule = '─' * (WIDTH - len(title) - 4)
    print(f'\n {CYAN}── {BOLD}{title}{RESET} {CYAN}{rule}{RESET}')


def kv(key: str, value: str, indent: int = 2) -> None:
    """Print a key-value pair with consistent alignment."""
    print(f'{" " * indent}{DIM}{key:<16}{RESET}{value}')


def bar_chart(label: str, count: int, max_count: int, bar_width: int = 36) -> None:
    """Print a single horizontal bar-chart row."""
    n = int(bar_width * count / max_count) if max_count > 0 else 0
    print(f'  {label:<10} {GREEN}{"█" * n:<{bar_width}}{RESET} {count:>7,}')


def cpu_model() -> str:
    """Best-effort CPU model name (Linux /proc, macOS sysctl)."""
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except (FileNotFoundError, PermissionError):
        pass
    if platform.system() == 'Darwin':
        try:
            return subprocess.check_output(
                ['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or 'unknown'


def available_cpus() -> int:
    """CPUs available to this process (Slurm-aware)."""
    if slurm := os.environ.get('SLURM_CPUS_ON_NODE'):
        return int(slurm)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# ---------------------------------------------------------------------------
# Physics kernels
#
# The force/energy kernels process the N×N interaction matrix in blocks of
# CHUNK rows: peak memory stays at chunk×N rather than N×N, while XLA still
# parallelizes each block across every CPU core. With softening > 0 the
# i == j self-term vanishes naturally (diff is exactly zero), so no diagonal
# masking is required.
# ---------------------------------------------------------------------------

@partial(jit, static_argnames=('chunk',))
def compute_accelerations(positions: jax.Array, masses: jax.Array,
                          g: float, softening: float, *, chunk: int) -> jax.Array:
    """Gravitational accelerations via chunked direct O(N²) summation."""
    n = positions.shape[0]
    pad = -n % chunk
    pos_pad = jnp.pad(positions, ((0, pad), (0, 0)))

    def block(pos_block: jax.Array) -> jax.Array:
        diff = positions[None, :, :] - pos_block[:, None, :]            # (chunk, n, 3)
        dist_sq = jnp.sum(diff * diff, axis=-1) + softening * softening
        weight = masses[None, :] * dist_sq ** -1.5                       # (chunk, n)
        return g * jnp.einsum('cn,cnd->cd', weight, diff)

    acc = lax.map(block, pos_pad.reshape(-1, chunk, 3))
    return acc.reshape(-1, 3)[:n]


@partial(jit, static_argnames=('chunk',))
def run_block(positions: jax.Array, velocities: jax.Array, accelerations: jax.Array,
              masses: jax.Array, g: float, dt: float, softening: float,
              n_steps: int, *, chunk: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Advance the system n_steps velocity-Verlet steps in one fused XLA call.

    Fusing the integration loop into a single jitted call eliminates
    Python dispatch overhead between steps; n_steps is a traced value so
    differing block lengths do not trigger recompilation.
    """
    def body(_, state):
        pos, vel, acc = state
        pos = pos + vel * dt + 0.5 * acc * dt * dt
        acc_new = compute_accelerations(pos, masses, g, softening, chunk=chunk)
        vel = vel + 0.5 * (acc + acc_new) * dt
        return pos, vel, acc_new

    return lax.fori_loop(0, n_steps, body, (positions, velocities, accelerations))


@jit
def kinetic_energy(velocities: jax.Array, masses: jax.Array) -> jax.Array:
    """Total kinetic energy of the system."""
    return 0.5 * jnp.sum(masses[:, None] * velocities ** 2)


@partial(jit, static_argnames=('chunk',))
def potential_energy(positions: jax.Array, masses: jax.Array,
                     g: float, softening: float, *, chunk: int) -> jax.Array:
    """Total gravitational potential energy (softened), chunked like the forces.

    The i == j diagonal is masked exactly inside each block — subtracting
    it analytically afterward loses all precision in float32 whenever the
    softening length is small compared to the masses.
    """
    n = positions.shape[0]
    pad = -n % chunk
    pos_pad = jnp.pad(positions, ((0, pad), (0, 0)))
    m_pad = jnp.pad(masses, (0, pad))
    row_ids = jnp.arange(n + pad)

    def block(args):
        pos_block, m_block, rows = args
        diff = positions[None, :, :] - pos_block[:, None, :]
        dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + softening * softening)
        pair = m_block[:, None] * masses[None, :] / dist
        pair = jnp.where(rows[:, None] == jnp.arange(n)[None, :], 0.0, pair)
        return jnp.sum(pair)

    total = jnp.sum(lax.map(block, (pos_pad.reshape(-1, chunk, 3),
                                    m_pad.reshape(-1, chunk),
                                    row_ids.reshape(-1, chunk))))
    return -0.5 * g * total


# ---------------------------------------------------------------------------
# Initial conditions
#
# All builders return (positions, velocities, masses) as float64 NumPy
# arrays; main() casts to the working precision. Units are dimensionless
# (G = 1, M_total ~ 1) except the solar system, which uses AU, years, and
# solar masses with G = 4π².
# ---------------------------------------------------------------------------

def random_directions(n: int, rng: np.random.Generator) -> np.ndarray:
    """n isotropically random unit vectors."""
    cos_theta = rng.uniform(-1.0, 1.0, size=n)
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)
    phi = rng.uniform(0.0, math.tau, size=n)
    return np.column_stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])


def sample_plummer(n: int, rng: np.random.Generator, *,
                   total_mass: float = 1.0, a: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Positions and velocities for a Plummer sphere in virial equilibrium.

    Radii follow the inverse CDF of the Plummer mass profile (truncated at
    r ≈ 10a); speeds are drawn from the isotropic distribution function
    f(q) ∝ q²(1-q²)^(7/2) by rejection sampling.
    """
    u = rng.uniform(0.0, 0.985, size=n)            # u < 0.985 caps r at ~10a
    r = a / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
    positions = random_directions(n, rng) * r[:, None]

    q = np.empty(n)
    need = np.arange(n)
    g_max = 0.0923                                 # max of q²(1-q²)^3.5 is ≈0.0922
    while need.size:
        cand = rng.uniform(0.0, 1.0, size=need.size)
        accept = rng.uniform(0.0, g_max, size=need.size) < cand ** 2 * (1 - cand ** 2) ** 3.5
        q[need[accept]] = cand[accept]
        need = need[~accept]

    v_escape = np.sqrt(2.0 * total_mass / np.sqrt(r ** 2 + a ** 2))
    velocities = random_directions(n, rng) * (q * v_escape)[:, None]
    return positions, velocities


def make_plummer(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classic Plummer cluster: should sit quietly in virial equilibrium."""
    positions, velocities = sample_plummer(n, rng)
    masses = np.full(n, 1.0 / n)
    velocities -= np.mean(velocities, axis=0)
    return positions, velocities, masses


def make_disk_galaxy(n: int, rng: np.random.Generator, *,
                     total_mass: float = 1.0, r_disk: float = 1.0,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotating disk galaxy: exponential disk (75%) + Plummer bulge (25%).

    Circular speeds come from the spherically-averaged enclosed mass; a
    small velocity dispersion keeps the disk from being perfectly cold.
    The disk is dynamically responsive on purpose — spiral structure and a
    central bar develop within a couple of rotation periods.
    """
    n_bulge = max(1, round(0.25 * n))
    n_disk = n - n_bulge
    a_bulge = 0.2
    m_bulge = total_mass * n_bulge / n
    m_disk = total_mass - m_bulge

    # Disk radii from the exponential-profile CDF, inverted numerically
    grid = np.linspace(1e-3, 6.0 * r_disk, 4096)
    x = grid / r_disk
    cdf = 1.0 - (1.0 + x) * np.exp(-x)
    r = np.interp(rng.uniform(0.0, cdf[-1], size=n_disk), cdf, grid)
    phi = rng.uniform(0.0, math.tau, size=n_disk)
    z = rng.normal(0.0, 0.08 * r_disk, size=n_disk)
    pos_disk = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def v_circ(rr: np.ndarray) -> np.ndarray:
        m_enc = (m_bulge * rr ** 3 / (rr ** 2 + a_bulge ** 2) ** 1.5
                 + m_disk * (1.0 - (1.0 + rr / r_disk) * np.exp(-rr / r_disk)))
        return np.sqrt(m_enc / np.maximum(rr, 1e-6))

    vc = v_circ(r)
    vel_disk = np.column_stack([-vc * np.sin(phi), vc * np.cos(phi), np.zeros(n_disk)])
    vel_disk += rng.normal(0.0, 1.0, size=(n_disk, 3)) * (0.08 * vc)[:, None]
    vel_disk[:, 2] *= 0.5

    # Bulge: Plummer sphere with its own DF (slightly cold in the combined
    # potential — it settles within the first few timesteps)
    pos_bulge, vel_bulge = sample_plummer(n_bulge, rng, total_mass=m_bulge, a=a_bulge)

    positions = np.vstack([pos_disk, pos_bulge])
    velocities = np.vstack([vel_disk, vel_bulge])
    masses = np.full(n, total_mass / n)
    velocities -= np.sum(masses[:, None] * velocities, axis=0) / total_mass
    return positions, velocities, masses


def rotation_x(theta: float) -> np.ndarray:
    """Rotation matrix about the x-axis."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def make_collision(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two equal disk galaxies on a sub-parabolic prograde encounter.

    The second disk is inclined 45°; the offset trajectory produces a
    grazing first passage, tidal tails, and an eventual merger.
    """
    n1 = n // 2
    p1, v1, m1 = make_disk_galaxy(n1, rng)
    p2, v2, m2 = make_disk_galaxy(n - n1, rng)

    tilt = rotation_x(math.radians(45.0))
    p2 = p2 @ tilt.T
    v2 = v2 @ tilt.T

    p1 += np.array([-4.0, -1.2, 0.0]);  v1 += np.array([+0.30, 0.0, 0.0])
    p2 += np.array([+4.0, +1.2, 0.0]);  v2 += np.array([-0.30, 0.0, 0.0])

    return np.vstack([p1, p2]), np.vstack([v1, v2]), np.concatenate([m1, m2])


# (name, semi-major axis [AU], mass [M_sun])
PLANETS: Final[tuple[tuple[str, float, float], ...]] = (
    ('Mercury', 0.387, 1.660e-7),
    ('Venus',   0.723, 2.448e-6),
    ('Earth',   1.000, 3.003e-6),
    ('Mars',    1.524, 3.227e-7),
    ('Jupiter', 5.203, 9.545e-4),
    ('Saturn',  9.537, 2.858e-4),
    ('Uranus',  19.19, 4.366e-5),
    ('Neptune', 30.07, 5.151e-5),
)
G_SOLAR: Final[float] = 4.0 * math.pi ** 2        # AU³ / (M_sun · yr²)


def make_solar(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The Sun, eight planets, the main asteroid belt, and Jupiter's trojans.

    Units: AU, years, solar masses (G = 4π²). Planets start on circular
    orbits at random phases; asteroids get small eccentricities and
    inclinations. Asteroid masses are tiny but every body still
    participates in the full O(N²) force sum.
    """
    if n < 64:
        raise SystemExit('error: solar scenario needs at least 64 particles')

    pos = [np.zeros(3)]
    vel = [np.zeros(3)]
    mass = [1.0]
    phases = rng.uniform(0.0, math.tau, size=len(PLANETS))
    for (name, a, m), ph in zip(PLANETS, phases):
        v = math.sqrt(G_SOLAR / a)
        pos.append(np.array([a * math.cos(ph), a * math.sin(ph), 0.0]))
        vel.append(np.array([-v * math.sin(ph), v * math.cos(ph), 0.0]))
        mass.append(m)

    n_ast = n - len(pos)
    n_trojan = round(0.15 * n_ast)
    n_belt = n_ast - n_trojan

    a_belt = rng.uniform(2.1, 3.3, size=n_belt)
    ph_belt = rng.uniform(0.0, math.tau, size=n_belt)

    phi_jup = phases[4]                            # Jupiter leads the trojan swarms
    side = rng.choice((-1.0, 1.0), size=n_trojan)
    a_tro = 5.203 + rng.normal(0.0, 0.12, size=n_trojan)
    ph_tro = phi_jup + side * (math.pi / 3.0) + rng.normal(0.0, 0.18, size=n_trojan)

    a = np.concatenate([a_belt, a_tro])
    ph = np.concatenate([ph_belt, ph_tro])
    v_kep = np.sqrt(G_SOLAR / a)
    v_tan = v_kep * (1.0 + rng.normal(0.0, 0.02, size=n_ast))
    v_rad = v_kep * rng.normal(0.0, 0.02, size=n_ast)
    v_z = v_kep * rng.normal(0.0, 0.015, size=n_ast)

    pos_ast = np.column_stack([a * np.cos(ph), a * np.sin(ph),
                               a * rng.normal(0.0, 0.03, size=n_ast)])
    vel_ast = np.column_stack([-v_tan * np.sin(ph) + v_rad * np.cos(ph),
                               +v_tan * np.cos(ph) + v_rad * np.sin(ph), v_z])

    positions = np.vstack([np.array(pos), pos_ast])
    velocities = np.vstack([np.array(vel), vel_ast])
    masses = np.concatenate([np.array(mass), np.full(n_ast, 1e-12)])
    velocities -= np.sum(masses[:, None] * velocities, axis=0) / np.sum(masses)
    return positions, velocities, masses


def make_collapse(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cold collapse: a uniform sphere released nearly from rest.

    A gentle solid-body spin (2T/|W| ≈ 0.1) breaks spherical symmetry, so
    the violent collapse and rebound settle into a flattened remnant.
    """
    radius = 1.8
    r = radius * rng.uniform(0.0, 1.0, size=n) ** (1.0 / 3.0)
    positions = random_directions(n, rng) * r[:, None]
    velocities = 0.15 * np.cross(np.array([0.0, 0.0, 1.0]), positions)
    masses = np.full(n, 1.0 / n)
    return positions, velocities, masses


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    blurb: str
    n: int
    steps: int
    dt: float
    softening: float
    g: float
    extent: float                                   # half-width of rendered view
    build: Callable[[int, np.random.Generator], tuple[np.ndarray, np.ndarray, np.ndarray]]
    time_unit: str = ''
    markers: tuple[tuple[int, str, int], ...] = ()  # (particle index, char, 256-color)
    legend: str = ''


SOLAR_MARKERS: Final[tuple[tuple[int, str, int], ...]] = (
    (0, '*', 226), (1, 'm', 250), (2, 'v', 229), (3, 'E', 45), (4, 'R', 203),
    (5, 'J', 216), (6, 'S', 223), (7, 'U', 123), (8, 'N', 75),
)
SOLAR_LEGEND: Final[str] = ('* Sun   m Mercury   v Venus   E Earth   R Mars\n'
                            'J Jupiter   S Saturn   U Uranus   N Neptune')

SCENARIOS: Final[dict[str, Scenario]] = {s.key: s for s in (
    Scenario('galaxy', 'Spiral Galaxy',
             'rotating exponential disk + central bulge; arms wind up live',
             n=30_000, steps=2_000, dt=0.005, softening=0.05, g=1.0, extent=4.0,
             build=make_disk_galaxy),
    Scenario('collision', 'Galaxy Collision',
             'two disk galaxies collide; tidal tails, then a merger',
             n=40_000, steps=3_000, dt=0.005, softening=0.05, g=1.0, extent=8.0,
             build=make_collision),
    Scenario('solar', 'Solar System',
             'Sun, 8 planets, asteroid belt + trojans (AU · yr · M_sun)',
             n=20_000, steps=6_000, dt=0.002, softening=1e-4, g=G_SOLAR, extent=6.0,
             build=make_solar, time_unit=' yr', markers=SOLAR_MARKERS, legend=SOLAR_LEGEND),
    Scenario('plummer', 'Plummer Cluster',
             'star cluster in virial equilibrium; the classic stability test',
             n=20_000, steps=500, dt=0.005, softening=0.05, g=1.0, extent=3.0,
             build=make_plummer),
    Scenario('collapse', 'Cold Collapse',
             'uniform sphere released from rest; violent relaxation in action',
             n=30_000, steps=1_500, dt=0.002, softening=0.05, g=1.0, extent=2.5,
             build=make_collapse),
)}


# ---------------------------------------------------------------------------
# Live terminal rendering
#
# Particles are binned onto a 2-D grid and shown as a log-scaled density
# map. In color mode each character cell encodes two vertical pixels with
# the '▀' half-block (foreground = top pixel, background = bottom pixel),
# giving a 64×60 pixel image in 30 terminal rows. ANSI codes land in Slurm
# output files too — view them with `less -R` or `cat`.
# ---------------------------------------------------------------------------

def render_frame(positions: np.ndarray, extent: float, *, title: str,
                 markers: tuple[tuple[float, float, str, int], ...] = (),
                 legend: str = '', cols: int = 64, rows: int = 30) -> None:
    """Render a top-down (x-y) density map of the particle field."""
    header(title)
    print()
    x, y = positions[:, 0], positions[:, 1]
    span = [[-extent, extent], [-extent, extent]]

    def cell_of(mx: float, my: float) -> tuple[int, int]:
        col = min(cols - 1, max(0, int((mx + extent) / (2 * extent) * cols)))
        row = min(rows - 1, max(0, int((extent - my) / (2 * extent) * rows)))
        return row, col

    overlay = {cell_of(mx, my): (ch, color) for mx, my, ch, color in markers
               if abs(mx) < extent and abs(my) < extent}

    if COLOR:
        hist, _, _ = np.histogram2d(x, y, bins=(cols, 2 * rows), range=span)
        img = hist.T[::-1]                          # row 0 = +y, pixel rows
        cmax = img.max()
        scale = (len(INFERNO) - 1) / math.log1p(cmax) if cmax > 0 else 0.0
        idx = np.minimum(np.log1p(img) * scale, len(INFERNO) - 1).astype(int)
        for r in range(rows):
            cells = []
            for c in range(cols):
                bg = INFERNO[idx[2 * r + 1, c]]
                if (r, c) in overlay:
                    ch, fg = overlay[r, c]
                    cells.append(f'\033[1;38;5;{fg};48;5;{bg}m{ch}')
                else:
                    cells.append(f'\033[38;5;{INFERNO[idx[2 * r, c]]};48;5;{bg}m▀')
            print('   ' + ''.join(cells) + RESET)
    else:
        hist, _, _ = np.histogram2d(x, y, bins=(cols, rows), range=span)
        img = hist.T[::-1]
        cmax = img.max()
        scale = (len(ASCII_RAMP) - 1) / math.log1p(cmax) if cmax > 0 else 0.0
        idx = np.minimum(np.log1p(img) * scale, len(ASCII_RAMP) - 1).astype(int)
        for r in range(rows):
            line = [overlay[r, c][0] if (r, c) in overlay else ASCII_RAMP[idx[r, c]]
                    for c in range(cols)]
            print('   ' + ''.join(line))

    print()
    print(f'   {DIM}x–y plane · view ±{extent:g}{RESET}')
    for line in filter(None, legend.split('\n')):
        print(f'   {DIM}{line}{RESET}')
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='sim.py',
        description='N-body gravitational simulation — JAX on CPU.',
        epilog='examples:\n'
               '  ./sim.py galaxy\n'
               '  ./sim.py collision -n 60000 -s 3000\n'
               '  ./sim.py solar --frames 12\n',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scenario', nargs='?', default='galaxy', choices=list(SCENARIOS),
                        help='initial conditions (default: galaxy)')
    parser.add_argument('-n', '--particles', type=int, metavar='N',
                        help='number of particles (default: per scenario)')
    parser.add_argument('-s', '--steps', type=int, metavar='K',
                        help='number of integration steps (default: per scenario)')
    parser.add_argument('--dt', type=float, help='timestep (default: per scenario)')
    parser.add_argument('--softening', type=float, metavar='EPS',
                        help='gravitational softening length (default: per scenario)')
    parser.add_argument('--extent', type=float, metavar='R',
                        help='half-width of the rendered view (default: per scenario)')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed (default: 42)')
    parser.add_argument('--frames', type=int, default=8, metavar='F',
                        help='density-map frames to render (default: 8; 0 disables)')
    parser.add_argument('--report-every', type=int, metavar='K',
                        help='steps between progress rows (default: steps/24)')
    parser.add_argument('--chunk', type=int, default=2048, metavar='C',
                        help='force-kernel block size (default: 2048)')
    parser.add_argument('--x64', action='store_true',
                        help='use float64 instead of float32')
    parser.add_argument('--plain', action='store_true',
                        help='disable ANSI color (also via NO_COLOR env)')
    parser.add_argument('--list', action='store_true',
                        help='list scenarios and exit')
    return parser.parse_args()


def list_scenarios() -> None:
    print()
    for s in SCENARIOS.values():
        print(f'  {BOLD}{s.key:<10}{RESET} {s.title:<18} {DIM}{s.blurb}{RESET}')
        print(f'  {"":<10} {DIM}defaults: n={s.n:,} steps={s.steps:,} '
              f'dt={s.dt} softening={s.softening}{RESET}')
    print()


def main() -> None:

    args = parse_args()
    if args.plain or os.environ.get('NO_COLOR'):
        disable_color()
    if args.list:
        list_scenarios()
        return

    scen = SCENARIOS[args.scenario]
    n = args.particles or scen.n
    n_steps = args.steps or scen.steps
    dt = args.dt if args.dt is not None else scen.dt
    softening = args.softening if args.softening is not None else scen.softening
    extent = args.extent or scen.extent
    g = scen.g
    chunk = max(16, min(args.chunk, max(16, n)))
    report_every = args.report_every or max(1, n_steps // 24)
    unit = scen.time_unit

    if args.x64:
        jax.config.update('jax_enable_x64', True)
    dtype = np.float64 if args.x64 else np.float32

    hostname = gethostname()
    now = datetime.now()

    print()
    print(f' {CYAN}{"═" * WIDTH}{RESET}')
    title = 'N-Body Gravitational Simulation'
    subtitle = 'Direct O(N²) · Velocity Verlet · JAX on CPU'
    print(f' {CYAN}║{RESET}{BOLD}{title:^{WIDTH - 2}}{RESET}{CYAN}║{RESET}')
    print(f' {CYAN}║{RESET}{DIM}{subtitle:^{WIDTH - 2}}{RESET}{CYAN}║{RESET}')
    print(f' {CYAN}{"═" * WIDTH}{RESET}')

    header('Environment')
    kv('Hostname', hostname)

    slurm_job = os.environ.get('SLURM_JOB_ID')
    if slurm_job:
        kv('Slurm Job ID', slurm_job)
        kv('Partition', os.environ.get('SLURM_JOB_PARTITION', '—'))
        kv('Node List', os.environ.get('SLURM_JOB_NODELIST', '—'))

    kv('CPUs', str(available_cpus()))
    kv('CPU Model', cpu_model())
    kv('Platform', platform.platform())
    kv('Date', now.strftime('%Y-%m-%d %H:%M:%S'))
    kv('Python', platform.python_version())
    kv('JAX', jax.__version__)
    kv('JAX backend', jax.default_backend())

    header('Simulation')
    kv('Scenario', f'{BOLD}{scen.title}{RESET} — {scen.blurb}')
    kv('Particles', f'{n:,}')
    kv('Steps', f'{n_steps:,}')
    kv('Timestep (dt)', f'{dt}{unit}')
    kv('Softening (ε)', f'{softening}')
    kv('Grav. const G', f'{g:g}')
    kv('Integrator', 'Velocity Verlet (symplectic)')
    kv('Precision', 'float64' if args.x64 else 'float32')
    kv('Force kernel', f'chunked direct sum · {chunk} rows/block')
    kv('Interactions', f'{n * n:,} / step')

    header('Initialization')
    t0 = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    pos_np, vel_np, mass_np = scen.build(n, rng)
    positions = jnp.asarray(pos_np, dtype=dtype)
    velocities = jnp.asarray(vel_np, dtype=dtype)
    masses = jnp.asarray(mass_np, dtype=dtype)
    t_ic = time.perf_counter()
    print(f'  {scen.title} initial conditions built in {t_ic - t0:.2f}s')

    # Warm up every kernel so the first timed block measures pure compute
    accelerations = compute_accelerations(positions, masses, g, softening, chunk=chunk)
    jax.block_until_ready(run_block(positions, velocities, accelerations,
                                    masses, g, dt, softening, 0, chunk=chunk))
    ke0 = float(kinetic_energy(velocities, masses))
    pe0 = float(potential_energy(positions, masses, g, softening, chunk=chunk))
    e0 = ke0 + pe0
    t_warm = time.perf_counter()
    print(f'  JIT compiled in {t_warm - t_ic:.1f}s')

    def frame(step_num: int) -> None:
        """Render the current particle field as a density map."""
        p = np.asarray(positions)
        marks = tuple((float(p[i, 0]), float(p[i, 1]), ch, color)
                      for i, ch, color in scen.markers)
        render_frame(p, extent, markers=marks, legend=scen.legend,
                     title=f'Snapshot · t = {step_num * dt:.2f}{unit} · step {step_num:,}')

    def progress_header() -> None:
        print(f'\n  {DIM}{"Step":>6}  {"Time":>8}  {"KE":>11}  {"PE":>11}'
              f'  {"E_total":>11}  {"ΔE/E₀":>10}  {"Rate":>11}{RESET}')

    last_wall = time.perf_counter()
    last_step = 0

    def report(step_num: int) -> None:
        """Print a progress row with energy diagnostics and throughput."""
        nonlocal last_wall, last_step
        ke = float(kinetic_energy(velocities, masses))
        pe = float(potential_energy(positions, masses, g, softening, chunk=chunk))
        e = ke + pe
        de = abs((e - e0) / e0) if e0 != 0 else 0.0

        wall = time.perf_counter()
        elapsed = wall - last_wall
        pair_rate = n * n * (step_num - last_step) / elapsed if elapsed > 0 else 0.0
        last_wall, last_step = wall, step_num

        rate_str = f'{pair_rate / 1e9:.2f}G int/s' if step_num > 0 else '---'
        color = GREEN if de < 1e-4 else YELLOW if de < 1e-2 else RED
        print(f'  {step_num:>6}  {step_num * dt:>8.3f}  {ke:>11.4e}  {pe:>11.4e}'
              f'  {e:>11.4e}  {color}{de:>10.2e}{RESET}  {rate_str:>11}', flush=True)

    if args.frames > 0:
        frame(0)
    header('Progress')
    progress_header()
    report(0)

    frame_targets = ({round(n_steps * i / args.frames) for i in range(1, args.frames + 1)}
                     if args.frames > 0 else set())

    sim_start = time.perf_counter()
    last_wall = sim_start
    done = 0
    while done < n_steps:
        block_steps = min(report_every, n_steps - done)
        positions, velocities, accelerations = run_block(
            positions, velocities, accelerations, masses,
            g, dt, softening, block_steps, chunk=chunk)
        jax.block_until_ready(positions)
        done += block_steps
        report(done)
        if frame_targets and done >= min(frame_targets):
            frame_targets = {t for t in frame_targets if t > done}
            frame(done)
            if done < n_steps:
                progress_header()
    sim_wall = time.perf_counter() - sim_start

    ke_f = float(kinetic_energy(velocities, masses))
    pe_f = float(potential_energy(positions, masses, g, softening, chunk=chunk))
    e_f = ke_f + pe_f

    header('Results')
    kv('Wall time', f'{sim_wall:.2f}s')
    kv('Particle-steps', f'{n * n_steps:,}')
    kv('Interactions', f'{n * n * n_steps:,}')
    if sim_wall > 0:
        kv('Throughput', f'{n * n_steps / sim_wall / 1e6:.2f}M particle-steps/s')
        kv('Force rate', f'{n * n * n_steps / sim_wall / 1e9:.2f}G interactions/s')
    kv('Energy drift', f'{abs((e_f - e0) / e0):.2e}' if e0 != 0 else 'n/a')
    virial = -2 * ke_f / pe_f if pe_f != 0 else float('nan')
    kv('Virial ratio', f'{virial:.4f}  (equilibrium: 1.0)')

    header('Radial Distribution')
    pos_np = np.asarray(positions)
    mass_np = np.asarray(masses)
    com = np.sum(mass_np[:, None] * pos_np, axis=0) / np.sum(mass_np)
    radii = np.linalg.norm(pos_np - com, axis=1)

    edges = np.array([0.0, 0.25, 0.5, 1.0, 2.0]) * extent
    bins = np.append(edges, np.inf)
    labels = [f'r < {e:g}' for e in edges[1:]] + [f'r > {edges[-1]:g}']
    counts, _ = np.histogram(radii, bins=bins)
    max_count = int(counts.max())

    for label, count in zip(labels, counts):
        bar_chart(label, int(count), max_count)

    total_wall = time.perf_counter() - t0
    print(f'\n {CYAN}{"═" * WIDTH}{RESET}')
    print(f' {DIM}Completed in {total_wall:.1f}s total on {hostname}{RESET}')
    print()


if __name__ == '__main__':
    main()
