#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib",
#     "pandas",
#     "requests",
# ]
# ///

"""
Live scaling chart for the PEARC26 gravity simulation workshop.

Polls a Google Sheet (published to the web as CSV) that is populated by a
Google Form, and redraws a projector-friendly chart every POLL_SECONDS.

Columns are auto-detected from the Form's question titles by keyword
(core / chunk / wall / int or throughput / user). Adjust KEYWORDS if your
question wording is unusual.
"""


# Type annotations
from __future__ import annotations
from typing import Final

# Standard libs
import os
import io
import re
import time
import warnings

# External libs
import numpy as np
import pandas as pd
import requests
import matplotlib

# Matplotlib configuration
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

SHEET_CSV_URL: Final[str] = os.getenv('GOOGLE_SHEETS_URL', 'dummy-url')
POLL_SECONDS: Final[int] = 5

CORE_CHOICES: Final[List[int]] = [2, 4, 8, 16, 32, 64, 128]
CHUNK_CHOICES: Final[List[int]] = [512, 1024, 2048, 4096, 8192, 16384]

TITLE: Final[str] = "PEARC26 · Gravity Simulation Scaling on Anvil"

# Keyword -> canonical column name (first matching sheet column wins)
KEYWORDS: Final[Dict[str, List[str]]] = {
    "cores": ["core"],
    "chunk": ["chunk"],
    "walltime": ["wall", "time"],
    "throughput": ["force", "int", "throughput", "interaction"],
    "user": ["user", "name"],
}


# ----------------------------------------------------------------------------
# Data fetching / parsing
# ----------------------------------------------------------------------------

def fetch() -> pd.DataFrame:
    """Pull CSV from the published URL endpoint."""
    r = requests.get(SHEET_CSV_URL, timeout=10)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def map_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Construct canonicalized column name mapping from Form/Sheet."""
    mapping = {}
    for canon, keys in KEYWORDS.items():
        for col in df.columns:
            low = col.lower()
            if "timestamp" in low:  # Google Forms bookkeeping column
                continue
            if any(k in low for k in keys) and col not in mapping.values():
                mapping[canon] = col
                break
    return mapping


def parse_walltime(value) -> float:
    """Accept seconds ('83', '83.2s') or clock format ('1:23', '0:01:23')."""
    try:
        s = str(value).strip().lower().rstrip("s")
        if ":" in s:
            parts = [float(p) for p in s.split(":")]
            return sum(p * 60**i for i, p in enumerate(reversed(parts)))
        return float(s)
    except ValueError:
        return np.nan


def parse_throughput(value) -> float:
    """Return throughput in G int/s. Accepts '0.61G int/s', '0.61', '610M'."""
    s = str(value).strip()
    m = re.search(r"([\d.]+)\s*([kmgt]?)", s, re.IGNORECASE)
    if not m:
        return np.nan
    num = float(m.group(1))
    scale = {"": 1, "k": 1e-6, "m": 1e-3, "g": 1, "t": 1e3}[m.group(2).lower()]
    # bare numbers > 100 are probably M int/s typos; leave them alone otherwise
    return num * scale


def load() -> pd.DataFrame:
    """End-to-end data refresh."""
    raw = fetch()
    cols = map_columns(raw)
    need = {"cores", "chunk", "throughput"}
    missing = need - cols.keys()
    if missing:
        raise ValueError(f"Could not find columns for {missing}; "
                         f"sheet has {list(raw.columns)}")
    df = pd.DataFrame({
        "cores": pd.to_numeric(raw[cols["cores"]], errors="coerce"),
        "chunk": pd.to_numeric(raw[cols["chunk"]], errors="coerce"),
        "gints": raw[cols["throughput"]].map(parse_throughput),
    })
    if "walltime" in cols:
        df["walltime"] = raw[cols["walltime"]].map(
            lambda v: parse_walltime(v) if pd.notna(v) else np.nan)
    if "user" in cols:
        df["user"] = raw[cols["user"]].astype(str).str.strip().str.lower()
    # keep only sane rows
    df = df.dropna(subset=["cores", "chunk", "gints"])
    df = df[df["cores"].isin(CORE_CHOICES) & df["chunk"].isin(CHUNK_CHOICES)]
    df = df[(df["gints"] > 0) & (df["gints"] < 100)]
    return df


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

plt.style.use("dark_background")
plt.rcParams.update({
    "font.size": 15,
    "font.family": "DejaVu Sans",
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#0d1117",
    "axes.edgecolor": "#30363d",
    "grid.color": "#21262d",
    "text.color": "#e6edf3",
    "axes.labelcolor": "#e6edf3",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
})

CMAP = plt.cm.plasma
CHUNK_COLORS = {c: CMAP(i / (len(CHUNK_CHOICES) - 1))
                for i, c in enumerate(CHUNK_CHOICES)}

fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=0.25,
                      left=0.06, right=0.97, top=0.88, bottom=0.12)
ax_line = fig.add_subplot(gs[0])
ax_heat = fig.add_subplot(gs[1])
fig.suptitle(TITLE, fontsize=24, fontweight="bold", y=0.96)
status = fig.text(0.06, 0.03, "waiting for first poll…",
                  fontsize=13, color="#8b949e")

rng = np.random.default_rng(0)


def draw(df: pd.DataFrame):
    """Update the graphics with new data."""

    ax_line.clear()
    ax_heat.clear()

    # --- main scaling plot -------------------------------------------------

    for chunk in CHUNK_CHOICES:
        sub = df[df["chunk"] == chunk]
        if sub.empty:
            continue
        color = CHUNK_COLORS[chunk]
        jitter = rng.normal(1.0, 0.02, len(sub))
        ax_line.scatter(sub["cores"] * jitter, sub["gints"],
                        s=45, color=color, alpha=0.45, edgecolors="none",
                        zorder=2)
        med = sub.groupby("cores")["gints"].median().sort_index()
        ax_line.plot(med.index, med.values, "-o", color=color, lw=2.5,
                     ms=8, label=f"{chunk:,}", zorder=3)

    # ideal-scaling reference from the best small-core median
    base = df[df["cores"] == df["cores"].min()]
    if not base.empty:
        b_cores = df["cores"].min()
        b_val = base["gints"].median()
        xs = np.array(CORE_CHOICES, dtype=float)
        ax_line.plot(xs, b_val * xs / b_cores, "--", color="#484f58",
                     lw=1.5, zorder=1, label="ideal scaling")

    ax_line.set_xscale("log", base=2)
    ax_line.set_yscale("log", base=2)
    ax_line.set_xticks(CORE_CHOICES)
    ax_line.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax_line.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_line.set_xlabel("NUM_CORES")
    ax_line.set_ylabel("throughput  (G int/s)")
    ax_line.grid(True, which="both", alpha=0.5)
    ax_line.legend(title="CHUNK_SIZE", fontsize=12, title_fontsize=12,
                   loc="upper left", framealpha=0.2, ncols=2)
    ax_line.axvline(64, color="#f85149", lw=1, ls=":", alpha=0.6)
    ax_line.text(64, ax_line.get_ylim()[0] * 1.1, " socket boundary",
                 color="#f85149", fontsize=11, alpha=0.8)

    # --- heatmap: median throughput per (cores, chunk) ---------------------

    grid = np.full((len(CHUNK_CHOICES), len(CORE_CHOICES)), np.nan)
    piv = df.groupby(["chunk", "cores"])["gints"].median()
    for (chunk, cores), val in piv.items():
        grid[CHUNK_CHOICES.index(chunk), CORE_CHOICES.index(cores)] = val
    masked = np.ma.masked_invalid(grid)
    im = ax_heat.imshow(masked, cmap="plasma", aspect="auto", origin="lower")
    ax_heat.set_xticks(range(len(CORE_CHOICES)), CORE_CHOICES)
    ax_heat.set_yticks(range(len(CHUNK_CHOICES)),
                       [f"{c:,}" for c in CHUNK_CHOICES])
    ax_heat.set_xlabel("NUM_CORES")
    ax_heat.set_ylabel("CHUNK_SIZE")
    ax_heat.set_title("median G int/s", fontsize=14)
    for i in range(len(CHUNK_CHOICES)):
        for j in range(len(CORE_CHOICES)):
            if not np.isnan(grid[i, j]):
                ax_heat.text(j, i, f"{grid[i, j]:.2f}", ha="center",
                             va="center", fontsize=10,
                             color="white" if grid[i, j] < np.nanmax(grid) * 0.6
                             else "black")

    # --- status line --------------------------------------------------------

    n_users = df["user"].nunique() if "user" in df else "?"
    covered = (~np.isnan(grid)).sum()
    status.set_text(
        f"{len(df)} submissions · {n_users} participants · "
        f"{covered}/{grid.size} grid cells covered · "
        f"updated {time.strftime('%H:%M:%S')}")


def update(_frame):
    """Apply drawing routing."""
    try:
        df = load()
    except Exception as exc:  # keep the show going on transient errors
        warnings.warn(f"poll failed: {exc}")
        status.set_text(f"poll failed ({exc}) — retrying "
                        f"· {time.strftime('%H:%M:%S')}")
        return
    if df.empty:
        status.set_text(f"no valid submissions yet · "
                        f"{time.strftime('%H:%M:%S')}")
        return
    draw(df)


anim = FuncAnimation(fig, update, interval=(POLL_SECONDS * 1_000), cache_frame_data=False)
update(None)
plt.show()
