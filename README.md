# Gravity Simulation Scaling Workshop

**PEARC26 Student Program — Hands-On Exercise on Anvil**

Welcome! In this exercise you'll run an N-body gravity simulation on
[Anvil](https://www.rcac.purdue.edu/anvil), Purdue's flagship supercomputer,
and explore how two knobs — **how many CPU cores you use** and **how the work
is chunked** — change performance. Everyone in the room will run a different
combination, and together we'll build a live scaling chart on the big screen.

No prior HPC experience needed. If you can edit a text file and press Enter,
you're qualified.

---

## 1. What you're running

`sim.py` is a brute-force gravitational N-body simulation: every particle
pulls on every other particle, every timestep. That's an enormous number of
*pairwise interactions*, which makes it a perfect stress test for a CPU.

The performance metric we care about is **throughput**: how many interactions
the machine computes per second. The simulation reports this as the
**Force rate** when it finishes, like:

```
Force rate      1.07G interactions/s
```

That's 1.07 *billion* interactions per second. Your mission is to make that
number as large as you can — and to understand *why* it changes.

## 2. Setup

Log in to Anvil and clone this repository:

```bash
git clone https://github.com/purduercac/pearc26-gravity-simulation
cd pearc26-gravity-simulation
```

Everything you need is in here. The two files that matter:

| File        | What it is                                                        |
|-------------|-------------------------------------------------------------------|
| `sim.py`    | The simulation. You don't need to edit it (but do peek inside).   |
| `submit.sh` | The Slurm job script you'll submit. **You edit the top of this.** |

## 3. Your two knobs

Open `submit.sh` in your editor of choice (`nano submit.sh` works fine).
At the top you'll find:

```bash
NUM_CORES=16      # pick from: 2, 4, 8, 16, 32, 64, 128
CHUNK_SIZE=2048   # pick from: 512, 1024, 2048, 4096, 8192, 16384
```

**`NUM_CORES`** — how many CPU cores the simulation may use. More cores means
more raw compute power... usually.

**`CHUNK_SIZE`** — how many particles each worker processes per unit of work.
Small chunks give fine-grained load balancing but more overhead; large chunks
have less overhead but can leave workers idle. Somewhere in between is a sweet
spot — and it may *move* depending on how many cores you have.

We'll assign combinations around the room so we cover the whole grid.
When in doubt: pick a pair nobody has claimed yet.

You will also need to update this line in the job script to use the real 
allocation (aka account) name:

```
#SBATCH --account=YOUR_ALLOCATION   # <-- EDIT: your ACCESS allocation
```

## 4. Submit the job and watch the queue

Submit:

```bash
sbatch submit.sh
```

Slurm replies with a job ID:

```
Submitted batch job 4815162
```

Watch your job in the queue:

```bash
squeue --me
```

```
JOBID        USER      ACCOUNT       NAME     NODES    CPUS     TIME_LIMIT   ST   TIME
19257550   x-user    <account>    gravity         1     128          30:00    R   0:15
```

The `ST` column is the state: `PD` = pending (waiting for resources),
`R` = running. Re-run `squeue -u $USER` to refresh — or use
`watch -n 5 squeue --me` to auto-refresh every 5 seconds (Ctrl-C to exit).
When your job disappears from the list, it's done. Small jobs here take on the
order of a minute or two to run.

Made a mistake? Cancel with `scancel <JOBID>` and resubmit.

## 5. Read your results

When the job finishes you'll find two files in the directory, named after
the job ID:

**Standard output (`gravity-4815162.out`)** — the simulation's own output:
an ASCII snapshot of the particle cloud (yes, really — enjoy it) followed
by a `Results` block:

```bash
cat gravity-*.out
```

```
 ── Results ─────────────────────────────────────────────────────────────
  Wall time       84.15s
  Particle-steps  3,000,000
  Interactions    90,000,000,000
  Throughput      0.04M particle-steps/s
  Force rate      1.07G interactions/s
  Energy drift    8.70e-07
  Virial ratio    0.9162  (equilibrium: 1.0)
```

The two numbers you need are **Wall time** (here `84.15`) and
**Force rate** (here `1.07`) — that's the interactions-per-second
throughput. Ignore the `Throughput` line in particle-steps/s; we're
collecting the Force rate.

**Standard error (`gravity-4815162.err`)** — the meta details: which variables
you ran with and which cores/NUMA domains the job was pinned to:

```bash
cat gravity-*.err
```

You need **four values** for your submission: your `NUM_CORES`, your
`CHUNK_SIZE`, the **Wall time** in seconds, and the **Force rate**
in G interactions/s.

## 6. Submit your data point

Enter your results in the Google Form (this is also your workshop attendance
credit — one submission per run, and yes, you can run more than one!):

> **FORM_URL_HERE**

The form asks for: your Anvil username, `NUM_CORES`, `CHUNK_SIZE`, walltime,
and interactions per second. As submissions come in, the scaling chart on the
screen updates live. Watch where your point lands.

---

## 7. What's actually happening: cores, memory, and NUMA

Here's the part that separates "I ran a job" from "I understand the machine."

### The Anvil CPU

Each Anvil compute node has **two AMD EPYC 7763 processors** ("sockets"),
each with 64 cores — 128 cores total. But memory is not one big pool.
Each socket is divided into **4 NUMA domains** (NUMA = Non-Uniform Memory
Access), each with its own memory controller and its own bank of RAM:

```
                           ╔═══════════════════════╗
  ┌─────────────┐          ║┌─────────────────────┐║▒         ┌────────────┐
  │ PURDUE®RCAC │▒         ║│ NON-UNIFORM MEMORY  │║▒         │ *~~~~~~~~* │▒
  │  PEARC  26  │▒         ║│ ARCHITECTURE OF THE │║▒         │ NUMA  101  │▒
  │  JUL  2026  │▒         ║│ ANVIL SUPERCOMPUTER │║▒         │ *~~~~~~~~* │▒
  └─────────────┘▒         ║└─────────────────────┘║▒         └────────────┘▒
    ▒▒▒▒▒▒▒▒▒▒▒▒▒▒         ╚═══════════════════════╝▒          ▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                             ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒


   ╔═════════ SOCKET 0 ═════════╗             ╔═════════ SOCKET 1 ═════════╗
   ║ ┌─ NUMA 0 ─┐  ┌─ NUMA 1 ─┐ ║▒            ║ ┌─ NUMA 4 ─┐  ┌─ NUMA 5 ─┐ ║▒
   ║ │  CORES   │  │  CORES   │ ║▒            ║ │  CORES   │  │  CORES   │ ║▒
   ║ │   0-15   │  │  16-31   │ ║▒            ║ │  64-79   │  │  80-95   │ ║▒
   ║ └────┬─────┘  └────┬─────┘ ║▒            ║ └────┬─────┘  └────┬─────┘ ║▒
   ║ ┌────┴─────┐  ┌────┴─────┐ ║▒  INFINITY  ║ ┌────┴─────┐  ┌────┴─────┐ ║▒
   ║ │   RAM    │  │   RAM    │ ║▒   FABRIC   ║ │   RAM    │  │   RAM    │ ║▒
   ║ └──────────┘  └──────────┘ ║▒◄──────────►║ └──────────┘  └──────────┘ ║▒
   ║ ┌─ NUMA 2 ─┐  ┌─ NUMA 3 ─┐ ║▒  ▒▒▒▒▒▒▒▒  ║ ┌─ NUMA 6 ─┐  ┌─ NUMA 7 ─┐ ║▒
   ║ │  CORES   │  │  CORES   │ ║▒            ║ │  CORES   │  │  CORES   │ ║▒
   ║ │  32-47   │  │  48-63   │ ║▒            ║ │  96-111  │  │ 112-127  │ ║▒
   ║ └────┬─────┘  └────┬─────┘ ║▒            ║ └────┬─────┘  └────┬─────┘ ║▒
   ║ ┌────┴─────┐  ┌────┴─────┐ ║▒            ║ ┌────┴─────┐  ┌────┴─────┐ ║▒
   ║ │   RAM    │  │   RAM    │ ║▒            ║ │   RAM    │  │   RAM    │ ║▒
   ║ └──────────┘  └──────────┘ ║▒            ║ └──────────┘  └──────────┘ ║▒
   ╚════════════════════════════╝▒            ╚════════════════════════════╝▒
     ▒▒▒▒▒▲▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒              ▒▒▒▒▒▲▒▒▒▒▒▒▒▒▒▒▒▒▒▲▒▒▒▒▒▒▒▒▒
          │                                          │             │
   ┌─────── THE TRAP ────────┐                ╔═════════ THE FIX ══════════╗
   │  DEFAULT: FIRST TOUCH   │▒               ║    NUMACTL --INTERLEAVE    ║▒
   │ ALL PAGES → ONE DOMAIN  │▒    ▒▒▒▒▒▒▒▒   ║  PAGES DEALT ROUND-ROBIN   ║▒
   │      ≈ 0.45G INT/S      │▒    ▒▒▒▒▒▒▒▒   ║      ≈ 0.85G INT/S ✓       ║▒
   └─────────────────────────┘▒               ╚════════════════════════════╝▒
     ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒                 ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒

                                                    N O N - U N I F O R M
```

A core can read *any* RAM on the node, but reading from its **own** NUMA
domain is fast, reading from another domain on the same socket is slower,
and reading across the inter-socket link is slowest. Hence *non-uniform*.

### The trap: first-touch

By default, Linux places each page of memory in the NUMA domain of whichever
core **first writes to it**. Our simulation initializes its particle arrays
from a single process — so *all* the data lands in **one** NUMA domain.

Now imagine 128 cores all hammering one memory controller while seven others
sit idle. The cores aren't the bottleneck; the *memory* is. This is why
"just add more cores" eventually stops working.

### The fix (built into `submit.sh`)

The job script does this for you:

1. It reads your `NUM_CORES` and pins the job to exactly that many cores
   (`taskset`), filling from core 0 upward — so 64 cores means exactly
   socket 0, 16 cores means exactly one NUMA domain.
2. It figures out which NUMA domains those cores live in and tells Linux to
   **interleave** memory across *those domains only*
   (`numactl --interleave=...`). Pages are dealt out round-robin, so every
   local memory controller shares the load — and no data ends up on a socket
   you're not even using.

You don't need to change any of this — but the `.err` file shows you exactly
what it decided, so look at it. On a full node, this trick nearly **doubles**
the throughput. Now you know why.

## 8. Questions to think about while your job runs

Watch the live chart with these in mind — we'll discuss as the results fill in:

1. Does doubling the cores double the throughput? Where does the curve bend,
   and does that happen at a NUMA or socket boundary?
2. Does the best `CHUNK_SIZE` depend on `NUM_CORES`?
3. The big one: one 128-core simulation gets X int/s. Two *independent*
   64-core simulations — one per socket — each get Y. Is 2Y > X?
   If so, what does that tell you about how to schedule work on this machine?

## Quick reference

```bash
sbatch submit.sh            # submit the job
squeue -u $USER             # check your jobs in the queue
scancel <JOBID>             # cancel a job
cat gravity-<JOBID>.out     # simulation output (Wall time + Force rate in Results block)
cat gravity-<JOBID>.err     # run configuration and core/NUMA pinning
```
