#!/bin/bash
# =====================================================================
#  N-body simulation — batch job for Anvil (Purdue / NSF ACCESS)
#
#  Submit:    sbatch submit.sh
#  Monitor:   squeue -u $USER
#  Follow:    tail -f gravity-<jobid>.out
#  View:      less -R gravity-<jobid>.out     (-R renders the colors)
# =====================================================================
#SBATCH --account=YOUR_ALLOCATION   # <-- EDIT: your ACCESS allocation
#SBATCH --partition=wholenode       # one full CPU node, exclusive
#SBATCH --nodes=1                   # single node (the code is one-node)
#SBATCH --ntasks=1                  # one process...
#SBATCH --cpus-per-task=128         # ...using all 128 cores
#SBATCH --time=00:30:00             # 30 minutes is plenty
#SBATCH --job-name=gravity
#SBATCH --output=gravity-%j.out     # %j expands to the job ID
#SBATCH --error=gravity-%j.err      # %j expands to the job ID


# Ensure application environment
source /etc/bashrc
module reset &>/dev/null
module --force purge &>/dev/null
module load gcc numactl


# Simulation parameters
NUM_CORES=32      # pick from: 2, 4, 8, 16, 32, 64, 128
CHUNK_SIZE=2048   # pick from: 512, 1024, 2048, 4096, 8192, 16384


# Simulation parameters (FIXED - DO NOT CHANGE THESE)
SCENARIO=galaxy
PARTICLES=30000
STEPS=100


# uv lives in ~/.local/bin; install it on first use so students can
# submit this job without any manual setup. (Anvil compute nodes have
# outbound network access; the install is a one-time ~15s download.)
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found — installing to ~/.local/bin ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh || {
        echo "ERROR: uv install failed (no network?)" >&2
        exit 1
    }
fi


# NUMA domain detection
NUMA=$(lscpu -p=CPU,NODE | grep -v '^#' | awk -F, -v ncpu="$NCPU" '$1 < ncpu { seen[$2]=1 } END { for (k in seen) s = s (s==""?"":",") k; print s }')


# The script is self-contained: uv supplies Python, NumPy, and JAX from
# its cache (populated by your first run on the login node).
echo "RUNNING SIMULATION:" >&2
echo "exec taskset -c 0-$((NUM_CORES-1)) numactl --interleave=$NUMA -- \\" >&2
echo "    ./sim.py $SCENARIO --particles=$PARTICLES --steps=$STEPS --chunk=$CHUNK_SIZE --plain" >&2
echo "-----------------------" >&2
exec taskset -c 0-$((NUM_CORES-1)) numactl --interleave=$NUMA -- ./sim.py $SCENARIO --particles=$PARTICLES --steps=$STEPS --chunk=$CHUNK_SIZE --plain

