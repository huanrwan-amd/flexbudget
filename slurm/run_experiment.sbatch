#!/bin/bash
# Generic SLURM + pyxis/enroot launcher for a FlexBudget experiment on ROCm.
#
# Usage:  sbatch --export=ALL,EXPERIMENT=e0_parity_gate slurm/run_experiment.sbatch
#
# Adapt the SBATCH block to your cluster. The GRES string below selects MI300X
# by *typed GRES tag*, which is how these nodes are addressed on our cluster --
# yours may simply use `--gres=gpu:1`.
#SBATCH --job-name=flexbudget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT="${EXPERIMENT:?set EXPERIMENT, e.g. e0_parity_gate}"
CONTAINER="${CONTAINER:-rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.14_pytorch_2.10.0}"

mkdir -p "$REPO"/{results,logs,.cache/{triton,inductor,hf,mpl,home}}

srun --container-image="$CONTAINER" \
     --container-mounts="$REPO:/work" \
     --container-workdir=/work \
     bash -lc '
        set -euo pipefail
        # $HOME is READ-ONLY inside the container. Without these redirects the
        # Triton JIT cache write fails with OSError errno 30 and nothing
        # compiles -- this is the single most common bring-up failure.
        export TRITON_CACHE_DIR=/work/.cache/triton \
               TORCHINDUCTOR_CACHE_DIR=/work/.cache/inductor \
               HF_HOME=/work/.cache/hf \
               MPLCONFIGDIR=/work/.cache/mpl \
               HOME=/work/.cache/home \
               OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} \
               RESULTS_DIR=/work/results \
               PYTHONPATH=/work
        python3 /work/experiments/'"$EXPERIMENT"'.py
     '
