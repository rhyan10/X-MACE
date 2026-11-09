#!/bin/bash
# Excited-state foundation model on several nodes with several GPUs each.
# Data parallel: every GPU gets batch_size frames of each step, DDP averages the
# gradients (the loss is that of the whole batch). Edit the #SBATCH lines and the
# paths, then: sbatch foundation/train.sbatch
#
#SBATCH --job-name=xmace-fm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4          # one task per GPU
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
WORK=${WORK:-/scratch/$USER/xmace_fm}      # --work_dir of prepare_datasets.py
# module load ... / conda activate xmace

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))

# one process per GPU; --launcher=slurm reads the rank from SLURM's variables.
# The config has restart_latest: true, so resubmitting this script continues from
# the last checkpoint.
srun python -m mace.cli.run_train \
    --config="${WORK}/train_config.yaml" \
    --distributed --launcher=slurm --device=cuda

# On a single node, torchrun works as well:
#   torchrun --standalone --nproc_per_node=4 -m mace.cli.run_train \
#       --config="${WORK}/train_config.yaml" --distributed --launcher=torchrun
