#!/bin/bash
#SBATCH --job-name=xmace_osc_polar_m
#SBATCH --partition=<gpu partition>
#SBATCH --account=<project>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=180G
#SBATCH --time=2-00:00:00
#SBATCH --array=1-3
#SBATCH --output=slurm-%x-%a-%j.out
#
# Committee of 3 X-MACE models for oscillator strengths only (S0 -> S1..S_n) on
# MACE-POLAR-1-M (LoRA, stage two). One array task per committee member; they differ
# only in the seed (1, 2, 3).
# Needs osc_data/ from:  python tools/prepare_osc_data.py your_data.xyz --out_dir osc_data
# Submit:   sbatch train_committee_osc.slurm
# Resume:   sbatch train_committee_osc.slurm   (--restart_latest continues each run)
set -euo pipefail

SEED=${SLURM_ARRAY_TASK_ID:-1}
DATA=osc_data
BATCH=32                    # frames per GPU
EPOCHS=100
STAGE_TWO=75                # epoch where stage two (lower learning rate) starts
NAME="osc_polar_m_s${SEED}"

module purge                        # your modules and environment:
# module load ...
# source /path/to/venv/bin/activate
cd "$SLURM_SUBMIT_DIR"

N_STATES=$(( $(cat "$DATA/n_osc.txt") + 1 ))   # S0 plus the excited states
NGPU=$(nvidia-smi -L | wc -l)
echo "member seed=$SEED gpus=$NGPU batch/gpu=$BATCH effective batch=$((NGPU * BATCH)) states=$N_STATES"

# Oscillator strengths only: energy and force weights 0 (also in stage two), so no
# forces are computed; E0s from the foundation model (the data has no energies).
torchrun --standalone --nproc_per_node="$NGPU" -m mace.cli.run_train \
  --distributed --launcher=torchrun --device=cuda \
  --name="$NAME" --work_dir="runs_osc/$NAME" --seed="$SEED" \
  --train_file="$DATA/train.xyz" --valid_file="$DATA/valid.xyz" --test_file="$DATA/test.xyz" \
  --model=ExcitedMACE --n_energies="$N_STATES" --compute_oscillator_strengths \
  --foundation_model=polar-1-m --excited_long_range=shared --multiheads_finetuning=False \
  --lora=True --lora_rank=4 --lora_alpha=1 \
  --E0s=foundation \
  --energy_weight=0 --forces_weight=0 --oscillator_strengths_weight=10 \
  --default_dtype=float32 --batch_size="$BATCH" --valid_batch_size=64 \
  --lr=0.001 --max_num_epochs="$EPOCHS" --patience=20 --ema --ema_decay=0.99 \
  --stage_two --start_stage_two="$STAGE_TWO" --swa_lr=1e-4 \
  --swa_energy_weight=0 --swa_forces_weight=0 \
  --error_table=EnergyNacsDipoleMAE --num_workers=4 --restart_latest
