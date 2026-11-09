#!/bin/bash
# Dry run of the excited-state foundation pipeline on mock data, on CPUs:
# mock data sets -> sharded HDF5 + statistics -> multi-head training with two
# data-parallel processes -> restart -> fine-tuning from the foundation model
# (continued training, and transfer to a new data set with LoRA).
#   bash foundation/dry_run.sh [work_dir]          (~5 minutes on 2 CPU cores)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mkdir -p "${1:-xmace_fm_dry_run}" && cd "${1:-xmace_fm_dry_run}" && pwd)"
NPROC=${NPROC:-2}
export OMP_NUM_THREADS=1

echo ">> 1. mock data sets"
python "$HERE/make_mock_data.py" "$WORK/mock"

echo ">> 2. HDF5 shards and statistics per data set, training config"
cat > "$WORK/template.yaml" <<YAML
name: xmace_fm_mock
model: ExcitedMACE
seed: 1
device: cpu
default_dtype: float64
hidden_irreps: 16x0e + 16x1o
MLP_irreps: 16x0e
num_interactions: 2
correlation: 2
pair_repulsion: true
excited_adapter_irreps: 8x0e + 8x1o
embedding_specs:
  total_charge: {type: categorical, per: graph, in_dim: 1, emb_dim: 8, num_classes: 5, offset: 2}
  total_spin: {type: categorical, per: graph, in_dim: 1, emb_dim: 8, num_classes: 5}
energy_weight: 1.0
forces_weight: 10.0
nacs_weight: 10.0
socs_weight: 1.0
dipoles_weight: 1.0
batch_size: 4
valid_batch_size: 8
lr: 0.005
max_num_epochs: 3
ema: true
ema_decay: 0.99
restart_latest: true
error_table: EnergyNacsDipoleRMSE
YAML
python "$HERE/prepare_datasets.py" --datasets "$WORK/mock/datasets.yaml" \
    --template "$WORK/template.yaml" --work_dir "$WORK"

echo ">> 3. training with $NPROC data-parallel processes"
torchrun --standalone --nproc_per_node="$NPROC" -m mace.cli.run_train \
    --config="$WORK/train_config.yaml" --distributed --launcher=torchrun

echo ">> 4. restart: two more epochs from the last checkpoint"
torchrun --standalone --nproc_per_node="$NPROC" -m mace.cli.run_train \
    --config="$WORK/train_config.yaml" --distributed --launcher=torchrun \
    --max_num_epochs=5

FM="$WORK/run/xmace_fm_mock.model"
echo ">> 5a. continue training one head of the foundation on additional data"
python -m mace.cli.run_train --name=more_tddft --work_dir="$WORK/finetune_continue" \
    --foundation_model="$FM" --model=ExcitedMACE \
    --heads="{'tddft_singlets': {'train_file': '$WORK/mock/tddft_singlets/test.xyz', 'valid_fraction': 0.3, 'n_energies': 4, 'compute_nacs': True, 'compute_dipoles': True}}" \
    --E0s=foundation --lr=0.001 --batch_size=2 --max_num_epochs=2 --device=cpu --default_dtype=float64

echo ">> 5b. transfer to a new data set (3 singlets, NACs only) with LoRA"
python - "$WORK" <<'PY'
import sys, ase.io
from pathlib import Path
work = Path(sys.argv[1])
frames = ase.io.read(work / "mock/tddft_singlets/train.xyz", ":")
for atoms in frames:
    atoms.info["REF_energy"] = atoms.info["REF_energy"][:3]
    atoms.info["REF_forces"] = atoms.info["REF_forces"][:, :3]
    atoms.info["REF_nacs"] = atoms.info["REF_nacs"][:, [0, 1, 3]]
    atoms.info.pop("REF_dipoles", None)
ase.io.write(work / "new_data.xyz", frames[:20])
PY
python -m mace.cli.run_train --name=new_molecule --work_dir="$WORK/finetune_transfer" \
    --foundation_model="$FM" --foundation_head=tddft_singlets --model=ExcitedMACE \
    --train_file="$WORK/new_data.xyz" --valid_fraction=0.2 --n_energies=3 --compute_nacs \
    --E0s=foundation --lora=True --lora_rank=4 --batch_size=4 --max_num_epochs=2 \
    --device=cpu --default_dtype=float64

echo ">> done: foundation $FM, fine-tuned models in $WORK/finetune_*"
