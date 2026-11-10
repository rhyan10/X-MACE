#!/bin/bash
# All commands of tutorial 04_foundation_models_and_finetuning.ipynb.
# Run from tutorials/xmace/work/foundations after the notebook's data cell (or with
# your own small.xyz, ground.xyz, replay.xyz, small_plus_extra.xyz, singlets.xyz).
# Usage: DEVICE=cuda bash ../../04_commands.sh
set -e
DEVICE=${DEVICE:-cpu}

# ---- 2. The baseline: training from scratch --------------------------------
mace_run_train \
    --name=scratch \
    --train_file=small.xyz --valid_fraction=0.15 --test_file=../so2_test.xyz \
    --model=ExcitedMACE --states="4 0 3" \
    --compute_nacs --compute_socs --compute_dipoles \
    --E0s=average \
    --r_max=5.0 --hidden_irreps="32x0e + 32x1o" --MLP_irreps=16x0e \
    --num_interactions=2 --correlation=2 \
    --batch_size=10 --max_num_epochs=15 --lr=0.005 --seed=1 \
    --energy_weight=10 --forces_weight=100 --nacs_weight=10 --socs_weight=1 --dipoles_weight=1 \
    --device=$DEVICE --default_dtype=float64 \
    > scratch.out 2>&1
grep -A5 "Error-table on TEST" logs/scratch_run-1.log | tail -n 5


# ---- MACE-OFF23 ------------------------------------------------------------
mace_run_train \
    --name=mace_off_small \
    --foundation_model=small_off \
    --train_file=small.xyz --valid_fraction=0.15 --test_file=../so2_test.xyz \
    --model=ExcitedMACE --states="4 0 3" \
    --compute_nacs --compute_socs --compute_dipoles \
    --E0s=average \
    --batch_size=10 --max_num_epochs=15 --lr=0.005 --seed=1 \
    --energy_weight=10 --forces_weight=100 --nacs_weight=10 --socs_weight=1 --dipoles_weight=1 \
    --device=$DEVICE --default_dtype=float64 \
    > mace_off_small.out 2>&1
grep -A5 "Error-table on TEST" logs/mace_off_small_run-1.log | tail -n 5


# ---- MACE-MP-0 (and MPA-0, OMAT-0, MATPES) ---------------------------------
mace_run_train \
    --name=mace_mp_small \
    --foundation_model=small --multiheads_finetuning=False \
    --train_file=small.xyz --valid_fraction=0.15 --test_file=../so2_test.xyz \
    --model=ExcitedMACE --states="4 0 3" \
    --compute_nacs --compute_socs --compute_dipoles \
    --E0s=average \
    --batch_size=10 --max_num_epochs=15 --lr=0.005 --seed=1 \
    --energy_weight=10 --forces_weight=100 --nacs_weight=10 --socs_weight=1 --dipoles_weight=1 \
    --device=$DEVICE --default_dtype=float64 \
    > mace_mp_small.out 2>&1
grep -A5 "Error-table on TEST" logs/mace_mp_small_run-1.log | tail -n 5


# ---- MACE-MH-1 -------------------------------------------------------------
mace_run_train \
    --name=mace_mh1_omol \
    --foundation_model=mh-1 --foundation_head=omol \
    --train_file=small.xyz --valid_fraction=0.15 --test_file=../so2_test.xyz \
    --model=ExcitedMACE --states="4 0 3" \
    --compute_nacs --compute_socs --compute_dipoles \
    --E0s=average \
    --batch_size=10 --max_num_epochs=15 --lr=0.001 --seed=1 \
    --energy_weight=10 --forces_weight=100 --nacs_weight=10 --socs_weight=1 --dipoles_weight=1 \
    --device=$DEVICE --default_dtype=float64 \
    > mace_mh1_omol.out 2>&1
grep -A5 "Error-table on TEST" logs/mace_mh1_omol_run-1.log | tail -n 5


# ---- MACE-POLAR-1 ----------------------------------------------------------
mace_run_train \
    --name=mace_polar_1_s \
    --foundation_model=polar-1-s \
    --train_file=small.xyz --valid_fraction=0.15 --test_file=../so2_test.xyz \
    --model=ExcitedMACE --states="4 0 3" \
    --compute_nacs --compute_socs --compute_dipoles \
    --E0s=average --excited_long_range=shared \
    --batch_size=10 --max_num_epochs=15 --lr=0.005 --seed=1 \
    --energy_weight=10 --forces_weight=100 --nacs_weight=10 --socs_weight=1 --dipoles_weight=1 \
    --device=$DEVICE --default_dtype=float64 \
    > mace_polar_1_s.out 2>&1
grep -A5 "Error-table on TEST" logs/mace_polar_1_s_run-1.log | tail -n 5


# ---- Frozen layers ---------------------------------------------------------
mace_run_train \
    --name=off_freeze \
    --foundation_model=small_off \
    --freeze=5 \
    --train_file=small.xyz --valid_fraction=0.15 --test_file=../so2_test.xyz \
    --model=ExcitedMACE --states="4 0 3" \
    --compute_nacs --compute_socs --compute_dipoles \
    --E0s=average \
    --batch_size=10 --max_num_epochs=15 --lr=0.005 --seed=1 \
    --energy_weight=10 --forces_weight=100 --nacs_weight=10 --socs_weight=1 --dipoles_weight=1 \
    --device=$DEVICE --default_dtype=float64 \
    > off_freeze.out 2>&1
grep "Parameters:" logs/off_freeze_run-1.log | cut -d" " -f4-
grep -A5 "Error-table on TEST" logs/off_freeze_run-1.log | tail -n 5


# ---- A smaller learning rate for the backbone ------------------------------
mace_run_train \
    --name=off_backbone_lr \
    --foundation_model=small_off \
    --lr_params_factors='{"embedding_lr_factor": 0.1, "interactions_lr_factor": 0.1, "products_lr_factor": 0.1, "readouts_lr_factor": 0.1}' \
    --train_file=small.xyz --valid_fraction=0.15 --test_file=../so2_test.xyz \
    --model=ExcitedMACE --states="4 0 3" \
    --compute_nacs --compute_socs --compute_dipoles \
    --E0s=average \
    --batch_size=10 --max_num_epochs=15 --lr=0.005 --seed=1 \
    --energy_weight=10 --forces_weight=100 --nacs_weight=10 --socs_weight=1 --dipoles_weight=1 \
    --device=$DEVICE --default_dtype=float64 \
    > off_backbone_lr.out 2>&1
grep -A5 "Error-table on TEST" logs/off_backbone_lr_run-1.log | tail -n 5


# ---- LoRA ------------------------------------------------------------------
mace_run_train \
    --name=off_lora \
    --foundation_model=small_off \
    --lora=True --lora_rank=4 --lora_alpha=1 \
    --train_file=small.xyz --valid_fraction=0.15 --test_file=../so2_test.xyz \
    --model=ExcitedMACE --states="4 0 3" \
    --compute_nacs --compute_socs --compute_dipoles \
    --E0s=average \
    --batch_size=10 --max_num_epochs=15 --lr=0.005 --seed=1 \
    --energy_weight=10 --forces_weight=100 --nacs_weight=10 --socs_weight=1 --dipoles_weight=1 \
    --device=$DEVICE --default_dtype=float64 \
    > off_lora.out 2>&1
grep "LoRA" logs/off_lora_run-1.log | cut -d" " -f4- | head -3
grep -A5 "Error-table on TEST" logs/off_lora_run-1.log | tail -n 5


# ---- Multi-head fine-tuning with replay and pseudolabels -------------------
mace_run_train \
    --name=off_replay \
    --foundation_model=small_off \
    --multiheads_finetuning=True --pt_train_file=replay.xyz \
    --pseudolabel_replay=True --weight_pt_head=1.0 --force_mh_ft_lr=True \
    --train_file=small.xyz --valid_fraction=0.15 --test_file=../so2_test.xyz \
    --model=ExcitedMACE --states="4 0 3" \
    --compute_nacs --compute_socs --compute_dipoles \
    --E0s=foundation \
    --batch_size=10 --max_num_epochs=15 --lr=0.005 --seed=1 \
    --energy_weight=10 --forces_weight=100 --nacs_weight=10 --socs_weight=1 --dipoles_weight=1 \
    --device=$DEVICE --default_dtype=float64 \
    > off_replay.out 2>&1
grep "Using heads\|pseudolabels to" logs/off_replay_run-1.log | cut -d" " -f4-
grep -A5 "Error-table on TEST" logs/off_replay_run-1.log | tail -n 5


# ---- An extra ground-state head, from a YAML config file -------------------
cat > heads.yaml <<'EOF'
heads:
  excited:
    train_file: small.xyz
    valid_fraction: 0.15
    test_file: ../so2_test.xyz
  ground:
    train_file: ground.xyz
    valid_fraction: 0.15
    ground_state: true
EOF
mace_run_train \
    --name=off_ground_head \
    --config=heads.yaml \
    --foundation_model=small_off \
    --model=ExcitedMACE --states="4 0 3" \
    --compute_nacs --compute_socs --compute_dipoles \
    --E0s=average \
    --batch_size=10 --max_num_epochs=15 --lr=0.005 --seed=1 \
    --energy_weight=10 --forces_weight=100 --nacs_weight=10 --socs_weight=1 --dipoles_weight=1 \
    --device=$DEVICE --default_dtype=float64 \
    > off_ground_head.out 2>&1
grep "Head excited:\|Head ground:" logs/off_ground_head_run-1.log | cut -d" " -f4- | sort -u
grep -A5 "Error-table on TEST" logs/off_ground_head_run-1.log | tail -n 5


# ---- The autoencoder model on a foundation ---------------------------------
mace_run_train \
    --name=off_autoencoder \
    --foundation_model=small_off \
    --train_file=small.xyz --valid_fraction=0.15 --test_file=../so2_test.xyz \
    --model=AutoencoderExcitedMACE --num_permutational_invariant=16 \
    --states="4 0 3" \
    --compute_nacs --compute_socs --compute_dipoles \
    --E0s=average \
    --batch_size=10 --max_num_epochs=15 --lr=0.005 --seed=1 \
    --energy_weight=10 --forces_weight=100 --nacs_weight=10 --socs_weight=1 --dipoles_weight=1 \
    --device=$DEVICE --default_dtype=float64 \
    > off_autoencoder.out 2>&1
grep -A5 "Error-table on TEST" logs/off_autoencoder_run-1.log | tail -n 5


# ---- State-specific electrostatics with MACE-POLAR-1 -----------------------
mace_run_train \
    --name=polar_state \
    --foundation_model=polar-1-s \
    --excited_long_range=state --excited_charge_smearing=1.0 \
    --train_file=small.xyz --valid_fraction=0.15 --test_file=../so2_test.xyz \
    --model=ExcitedMACE --states="4 0 3" \
    --compute_nacs --compute_socs --compute_dipoles \
    --E0s=average \
    --batch_size=10 --max_num_epochs=15 --lr=0.005 --seed=1 \
    --energy_weight=10 --forces_weight=100 --nacs_weight=10 --socs_weight=1 --dipoles_weight=1 \
    --device=$DEVICE --default_dtype=float64 \
    > polar_state.out 2>&1
grep -A5 "Error-table on TEST" logs/polar_state_run-1.log | tail -n 5


# ---- 5. Fine-tuning a fine-tuned model on additional data ------------------
mace_run_train \
    --name=continued \
    --foundation_model=mace_off_small.model \
    --train_file=small_plus_extra.xyz \
    --valid_fraction=0.15 --test_file=../so2_test.xyz \
    --model=ExcitedMACE --states="4 0 3" \
    --compute_nacs --compute_socs --compute_dipoles \
    --E0s=average \
    --batch_size=10 --max_num_epochs=10 --lr=0.001 --seed=1 \
    --energy_weight=10 --forces_weight=100 --nacs_weight=10 --socs_weight=1 --dipoles_weight=1 \
    --device=$DEVICE --default_dtype=float64 \
    > continued.out 2>&1
grep -o "Continuing from the excited-state foundation model.*" logs/continued_run-1.log | head -1
echo "before:"; grep -A5 "Error-table on TEST" logs/mace_off_small_run-1.log | tail -n 2 | head -n 1
echo "after: "; grep -A5 "Error-table on TEST" logs/continued_run-1.log | tail -n 2 | head -n 1


# ---- Transfer to new states ------------------------------------------------
mace_run_train \
    --name=singlets \
    --foundation_model=continued.model \
    --train_file=singlets.xyz --valid_fraction=0.15 \
    --model=ExcitedMACE --n_energies=4 \
    --compute_nacs --compute_dipoles \
    --lora=True --lora_rank=4 \
    --E0s=average \
    --batch_size=10 --max_num_epochs=10 --lr=0.005 --seed=1 \
    --energy_weight=10 --forces_weight=100 --nacs_weight=10 --dipoles_weight=1 \
    --device=$DEVICE --default_dtype=float64 \
    > singlets.out 2>&1
grep -o "Transfer from the excited-state foundation model.*" logs/singlets_run-1.log | cut -c1-120 | head -1
grep -A5 "Error-table on TRAIN" logs/singlets_run-1.log | tail -n 3

