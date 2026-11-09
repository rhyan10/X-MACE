# Training an excited-state foundation model

One X-MACE model trained on many excited-state data sets at once, to be fine-tuned
later on new molecules. Each data set is a **head**: it keeps its own number of
states, multiplicities, couplings, level of theory, E0s and per-state shifts, while
all heads share one backbone (and the equivariant adapter layer that feeds the
couplings). This folder has everything to go from your data sets to a trained model:

| file | what it does |
|---|---|
| `datasets_example.yaml` | the list of data sets and their label layouts — copy and edit |
| `prepare_datasets.py` | runs `mace_prepare_data` for every data set (sharded HDF5, statistics) and writes the training config |
| `train_template.yaml` | architecture, loss, optimiser, embeddings — the settings of the run |
| `train.sbatch` | multi-node, multi-GPU training with SLURM (restartable) |
| `make_mock_data.py`, `dry_run.sh` | the whole pipeline on small mock data, on CPUs, to check the setup |

## 0. Dry run first

```bash
bash foundation/dry_run.sh /tmp/xmace_fm_dry_run       # ~5 min on 2 CPU cores
```

It builds four mock data sets (singlets + triplets with SOCs; singlets with NACs and
dipoles; energies only; doublets + quartet, with different elements, charges and
spins), prepares them, trains with two data-parallel processes, restarts from the
checkpoint, and fine-tunes the result twice (continued training of one head, and
transfer to a new data set with LoRA). If this runs, your installation is ready.

## 1. Describe the data sets

Every data set is an extended XYZ file in the X-MACE layout (`XMACE.md`, "Data"):
`REF_energy` `[S]` in eV, `REF_forces` `[n_atoms, S, 3]` in eV/Å, and as available
`REF_nacs`, `REF_socs`, `REF_dipoles`. Labels a data set lacks are simply left out
(weight 0), so a public set with energies only is fine. Molecular charge and spin go
into `atoms.info["total_charge"]` and `atoms.info["total_spin"]`.

Copy `datasets_example.yaml` and list your data sets. Per data set:

* `n_energies` or `states` (SHARC order, e.g. `[3, 0, 2]` = 3 singlets + 2 triplets);
* `compute_nacs`, `compute_socs`, `compute_dipoles` — what to learn from it;
* label keys if they differ (`socs_key`, ...);
* `weight` — loss weight of its configurations, to balance large and small sets.

Rules of thumb:

* **One head per level of theory and state set.** S1 of a CASSCF set and S1 of a
  TDDFT set are different quantities; separate heads let the backbone share the
  chemistry while each head keeps its own reference. Merge files only when they
  come from the same method and active space.
* **One SOC unit per head** (e.g. cm⁻¹). The SOC loss is in the label unit.
* **Hold out test sets** (`test_file`) per data set; errors are reported per head.
* `elements` is the union over all data sets: the model's element table.

## 2. Prepare

```bash
python foundation/prepare_datasets.py --datasets datasets.yaml \
    --template foundation/train_template.yaml --work_dir /scratch/$USER/xmace_fm
```

Each data set becomes `work_dir/h5/<name>/{train,val}/*.h5` (`num_process` shards)
plus `statistics.json` (E0s, per-state shifts, scale, average number of neighbours).
`work_dir/train_config.yaml` is the template plus one head per data set. Add data
sets later with `--only new_set` (the others are not reprocessed).

## 3. Choose the settings

`train_template.yaml` is a starting point for a medium-size model:

* **Backbone.** `ScaleShiftMACE` (default) for most data. For charge-transfer or
  ionic states use `excited_backbone: MACELES` or `PolarMACE` and consider
  `excited_long_range: state` (every state gets its own charges).
* **Start from a ground-state foundation** instead of from scratch: add
  `foundation_model: mace_omol` (or `small_off`, `polar-1-s`, a path, ...). Every
  state then starts on that model's ground-state surface, which helps a lot with
  little excited-state data. MACE-OMOL already embeds charge and spin. With
  `multiheads_finetuning: true` and `pt_train_file:` (ground-state replay data) the
  ground state of the foundation is kept while the excited states are learned.
* **Size.** `hidden_irreps: 128x0e + 128x1o`, 2 layers, `correlation: 3` is a
  reasonable first model; `256x0e + 256x1o` for > 10⁶ configurations. The adapter
  (`excited_adapter_irreps`) feeds all couplings: 64 channels is plenty.
* **Charge and spin embeddings** (`embedding_specs`): `num_classes`/`offset` must
  cover every charge and spin in the data.
* **Loss weights.** Forces dominate dynamics; keep `forces_weight` ≥ `energy_weight`.
  Watch the initial per-head loss terms and rebalance `nacs_weight`/`socs_weight`
  so no term dominates by orders of magnitude.
* **Stage Two** (`swa`, `start_swa`) for the last quarter of the epochs, with a
  higher energy weight.

## 4. Train

```bash
sbatch foundation/train.sbatch        # edit the #SBATCH lines and WORK first
```

One process per GPU; `batch_size` is per GPU (effective batch = batch_size × GPUs).
The loss is that of the whole effective batch (also the RMSE-type coupling terms),
and the statistics are taken over all GPUs. `restart_latest: true` makes a
resubmitted job continue from its last checkpoint, so long runs can be split into
jobs that fit the queue's time limit.

Follow the run in `work_dir/run/logs/` (per-head errors every epoch) and plot it:

```bash
mace_plot_train --path work_dir/run/results --heads sharc_casscf,tddft_singlets
```

## 5. Evaluate and release

```bash
mace_eval_configs --configs test.xyz --model xmace_fm.model --head tddft_singlets --output pred.xyz
mace_select_head -n tddft_singlets -o xmace_fm_tddft.model xmace_fm.model   # one head only
```

The full multi-head model is the foundation; single-head copies are convenient for
dynamics (SHARC: `head` in `MACE.template` works with the full model too).

## 6. Fine-tune from it

`--foundation_model=xmace_fm.model` accepts an excited-state model:

* **Continue** — the run's heads have the layouts of foundation heads (same name, or
  `--foundation_head`): the model keeps *all* weights, shifts and scales and simply
  trains on the new data. Use this for more data of the same kind, with a smaller
  learning rate (`--lr=1e-3` or less).
* **Transfer** — new states, couplings or methods: backbone and adapter keep their
  weights; the readouts of the new heads start so that every state sits on the
  backbone energy, with shifts fitted to the data. Combine with `--lora=True`
  (low-rank updates of the backbone) or `--freeze=N` for small data sets.

```bash
mace_run_train --foundation_model=xmace_fm.model --foundation_head=tddft_singlets \
    --model=ExcitedMACE --train_file=new.xyz --n_energies=3 --compute_nacs \
    --E0s=foundation --lora=True --lr=1e-3 ...
```

Continue applies when *every* head of the run matches a foundation head; as soon as
one head is new, all heads are built as in transfer (backbone and adapter kept,
readouts new). To keep the backbone general while fine-tuning on a new data set,
add a sample of the foundation's data as extra heads (replay): their readouts are
relearned quickly, and they stop the shared backbone from drifting towards the new
data only.
