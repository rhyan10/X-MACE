# Excited states with MACE (X-MACE)

X-MACE extends MACE to several electronic states. One network predicts, for S states:

| quantity | output | shape | how |
|---|---|---|---|
| energies | `energy` | `[n_graphs, S]` | the backbone's ground-state energy plus per-state invariant readouts |
| forces | `forces` | `[n_atoms, S, 3]` | −∂E_s/∂r for every state |
| non-adiabatic couplings | `nacs` | `[n_atoms, P, 3]` | per-atom equivariant vectors (l = 1); smooth NACs, NAC × ΔE |
| spin–orbit couplings | `socs` | `[n_graphs, L]` | sums of per-atom invariant scalars (`--soc_mode=free`) |
| spin–orbit coupling vectors | `soc_vectors` | `[n_graphs, Q, 3]` | sums of per-atom axial vectors (`--soc_mode=spin_orbit`), see [Spin multiplicities](#spin-multiplicities) |
| permanent and transition dipoles | `dipoles` | `[n_graphs, D, 3]` | sums of per-atom equivariant vectors |
| oscillator strengths | `oscillator_strengths` | `[n_graphs, O]` | from the ground state to states 1..O: scale · softplus(sum of per-atom invariant scalars), so f ≥ 0 |

P = S(S−1)/2 state pairs in the order (0,1), (0,2), …, (1,2), …. By default L = P and D = S + P (S permanent dipoles, then one transition dipole per pair). With `--states`, P counts only pairs of the same multiplicity.

There are two models:

* **`ExcitedMACE`** predicts all states directly.
* **`AutoencoderExcitedMACE`** predicts a permutation-invariant latent vector. A decoder turns it into a symmetric tridiagonal S × S matrix, and the energies are its eigenvalues. This keeps the states ordered and their crossings smooth near conical intersections. With several multiplicities, the matrix has one block per multiplicity, the encoder tags each energy with its multiplicity, and the energy shift is shared within a multiplicity. The states are then ordered within each multiplicity, as SHARC's MCH states are. During training an encoder (a deep set over the states) maps the reference energies to the latent space. The loss asks the decoder to reconstruct them and the network to reproduce the encoded vector.

Both live in `mace/modules/excited.py` and work with the rest of MACE: the CLI, checkpoints, Stage Two, EMA, multi-head fine-tuning, LoRA, foundation models, TorchScript export, ASE and SHARC.

**Architecture.** Both models sit on top of a ground-state MACE model of any class, the *backbone*: `ScaleShiftMACE` (default), `MACE`, `MACELES`, `PolarMACE` or `MagneticScaleShiftMACE`, built from scratch or taken from a foundation model. For a frame of head h,

  E_{h,s} = E_backbone,h + scale_h · Σ_layers readout_{h,s} + n_atoms · shift_{h,s}  [+ state electrostatics]

The energy readouts start at zero, so before training every state lies on the backbone's surface (a foundation model's ground state, exactly). An *adapter* layer (one MACE interaction and product block with l = 1 features) on top of the backbone's last features feeds the coupling readouts, so NACs, dipoles and SOC vectors also work with scalar-only foundation models (MACE-MP small, MACE-OFF, MACE-POLAR-1, ...). The readouts read the features of every backbone layer and of the adapter. See [Backbones, heads, foundation models and LoRA](#backbones-heads-foundation-models-and-lora).

**Tutorials.** `tutorials/xmace/` has eight Jupyter notebooks for new users, from the physics (smooth NACs, phases, multiplicities, Wigner–Eckart SOCs, equivariance) through building a data set with SHARC, training and validation, to surface-hopping dynamics with SHARC, to foundation models and fine-tuning strategies (tutorial 7), and to oscillator strengths (tutorial 8). They use SO2 with SHARC's LVC model as the reference.

## Data

Each frame of an extended XYZ file carries its labels in `atoms.info`, as in X-MACE:

| key (CLI option) | shape | |
|---|---|---|
| `REF_energy` (`--energy_key`) | `[S]` or `[1, S]` | eV |
| `REF_forces` (`--forces_key`) | `[n_atoms, S, 3]` (also `[S, n_atoms, 3]`, or `[n_atoms, 3S]` in `atoms.arrays`) | eV/Å |
| `REF_nacs` (`--nacs_key`) | `[n_atoms, P, 3]` (also `[P, n_atoms, 3]`) | raw or smooth, see below |
| `REF_socs` (`--socs_key`) | `[L]` real | any unit |
| `REF_dipoles` (`--dipoles_key`) | `[D, 3]` | any unit |
| `REF_oscillator_strengths` (`--oscillator_strengths_key`) | `[O]`, f(S0→S1), …, f(S0→S_O) (also `[O + 1]` with a leading 0 for S0 itself) | dimensionless |

* **Smooth NACs.** NACs diverge as 1/ΔE at conical intersections, so the model learns NAC × |E_j − E_i| instead. `--nacs_format=raw` multiplies the NACs in the data by the energy gap of their pair, using the reference energies of the same frame. `--nacs_format=smooth` uses them as they are. The default, `auto`, picks `smooth` when the NAC key contains "smooth" (e.g. `REF_smooth_nacs`) and `raw` otherwise. The log says which was used. The ASE calculator returns both the smooth NACs and the physical NACs (smooth NACs / ΔE).
* **Missing labels.** A frame may lack any label. The missing label is filled with zeros and gets weight 0, so that frame is skipped for that term only. `config_weight` and `config_<property>_weight` in `atoms.info` set per-frame weights, as in MACE.
* **Arbitrary phases.** The sign of a coupling between two states is arbitrary in quantum-chemistry data. The losses and error metrics use the better of the two signs. For NACs this is chosen per configuration and state pair, for transition dipoles per configuration and pair, and for SOCs once per configuration (X-MACE's free SOC layout). Labels need no consistent phases.
* **E0s.** `--E0s=average` fits the isolated-atom energies to the mean energy over the states, using only frames that have energies.

The training log lists how many frames carry each label.

## Spin multiplicities

`--states` gives the spatial states per multiplicity, in SHARC's order: `--states="3 0 2"` is three singlets and two triplets, `--states="0 2 0 1"` two doublets and a quartet, `--states="2 0 2 0 1"` adds a quintet. The model's states are listed in that order (S0, S1, S2, T1, T2, …), and `--n_energies` is optional (it must equal the total). Any multiplicity SHARC supports can be used.

The symmetry of the spin states is part of the model, not something it has to learn:

* **Degenerate M_S components.** The model predicts one energy, one set of forces and one permanent dipole per spatial state. They are shared by all its M_S components, so a triplet's three components are exactly degenerate.
* **NACs and transition dipoles** couple states of the same multiplicity only (and, in SHARC, only equal M_S). The model predicts one NAC and one transition dipole per such pair, so `--nac_num` defaults to the number of same-multiplicity pairs. Labels may give every pair of states (X-MACE layout); the coupled pairs are picked out.
* **Spin–orbit couplings (`--soc_mode=spin_orbit`, the default when there are spin-orbit-coupled multiplicities).** By the Wigner–Eckart theorem, all M_S-resolved SOC matrix elements between two spatial states follow from one complex vector of reduced matrix elements, which is i·**a** for real wavefunctions. The model predicts **a**, an axial vector, for each pair of spatial states the selection rules allow: ΔS ∈ {0, ±1}, not singlet–singlet, and not within one spatial state. Singlet–doublet and singlet–quintet pairs, for example, have no coupling. The SOC matrix is then built exactly:
  H[(I,M),(J,M′)] = (−1)^q A_q c(S,S′) ⟨S′ M′ 1 −q | S M⟩, with q = M′ − M, A = i·**a** in spherical components, and c(S,S) = √(S(S+1)), else 1.
  So the predicted SOC matrix is always Hermitian. It obeys the ΔM_S selection rules, and when the molecule rotates it rotates with the spins, H(R**x**) = U(R) H(**x**) U(R)†. Under inversion it is unchanged, because the coupling vectors are axial (built from 1o ⊗ 1o → 1e features).
* **Phases.** The sign of each spatial state is arbitrary. NACs, transition dipoles and SOC vectors are compared with the best sign per configuration and pair.
* **Polar and axial couplings (`--coupling_parity`).** An improper operation (inversion, reflection) can also flip the signs of the states, so a coupling vector may behave as a polar or as an axial vector. This matters for planar molecules, where every state is symmetric or antisymmetric under reflection through the plane. Any polar function of the positions then lies in the plane and any axial one is perpendicular to it. But NACs and transition dipoles between states of opposite reflection symmetry (e.g. the out-of-plane transition dipole of an n→π* excitation) are perpendicular to the plane, and SOC vectors between states of equal symmetry lie in it. With the default `--coupling_parity=both`, every coupling vector has a polar and an axial part, so these couplings can be learned, and the model stays exactly equivariant under rotations. `--coupling_parity=fixed` makes NACs and transition dipoles purely polar and SOC vectors purely axial (also equivariant under inversion). For SO2 (a triatomic, so always planar) all SOC vectors lie in the molecular plane, and `fixed` cannot learn any of them. Permanent dipoles are always polar.

**SOC labels** are M_S-resolved matrices in SHARC order (for each multiplicity, for M_S = −S…S, for each state), in any unit, as:

* a complex `[nm, nm]` array,
* `[nm, nm, 2]` or `[2, nm, nm]` (re, im),
* the upper triangle as nm(nm−1)/2 complex values,
* or nm(nm−1) real values with re and im interleaved, which is X-MACE's layout and what `mace_eval_configs` writes.

The coupling vectors are fitted to each matrix by least squares. If a matrix is not of Wigner–Eckart form (wrong state order, M_S basis or phase convention), the log warns with the largest relative residual. The SOC loss is weighted per pair so that it measures the M_S-resolved matrix elements. The SOC errors in the error table are those of the real and imaginary parts of the allowed matrix elements.

```bash
mace_run_train --model=ExcitedMACE --states="2 0 2 0 1" --compute_nacs --compute_socs \
  --compute_dipoles --socs_key=REF_socs --hidden_irreps='64x0e + 64x1o' ...
# log: ExcitedMACE: 5 states (2 singlets, 2 triplets, 1 quintet), NACs=2 (raw),
#      SOCs=7 coupling vectors (spin_orbit), dipoles=7, ...
```

**Data from SHARC.** `mace.data.sharc_labels` turns SHARC's M_S-resolved quantities (`h`, `grad`, `nacdr`, `dm` of a `QM.out`, in atomic units) into X-MACE labels in eV and Å. It checks that every multiplet is degenerate (i.e. `h` is in the MCH basis). `atoms_from_sharc` returns an `ase.Atoms` frame with the default keys, ready for `ase.io.write`:

```python
from qmout import QMout                       # SHARC's lib directory on PYTHONPATH
from mace.data.sharc_labels import atoms_from_sharc
q = QMout(filepath="QM.out")
atoms = atoms_from_sharc(symbols, coords_bohr, [4, 0, 3], q["h"], q["grad"], q["nacdr"], q["dm"], soc_unit="cm-1")
```

The SOCs of SHARC's reference outputs from MOLCAS, MOLPRO and ADF (in SHARC's `tests/RESULTS`) fit the Wigner–Eckart form to machine precision. The SOC matrices of SHARC's SO2 LVC example and of its COLUMBUS test are real and do not fit: use `--soc_mode=free` for such data, or convert it.

`--soc_mode=free` keeps X-MACE's invariant SOC outputs (`--soc_num` numbers, any layout). `mace.modules.spin_states` has the tools: the state order, selection rules, Clebsch–Gordan coefficients, `soc_matrix_from_vectors`, `soc_vectors_from_matrix` and `spin_rotation`.

## Training

```bash
mace_run_train --name=ch2nh2 --train_file=train.xyz --valid_fraction=0.1 --E0s=average \
  --model=ExcitedMACE --n_energies=3 --compute_nacs --compute_socs --compute_dipoles \
  --pair_repulsion --r_max=5.0 --hidden_irreps='64x0e + 64x1o' --MLP_irreps=64x0e \
  --num_interactions=2 --correlation=3 --batch_size=10 --max_num_epochs=500 --lr=0.001 \
  --ema --ema_decay=0.99 --energy_weight=100 --forces_weight=1 --nacs_weight=100 \
  --socs_weight=100 --dipoles_weight=1 --error_table=EnergyNacsDipoleMAE \
  --default_dtype=float32 --device=cuda
```

The X-MACE command lines keep working, apart from `python scripts/run_train.py` becoming `mace_run_train`.

| option | default | |
|---|---|---|
| `--model` | | `ExcitedMACE` or `AutoencoderExcitedMACE` |
| `--n_energies` (`--n_states`) | 1 | number of states S |
| `--states` | S states of one multiplicity | states per multiplicity, e.g. `"3 0 2"` |
| `--soc_mode` | `auto` | `spin_orbit`, `free` or `auto` (spin_orbit when `--states` has spin-orbit-coupled multiplicities) |
| `--coupling_parity` | `both` | `both`: couplings have polar and axial parts; `fixed`: NACs and transition dipoles polar, SOC vectors axial |
| `--compute_nacs` / `--compute_socs` / `--compute_dipoles` | off | bare flag or `=True` |
| `--nac_num` / `--soc_num` / `--dipole_num` | P / P / S + P | sizes of the outputs (P: coupled pairs) |
| `--nacs_format` | `auto` | `raw`, `smooth` or `auto`, see above |
| `--nacs_weight` / `--socs_weight` / `--dipoles_weight` | 100 / 100 / 1 | loss weights |
| `--compute_oscillator_strengths` | off | predict oscillator strengths, see below |
| `--osc_num` | states of the ground state's multiplicity − 1 | number of oscillator strengths O (S0 → S1 … S_O) |
| `--oscillator_strengths_weight` | 10 | loss weight |
| `--train_only_oscillator_strengths` | off | train only the oscillator-strength readouts (e.g. when adding them to a trained model) |
| `--num_permutational_invariant` | 16 | latent size of `AutoencoderExcitedMACE` |
| `--excited_backbone` | `ScaleShiftMACE` | backbone built from scratch: `ScaleShiftMACE`, `MACE`, `MACELES`, `PolarMACE`, `MagneticScaleShiftMACE` (a foundation model sets it) |
| `--excited_long_range` | `shared` | `shared`: the backbone's electrostatics for every state; `state`: state-specific charges, see below |
| `--excited_charge_smearing` | 1.0 | width (Å) of the Gaussian charges of `--excited_long_range=state` |
| `--excited_adapter_irreps` | `Cx0e + Cx1o` | irreps of the adapter layer |
| `--pair_repulsion` | off | ZBL core repulsion, the same for every state |
| `--scaling` | `rms_forces_scaling` | `no_scaling` reproduces X-MACE |
| `--error_table` | `EnergyNacsDipoleMAE` | or `EnergyNacsDipoleRMSE` |

**Loss.** The loss is

  energy_weight · MSE(E) + forces_weight · MSE(F) + nacs_weight · RMSE(NACs) + socs_weight · RMSE(SOCs) + dipoles_weight · 100 · MSE(μ / n_atoms) [+ oscillator_strengths_weight · MSE(f / scale)].

The energies are total energies, and the couplings are phase free. The autoencoder adds the reconstruction and latent terms to the energy term. These are X-MACE's definitions, so its weights carry over.

**Oscillator strengths.** `--compute_oscillator_strengths` learns the oscillator strengths f(S0 → S_k) directly, from labels of f alone (no transition dipole vectors needed), e.g. from TDDFT outputs. They are rotation invariant, so the model predicts them as sums of per-atom invariant scalars, made non-negative by a softplus and multiplied by a scale per head and transition: the rms of that transition's labels in the training data (logged). The loss term is oscillator_strengths_weight · MSE(f / scale), so weak and bright transitions count alike. By default there is one per state of the ground state's multiplicity (S0 → S1 … S_{n−1} with n singlets; with a doublet ground state, the doublets). Heads may differ (`compute_oscillator_strengths` and `osc_num` per head); a head without them is masked. The calculators return `oscillator_strengths` `[O]`, and `mace_eval_configs` writes `MACE_oscillator_strengths`. A trained model without oscillator strengths can be the foundation of the same model with them (same heads and options plus `--compute_oscillator_strengths`): every weight, shift and scale is kept and only the oscillator-strength readouts are new; with `--train_only_oscillator_strengths` only they train, so the energies, forces and couplings stay exactly as they were and the data may have oscillator strengths only. With other changes, backbone and adapter are transferred as usual. Data with oscillator strengths only (no energies or forces) also train: point `--energy_key` and `--forces_key` to keys the file does not have, give `--E0s` as a dictionary (e.g. `"{1: 0.0, 6: 0.0, 7: 0.0, 8: 0.0}"`) and set `--energy_weight=0 --forces_weight=0`; with a force weight of 0 the forces are not computed in training, which saves the S backward passes.

Tutorial 8 (`tutorials/xmace/08_oscillator_strengths.ipynb`) walks through all of this, with scripts to convert and split data (`tutorials/xmace/tools/prepare_osc_data.py`), committee predictions (`osc_committee.py`) and a SLURM committee job (`train_committee_osc.slurm`).

**Core repulsion.** `--pair_repulsion` gives the backbone a ZBL pair potential, so it is the same for every state. The repulsion between nuclei does not depend on the electronic state, and it keeps dynamics stable at short distances where there is no training data.

**Scaling.** Per head, the backbone is scaled by the RMS of the forces of all states and shifted by the mean interaction energy per atom over the states; each state's readouts have the same scale and their own shift (that state's mean minus the head's mean). With a foundation backbone, the per-state shifts are fitted instead to the mean difference between the data and the foundation's energy per atom. `--scaling=no_scaling` turns the statistics off.

**Vectors.** A backbone built from scratch keeps l = 1 features in its last layer when `--hidden_irreps` has them, and the adapter layer always has them (`--excited_adapter_irreps`, default `Cx0e + Cx1o` with C = the backbone's channels, between 8 and 64). Every layer with l = 1 features contributes to the couplings. In X-MACE the last layer could not contribute to NACs.

**Transfer learning.** `--foundation_model=...` makes a foundation model the backbone, see the next section.

## Backbones, heads, foundation models and LoRA

**Backbones from scratch.** `--excited_backbone=MACELES` or `PolarMACE` builds that model (with its own options: `--les_arguments`, `--field_feature_widths`, ...) as the backbone; `MagneticScaleShiftMACE` needs magnetic moments in the data.

**Long-range electrostatics** (`--excited_long_range`). With a backbone that has electrostatics (MACELES, PolarMACE), `shared` (default) gives every state the backbone's long-range energy. `state` gives each state its own atomic charges, q_s = q_backbone + Δq_s, where Δq_s comes from readouts and sums to zero over each molecule (each state keeps the total charge). The energy of state s then changes by C(q_s) − C(q_backbone), with C the Coulomb energy of Gaussian charges of width `--excited_charge_smearing` (molecules only, no periodic cells). Δq starts at zero. The model outputs `state_charges` `[n_atoms, S]` and `state_electrostatic_energy` `[n_graphs, S]`; with a non-electrostatic backbone q_backbone = 0.

**Foundation models.** `--foundation_model` (a name or a path) makes a foundation model the backbone, with every state starting from its energy. Tested end to end (excited-state fine-tuning, multi-head fine-tuning with replay, LoRA) with:

| foundation | class | `--foundation_model` | notes |
|---|---|---|---|
| MACE-MP-0, MPA-0, OMAT-0, MATPES | `ScaleShiftMACE` | `small`, `medium-mpa-0`, `small-omat-0`, `mace-matpes-pbe-0`, ... or a path | |
| MACE-OFF23 | `ScaleShiftMACE` | `small_off`, `medium_off`, `large_off` | |
| MACE-OMOL-0 | `ScaleShiftMACE` (charge/spin embedding) | `mace_omol` or a path | optional `total_charge`, `total_spin` in `atoms.info` |
| MACE-MH-1 | `ScaleShiftMACE`, 6 heads | `mh-1` | pick the head with `--foundation_head` (e.g. `omol`, `spice_wB97M`, `omat_pbe`) |
| MACE-POLAR-1 | `PolarMACE` | `polar-1-s`, `polar-1-m`, `polar-1-l` | `--excited_long_range=shared` or `state` |

ScaleShiftMACE, MACELES and PolarMACE foundations are re-built for the elements of the training data, as in MACE fine-tuning (`--foundation_model_elements=True` keeps all of them). Other classes (plain `MACE`, `MagneticScaleShiftMACE`) are used as they are, which needs the training data to use the foundation's elements; every head then uses the foundation's head.

```bash
mace_run_train --model=ExcitedMACE --foundation_model=polar-1-s --n_energies=3 \
  --compute_nacs --compute_dipoles --excited_long_range=state --train_file=train.xyz ...
```

**Heads.** `--heads` takes a dictionary of heads, as in MACE. Every head has its own data and its own number of states and couplings: the keys `n_energies` (or `n_states`), `states`, `compute_nacs`, `compute_socs`, `compute_dipoles`, `nac_num`, `soc_num`, `dipole_num`, `soc_mode`, `nacs_format`, `nacs_key`, `socs_key`, `dipoles_key` override the command line for that head. A head with `ground_state: True` (or `n_energies: 1`) is a ground-state head: one energy and forces, plain MACE data (`REF_energy` a number, `REF_forces` an `[n_atoms, 3]` array). Each head has its own backbone head, readouts, scale and shifts; frames of different heads share batches, with labels padded to the largest head and masked in the loss and the metrics.

```bash
mace_run_train --model=ExcitedMACE --n_energies=3 --compute_nacs --compute_dipoles \
  --heads="{'excited': {'train_file': 'excited.xyz', 'valid_fraction': 0.1},
            'ground': {'train_file': 'ground.xyz', 'valid_fraction': 0.1, 'ground_state': True},
            'triplets': {'train_file': 't.xyz', 'states': [0, 0, 3], 'compute_dipoles': False}}" ...
```

**Multi-head fine-tuning.** With a foundation model, `--multiheads_finetuning=True` adds MACE's `pt_head` (replay of the foundation's data from `--pt_train_file`, or the downloaded MP data for MP models, optionally with `--pseudolabel_replay`), a ground-state head, next to the excited-state and ground-state heads. Use `--E0s=foundation`.

```bash
mace_run_train --model=ExcitedMACE --foundation_model=small_off --multiheads_finetuning=True \
  --pt_train_file=replay.xyz --E0s=foundation --n_energies=3 --compute_nacs \
  --heads="{'excited': {...}, 'ground': {..., 'ground_state': True}}" ...
```

**LoRA.** `--lora=True --lora_rank=4 --lora_alpha=1` freezes the backbone and trains low-rank adapters in it; the excited-state readouts and the adapter layer stay fully trainable. It works with single-head and multi-head fine-tuning, and the LoRA weights are merged into the backbone at the end.

**Head weights.** `weight` in a head's dictionary multiplies the loss weight of every configuration of that head (also for HDF5 data), e.g. to balance data sets of different sizes.

**A trained X-MACE model as the foundation.** `--foundation_model` also accepts an excited-state model (its element table is used, `--foundation_model_elements=True`):

* *continue* — every head of the run has the layout of a foundation head (same name, or `--foundation_head`) and the model class is the same: all weights, shifts and scales are kept and training continues on the new data (e.g. more data of the same kind, with a smaller `--lr`);
* *transfer* — otherwise: the backbone and the adapter layer keep their weights, the readouts of the run's heads are new and their shifts are fitted to the data. Combine with `--lora`, `--freeze` or `--lr_params_factors`.

The log says which of the two was done. With `--pseudolabel_replay`, the replay data are labelled by the foundation's ground-state backbone. Tutorial 7 shows both cases; [`foundation/README.md`](foundation/README.md) is the recipe for training an excited-state foundation model on many data sets.

**Using a head.** `ExcitedMACECalculator(model, head="ground")`, `SharcCalculator(..., head=...)`, `head <name>` in `MACE.template` and `mace_eval_configs --head=...` select a head; the default is the first head that is not `pt_head`. The calculator returns only that head's states and couplings. `model.heads`, `model.head_layouts` and `model.mask_tables()` describe what each head predicts.

**Large data sets.** `mace_prepare_data` stores the excited-state labels in HDF5 when given `--n_energies` (or `--states`) and the property options. With `--compute_statistics` it writes one shift per state:

```bash
mace_prepare_data --train_file=train.xyz --valid_fraction=0.1 --h5_prefix=h5/ --r_max=5.0 \
  --E0s=average --compute_statistics --n_energies=3 --compute_nacs --compute_socs
mace_run_train --model=ExcitedMACE --n_energies=3 --compute_nacs --compute_socs \
  --train_file=h5/train/train_0.h5 --valid_file=h5/val/val_0.h5 --statistics_file=h5/statistics.json ...
```

Heads with different layouts can also come from separate HDF5 files: give each head its own `train_file`/`valid_file` (`.h5`) and `E0s` in `--heads`, and `--atomic_numbers`. Each head's labels are padded to the largest head as they are read.

**Several GPUs.** Training is data parallel, as for any MACE model: every GPU gets its share of each batch, and DDP averages the gradients, so all GPUs take the same optimizer step. Start one process per GPU with torchrun (or `--launcher=slurm` / `mpi` on a cluster):

```bash
torchrun --standalone --nproc_per_node=4 -m mace.cli.run_train \
  --distributed --launcher=torchrun --device=cuda --batch_size=8 ... # 8 frames per GPU
```

`--batch_size` is per process, so the effective batch is batch_size × number of GPUs. The loss is that of the whole effective batch on one device: the means in the energy, force and dipole terms run over the frames of all GPUs, and the RMSE-type coupling terms (NACs, SOCs) are differentiated through the RMSE of the whole batch, not averaged over per-GPU RMSEs. The training statistics (per-state shifts and scales, and the fitted shifts on a foundation backbone) are summed over all GPUs. Mixed heads, the autoencoder, foundation backbones, multi-head fine-tuning and LoRA all work this way.

## Evaluation and dynamics

```bash
mace_eval_configs --configs=test.xyz --model=ch2nh2.model --output=pred.xyz
```

`pred.xyz` gets `MACE_energies` `[S]`, `MACE_energy` (state 0) and `MACE_forces` `[n_atoms, S, 3]` in `atoms.info`. If the model predicts them, it also gets `MACE_smooth_nacs`, `MACE_nacs`, `MACE_socs` and `MACE_dipoles`. Spin–orbit models write `MACE_soc_vectors` `[Q, 3]`, and `MACE_socs` holds the SOC matrix as an interleaved upper triangle.

```python
from mace.calculators import ExcitedMACECalculator

calc = ExcitedMACECalculator("ch2nh2.model", device="cuda", state=1)  # glob or list -> committee
atoms.calc = calc
atoms.get_potential_energy(), atoms.get_forces()          # state 1, for ASE MD
calc.results["energies"], calc.results["forces_states"]   # all states
calc.results["nacs"], calc.results["smooth_nacs"], calc.results["socs"], calc.results["dipoles"]
calc.results["soc_vectors"], calc.results["soc_matrix"]  # spin_orbit models: complex [nm, nm]
calc.layout   # which states the couplings belong to (multiplicities, NAC pairs, ...)
```

For a committee, each member's couplings are sign-aligned to the first model before averaging. `energies_var` and `forces_states_var` give the spread.

## Surface hopping with SHARC 4

`scripts/sharc/SHARC_MACE.py` is a SHARC interface (`SHARC_FAST`) for the SHARC 4.1 interface API. Copy or link it into `$SHARC/bin` (or put `scripts/sharc` on `PYTHONPATH`), and put `scripts/sharc/MACE.template` into the `QM` directory:

```
model_file     /path/to/model.model     # glob pattern -> committee; or modelpath in MACE.resources
properties     energy forces nacs socs dipoles   # smooth_nacs passes NAC*dE instead of NACs
device         cuda
energy_unit    eV                        # units the model was trained in (eV, Hartree)
distance_unit  Ang                       # Ang or Bohr
soc_unit       cm-1                      # optional: unit of the SOC labels (Hartree, eV, cm-1)
head           excited                   # optional: head of a multi-head model
```

The interface works both ways SHARC runs it. With `sharc.x`, `QM/runQM.sh` calls `SHARC_MACE.py QM.in` once per step. With pysharc, run `$SHARC/driver.py -i MACE input`; the model stays in memory, which is about ten times faster. It answers `h`, `soc`, `grad`, `nacdr`, `dm` and `phases`. `MACE.resources` is optional.

`mace.calculators.sharc_calculator.SharcCalculator` does the work and can be used without SHARC. It converts to atomic units and returns `h` (energies, plus SOCs if requested), `grad`, `nacdr` and `dm` for SHARC's `states`, for any multiplicities:

* **Models trained with `--states`.** SHARC may ask for as many states per multiplicity as the model has, or fewer (the lowest are used). Each spatial state's energy, gradient and permanent dipole go to all its M_S components. NACs and transition dipoles go between states of equal multiplicity and equal M_S. Spin–orbit models give the full Wigner–Eckart SOC matrix.
* **Models without multiplicities (X-MACE).** n_energies states of one multiplicity, for example singlets only or doublets only. Or singlets + triplets, singlets first (spatial). Or nmstates states in SHARC order (M_S-resolved). Their SOCs are the upper triangle of the nmstates × nmstates SOC matrix: nmstates(nmstates−1)/2 real values, or twice as many as (re, im) pairs.
* **Phases.** Couplings are sign-aligned with the previous step, so they change smoothly along a trajectory, and the interface reports all phases as +1. With `sharc.x` the previous couplings are kept in the save directory (`mace_phases.<step>`), so continuity holds across the separate processes.
* **Requests the model cannot answer.** These fail with a clear error, for example NACs from a model without NACs, or more states of a multiplicity than the model predicts. SOCs requested from a model without them are zero, with a warning.

**Tested with SHARC 4.1** (sharc-md/sharc4, main): single points, and `sharc.x` and pysharc trajectories with 2 singlets, 2 triplets and 1 quintet (13 M_S states), NACs, SOCs, dipoles and gradient correction. The total energy is conserved to < 0.3 meV over 10 fs, and the pysharc and `sharc.x` trajectories are identical. Doublets and quartets were checked in single points.

**pysharc input bug in SHARC 4.1.** In `source/input_list.f90`, `next_keyword` never sets its `stat` argument when it reads a line (`! stat = 0` is commented out). With optimisation, whether SHARC reads its `input` file then depends on leftover stack contents. When it fails, pysharc stops with `Keyword CHARGE not found.` and an empty input echo in `output.log`. Importing packages such as numpy's linear algebra or scipy (and so PyTorch models) is enough to trigger it, even with the LVC interface. The fix is to uncomment that line (`stat = 0`) and rebuild SHARC; `tests/test_sharc_real.py` detects the problem and says so.

## What changed compared with X-MACE

**Fixes**

* **Untrained readouts.** The SOC readouts and the autoencoder's latent readouts were not in the optimizer and stayed at their initial values. Every parameter is now in a parameter group, which MACE checks.
* **Core repulsion.** `--pair_repulsion` built a ZBL module, but its energy was never added. It now is, for every state.
* **No last-layer NACs.** The last-layer NAC readout had no l = 1 input, so it always returned zero. The last layer now keeps l = 1 features when vectors are predicted.
* **Per-atom NAC phases.** The NAC loss chose the sign per atom, which made the loss blind to the direction of the coupling vector. The sign is now chosen per configuration and pair.
* **Weights.** Per-frame weights and missing labels were ignored by the losses, and a missing NAC label had the wrong shape. A batch mixing labelled and unlabelled frames now works.
* **Single-state crash.** `compute_forces` crashed for a single state outside training (`retain == False` was a comparison, not an assignment). Forces are now computed per state with one backward pass each.
* **Dipoles.** Dipoles were declared but never predicted. They are now predicted as permanent and transition dipoles, with phase-free transition dipoles.
* **Decoder.** The autoencoder's decoder looped over the batch and used a general eigensolver. It now builds the batch of symmetric matrices at once and uses `eigvalsh`.
* **SHARC triplets and units.** The SHARC interface built `h` for spatial states but the SOC matrix for M_S states, so it failed with triplets, and it passed NACs to SHARC without unit conversion. Triplets, units and phases are now handled, see above.
* **SHARC 4.1.** The interface used the SHARC 4.0 template API (`template.types = {...}`, `.data`) and failed with SHARC 4.1. It now uses the 4.1 API, and allocates `QMout` per request.

**Added**

* **Spin multiplicities.** Doublets, quartets, quintets and any other multiplicity, with exactly degenerate M_S components and the NAC and dipole selection rules built in.
* **Any backbone.** Excited states on MACE, ScaleShiftMACE, MACELES, PolarMACE and MagneticScaleShiftMACE, from scratch or from MACE-MP/MPA/OMAT/MATPES, MACE-OFF, MACE-OMOL, MACE-MH-1 and MACE-POLAR-1, with shared or state-specific electrostatics.
* **Heads.** Several excited-state and ground-state heads in one model, multi-head fine-tuning with replay of the foundation's data, and LoRA.
* **Wigner–Eckart SOCs.** Rotation-equivariant SOC matrices, see [Spin multiplicities](#spin-multiplicities).

**Removed**

* **X-MACE's copy of MACE 0.3.6.** It changed `LinearReadoutBlock` and `compute_forces` for all models, removed BOTNet and the dipole models, and stopped exporting `MACE` and `ScaleShiftMACE`. X-MACE is now an addition: the ground-state MACE models are untouched.
* **Repository clutter.** Removed `build/`, `*.egg-info`, `__pycache__`, the committed `checkpoints/`, `logs/` and `results/` directories, the duplicated `SHARC_MACE.py` (`scripts/` and `tutorials/`), and the `model_type` special cases in the training loop. The autoencoder now computes its training outputs in its own forward pass.

**Known limitations**

* **Multi-head LMDB data.** Labels of heads with different layouts are padded for extended XYZ and HDF5 data, not for LMDB (`.aselmdb`) data.
* **Mixed continue/transfer.** Fine-tuning from an X-MACE model keeps all readouts only if every head of the run matches a foundation head; with one new head, all readouts are new (backbone and adapter are kept).
* **Evaluation options.** `mace_eval_configs` writes energies, forces and couplings of all states for excited-state models; `--compute_stress`, `--return_descriptors`, `--return_contributions` and `--return_node_energies` are refused for them.
* **TorchScript with PolarMACE.** PolarMACE models cannot be compiled to TorchScript (an upstream limitation); the `.model` file works in Python, the ASE calculator and SHARC.

* **Couplings at symmetric geometries.** At a geometry with a rotation axis (e.g. a C2v molecule), an exactly equivariant model can only produce coupling vectors that are unchanged by that rotation. Couplings between states of different symmetry species change sign under it, because the phase of the states absorbs the sign. So the model gives zero for them exactly at the symmetric geometry and is accurate again a little way off it. For SO2 the dip is about 0.01 Å wide along the antisymmetric stretch (tutorial 4). Mirror planes are handled by `--coupling_parity=both`. Sign-free outputs such as **v v**ᵀ would remove the restriction; that is not implemented.
* **Phase alignment per pair.** The losses choose the best sign per state pair, not per state (2^(S−1) combinations), so they do not constrain the sign of products of couplings around a loop of states.

**Not supported**

* **Training and deployment features.** Stress/virials, cuEquivariance/OpenEquivariance acceleration and LAMMPS.

## Tests

`tests/` covers the data pipeline, the models, the losses and metrics, the calculators, SHARC and CLI training:

* **Data pipeline.** Layouts, smooth NACs and missing labels.
* **Models.** Invariance and equivariance under rotation, translation, inversion and atom permutation. Finite-difference forces for every state. State-independent ZBL repulsion. Batching, TorchScript, float32. Foundation backbones (with and without l = 1 features), which must reproduce the ground state exactly, and the fit of the per-state shifts. Mixed ground- and excited-state heads (padding, masks, a frame alone equals the frame in a mixed batch). LoRA on the backbone. State-specific charges (charge conservation, invariance, finite-difference forces).
* **Losses and metrics.** Phase invariance, per-frame weights, autoencoder terms, and optimizer coverage.
* **Calculators.** ASE, committees, MD energy conservation, and the SHARC layer in all state modes and units, including phase continuity.
* **Spin states** (`test_spin_states.py`, `test_multiplicities.py`). Clebsch–Gordan coefficients against sympy. The same-spin SOC block against **a**·i**S** for doublets to quintets. Hermiticity, selection rules and rotation covariance H(R**a**) = U(R) H(**a**) U(R)† for six state layouts. SOC label layouts and the Wigner–Eckart residual warning. Axial SOC vectors (rotation, inversion) and TorchScript. The phase-free SOC loss and matrix-element metrics, and that a model learns the couplings. The SHARC layer with singlets/triplets/quintets and doublets/quartets: M_S degeneracy, NAC/dipole selection rules, the SOC matrix, rotation covariance, finite-difference gradients, fewer states than the model, and phase continuity.
* **CLI training.** All properties, the autoencoder with Stage Two, smooth NACs with missing labels, foundation models, float32, HDF5 preprocessing (also with `--states`) and `mace_eval_configs`. Also `--states="1 0 2"` with spin–orbit couplings, evaluated and passed to the SHARC layer. Mixed ground/excited heads for both models (also from HDF5), ScaleShiftMACE/MACE/MagneticScaleShiftMACE foundation backbones, multi-head fine-tuning with `pt_head` replay with and without LoRA, and MACE/MACELES/PolarMACE backbones from scratch with shared and state-specific electrostatics.
* **Data-parallel training** (`test_excited_distributed.py`, two CPU processes with gloo). The gradient averaged by DDP over the ranks equals the gradient of the whole batch on one device, for every loss term, both models, mixed heads (one rank with ground-state frames only), state-specific charges and fixed parity; the ranks stay identical over optimizer steps. Through the CLI (torchrun, two processes), mixed-head training starts from the same statistics as a single-process run.
* **Heads** (`test_excited_heads.py`). Per-head specs and padding, padded entries ignored by the loss and the metrics, head selection in the calculators and `evaluate_excited_configs`, several heads on one backbone head, per-head statistics.
* **Core MACE components with excited-state models** (`test_core_components.py`). Every backbone option — Bessel, Gaussian and Chebyshev radial bases, Agnesi and Soft distance transforms, residual, density, non-linear and attention interaction blocks, gates, correlation orders 1–3, scalar-only and l = 2 features, reduced CG, agnostic products, last-readout-only, no cutoff, three layers — for both models: invariance, equivariance, finite-difference forces, copies and serialisation, TorchScript. Charge and spin embeddings. Head selection (`select_heads`, `remove_pt_head`), E0 estimation from multi-state energies, optimiser groups with freezing and learning-rate factors, HDF5 head assignment, padding and head weights.
* **Core MACE training features** (`test_core_training.py`). EMA and Stage Two, restarts, checkpoint retention, LBFGS / AdamW / schedule-free optimisers and schedulers with gradient clipping, freezing and learning-rate factors on a foundation, config files, several files per head with validation, test files, config-type and head weights, every E0 option, data-loader workers, scaling and error tables, foundation elements and readouts, multi-head replay with FPS, pseudolabels and filtering, LoRA with Stage Two and EMA, continuing and transferring from a trained X-MACE model, `mace_select_head`, `mace_convert_device`, `mace_plot_train`, `mace_eval_configs`, and the foundation pipeline dry run (`foundation/dry_run.sh`).
* **Parity and autoencoder.** Polar and axial parts of couplings under inversion. Planar geometries: `fixed` gives no out-of-plane transition dipoles and no in-plane SOC vectors, `both` does. Autoencoder states ordered within each multiplicity, and its encoder distinguishes multiplicities.
* **SHARC labels.** `labels_from_sharc` applied to the SHARC layer's own output gives back the model's predictions, and it rejects non-degenerate multiplets.
* **Real SHARC** (`test_sharc_real.py`, skipped without `$SHARC`). A `SHARC_MACE.py` single point against `SharcCalculator`, read with SHARC's own `QM.out` parser. A `sharc.x` trajectory with energy conservation. A pysharc trajectory that must match `sharc.x`.

```bash
pytest tests                    # about 20 minutes on a CPU
pytest tests -m "not slow"
source $SHARC/sharcvars.sh && pytest tests/test_sharc_real.py
```
