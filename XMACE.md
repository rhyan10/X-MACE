# Excited states with MACE (X-MACE)

X-MACE extends MACE to several electronic states. One network predicts, for S states:

| quantity | output | shape | how |
|---|---|---|---|
| energies | `energy` | `[n_graphs, S]` | per-atom invariant readouts, plus E0s and the ZBL core repulsion |
| forces | `forces` | `[n_atoms, S, 3]` | −∂E_s/∂r for every state |
| non-adiabatic couplings | `nacs` | `[n_atoms, P, 3]` | per-atom equivariant vectors (l = 1); smooth NACs, NAC × ΔE |
| spin–orbit couplings | `socs` | `[n_graphs, L]` | sums of per-atom invariant scalars (`--soc_mode=free`) |
| spin–orbit coupling vectors | `soc_vectors` | `[n_graphs, Q, 3]` | sums of per-atom axial vectors (`--soc_mode=spin_orbit`), see [Spin multiplicities](#spin-multiplicities) |
| permanent and transition dipoles | `dipoles` | `[n_graphs, D, 3]` | sums of per-atom equivariant vectors |

P = S(S−1)/2 state pairs in the order (0,1), (0,2), …, (1,2), …. By default L = P and D = S + P (S permanent dipoles, then one transition dipole per pair). With `--states`, P counts only pairs of the same multiplicity.

There are two models:

* **`ExcitedMACE`** predicts all states directly.
* **`AutoencoderExcitedMACE`** predicts a permutation-invariant latent vector. A decoder turns it into a symmetric tridiagonal S × S matrix, and the energies are its eigenvalues. This keeps the states ordered and their crossings smooth near conical intersections. With several multiplicities, the matrix has one block per multiplicity, the encoder tags each energy with its multiplicity, and the energy shift is shared within a multiplicity. The states are then ordered within each multiplicity, as SHARC's MCH states are. During training an encoder (a deep set over the states) maps the reference energies to the latent space. The loss asks the decoder to reconstruct them and the network to reproduce the encoded vector.

Both live in `mace/modules/excited.py` and work with the rest of MACE: the CLI, checkpoints, Stage Two, EMA, foundation models, TorchScript export, ASE and SHARC.

**Tutorials.** `tutorials/xmace/` has six Jupyter notebooks for new users, from the physics (smooth NACs, phases, multiplicities, Wigner–Eckart SOCs, equivariance) through building a data set with SHARC, training and validation, to surface-hopping dynamics with SHARC. They use SO2 with SHARC's LVC model as the reference.

## Data

Each frame of an extended XYZ file carries its labels in `atoms.info`, as in X-MACE:

| key (CLI option) | shape | |
|---|---|---|
| `REF_energy` (`--energy_key`) | `[S]` or `[1, S]` | eV |
| `REF_forces` (`--forces_key`) | `[n_atoms, S, 3]` (also `[S, n_atoms, 3]`, or `[n_atoms, 3S]` in `atoms.arrays`) | eV/Å |
| `REF_nacs` (`--nacs_key`) | `[n_atoms, P, 3]` (also `[P, n_atoms, 3]`) | raw or smooth, see below |
| `REF_socs` (`--socs_key`) | `[L]` real | any unit |
| `REF_dipoles` (`--dipoles_key`) | `[D, 3]` | any unit |

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
| `--num_permutational_invariant` | 16 | latent size of `AutoencoderExcitedMACE` |
| `--pair_repulsion` | off | ZBL core repulsion, the same for every state |
| `--scaling` | `rms_forces_scaling` | `no_scaling` reproduces X-MACE |
| `--error_table` | `EnergyNacsDipoleMAE` | or `EnergyNacsDipoleRMSE` |

**Loss.** The loss is

  energy_weight · MSE(E) + forces_weight · MSE(F) + nacs_weight · RMSE(NACs) + socs_weight · RMSE(SOCs) + dipoles_weight · 100 · MSE(μ / n_atoms).

The energies are total energies, and the couplings are phase free. The autoencoder adds the reconstruction and latent terms to the energy term. These are X-MACE's definitions, so its weights carry over.

**Core repulsion.** `--pair_repulsion` adds a ZBL pair potential to every state. The repulsion between nuclei does not depend on the electronic state, and it keeps dynamics stable at short distances where there is no training data. The ZBL energy is added in eV and is not scaled (`ScaleShiftMACE` multiplies its ZBL energy by the scale).

**Scaling.** The atomic interaction energies are scaled by the RMS of the forces of all states, and shifted by one value per state: the mean interaction energy per atom of that state. `--scaling=no_scaling` turns this off, as in X-MACE.

**Vectors in the last layer.** MACE reduces its last layer to scalars. When NACs, dipoles or SOC vectors are predicted, the last layer keeps its l = 1 features, so every layer contributes to them. In X-MACE the last layer could not contribute to NACs.

**Transfer learning.** `--foundation_model=medium_off` (or `small`, `medium`, `large`, `*_off`, or a path to a MACE model) takes the architecture and weights of the backbone from a ground-state MACE model. With `--foundation_model_readout=True` (the default), every state starts from the foundation's energy readout and scale/shift. The model then reproduces the foundation's potential energy surface for every state before training. Where the last layer keeps l = 1 features, the foundation weights fill the scalar part.

**Large data sets.** `mace_prepare_data` stores the excited-state labels in HDF5 when given `--n_energies` (or `--states`) and the property options. With `--compute_statistics` it writes one shift per state:

```bash
mace_prepare_data --train_file=train.xyz --valid_fraction=0.1 --h5_prefix=h5/ --r_max=5.0 \
  --E0s=average --compute_statistics --n_energies=3 --compute_nacs --compute_socs
mace_run_train --model=ExcitedMACE --n_energies=3 --compute_nacs --compute_socs \
  --train_file=h5/train/train_0.h5 --valid_file=h5/val/val_0.h5 --statistics_file=h5/statistics.json ...
```

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
```

The interface works both ways SHARC runs it. With `sharc.x`, `QM/runQM.sh` calls `SHARC_MACE.py QM.in` once per step. With pysharc, run `$SHARC/driver.py -i MACE input`; the model stays in memory, which is about ten times faster. It answers `h`, `soc`, `grad`, `nacdr`, `dm` and `phases`. `MACE.resources` is optional.

`mace.calculators.sharc_calculator.SharcCalculator` does the work and can be used without SHARC. It converts to atomic units and returns `h` (energies, plus SOCs if requested), `grad`, `nacdr` and `dm` for SHARC's `states`, for any multiplicities:

* **Models trained with `--states`.** SHARC may ask for as many states per multiplicity as the model has, or fewer (the lowest are used). Each spatial state's energy, gradient and permanent dipole go to all its M_S components. NACs and transition dipoles go between states of equal multiplicity and equal M_S. Spin–orbit models give the full Wigner–Eckart SOC matrix.
* **Models without multiplicities (X-MACE).** n_energies states of one multiplicity, for example singlets only or doublets only. Or singlets + triplets, singlets first (spatial). Or nmstates states in SHARC order (M_S-resolved). Their SOCs are the upper triangle of the nmstates × nmstates SOC matrix: nmstates(nmstates−1)/2 real values, or twice as many as (re, im) pairs.
* **Phases.** Couplings are sign-aligned with the previous step, so they change smoothly along a trajectory, and the interface reports all phases as +1. With `sharc.x` the previous couplings are kept in the save directory (`mace_phases.<step>`), so continuity holds across the separate processes.
* **Requests the model cannot answer.** These fail with a clear error, for example NACs from a model without NACs, or more states of a multiplicity than the model predicts. SOCs requested from a model without them are zero, with a warning.

**Tested with SHARC 4.1** (sharc-md/sharc4, main): single points, and `sharc.x` and pysharc trajectories with 2 singlets, 2 triplets and 1 quintet (13 M_S states), NACs, SOCs, dipoles and gradient correction. The total energy is conserved to < 0.3 meV over 10 fs, and the pysharc and `sharc.x` trajectories are identical. Doublets and quartets were checked in single points.

**pysharc input bug in SHARC 4.1.** In `source/input_list.f90`, `next_keyword` never sets its `stat` argument when it reads a line (`! stat = 0` is commented out). With optimisation, whether SHARC reads its `input` file then depends on leftover stack contents. When it fails, pysharc stops with `Keyword CHARGE not found.` and an empty input echo in `output.log`. Importing packages such as numpy's linear algebra or scipy (and so PyTorch models) is enough to trigger it, even with the LVC interface. The fix is to uncomment that line (`stat = 0`) and rebuild SHARC; `tests/excited/test_sharc_real.py` detects the problem and says so.

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
* **Wigner–Eckart SOCs.** Rotation-equivariant SOC matrices, see [Spin multiplicities](#spin-multiplicities).

**Removed**

* **X-MACE's copy of MACE 0.3.6.** It changed `LinearReadoutBlock` and `compute_forces` for all models, removed BOTNet and the dipole models, and stopped exporting `MACE` and `ScaleShiftMACE`. X-MACE is now an addition: the ground-state MACE models are untouched.
* **Repository clutter.** Removed `build/`, `*.egg-info`, `__pycache__`, the committed `checkpoints/`, `logs/` and `results/` directories, the duplicated `SHARC_MACE.py` (`scripts/` and `tutorials/`), and the `model_type` special cases in the training loop. The autoencoder now computes its training outputs in its own forward pass.

**Known limitations**

* **Couplings at symmetric geometries.** At a geometry with a rotation axis (e.g. a C2v molecule), an exactly equivariant model can only produce coupling vectors that are unchanged by that rotation. Couplings between states of different symmetry species change sign under it, because the phase of the states absorbs the sign. So the model gives zero for them exactly at the symmetric geometry and is accurate again a little way off it. For SO2 the dip is about 0.01 Å wide along the antisymmetric stretch (tutorial 4). Mirror planes are handled by `--coupling_parity=both`. Sign-free outputs such as **v v**ᵀ would remove the restriction; that is not implemented.
* **Phase alignment per pair.** The losses choose the best sign per state pair, not per state (2^(S−1) combinations), so they do not constrain the sign of products of couplings around a loop of states.

**Not supported**

* **Training and deployment features.** Multi-head fine-tuning, stress/virials, cuEquivariance/OpenEquivariance acceleration and LAMMPS.

## Tests

`tests/excited` covers the data pipeline, the models, the losses and metrics, the calculators, SHARC and CLI training:

* **Data pipeline.** Layouts, smooth NACs and missing labels.
* **Models.** Invariance and equivariance under rotation, translation, inversion and atom permutation. Finite-difference forces for every state. The ZBL repulsion against a direct calculation. Batching, TorchScript, float32, and transfer from a foundation model, which must reproduce the ground state exactly.
* **Losses and metrics.** Phase invariance, per-frame weights, autoencoder terms, and optimizer coverage.
* **Calculators.** ASE, committees, MD energy conservation, and the SHARC layer in all state modes and units, including phase continuity.
* **Spin states** (`test_spin_states.py`, `test_multiplicities.py`). Clebsch–Gordan coefficients against sympy. The same-spin SOC block against **a**·i**S** for doublets to quintets. Hermiticity, selection rules and rotation covariance H(R**a**) = U(R) H(**a**) U(R)† for six state layouts. SOC label layouts and the Wigner–Eckart residual warning. Axial SOC vectors (rotation, inversion) and TorchScript. The phase-free SOC loss and matrix-element metrics, and that a model learns the couplings. The SHARC layer with singlets/triplets/quintets and doublets/quartets: M_S degeneracy, NAC/dipole selection rules, the SOC matrix, rotation covariance, finite-difference gradients, fewer states than the model, and phase continuity.
* **CLI training.** All properties, the autoencoder with Stage Two, smooth NACs with missing labels, foundation models, float32, HDF5 preprocessing (also with `--states`) and `mace_eval_configs`. Also `--states="1 0 2"` with spin–orbit couplings, evaluated and passed to the SHARC layer.
* **Parity and autoencoder.** Polar and axial parts of couplings under inversion. Planar geometries: `fixed` gives no out-of-plane transition dipoles and no in-plane SOC vectors, `both` does. Autoencoder states ordered within each multiplicity, and its encoder distinguishes multiplicities.
* **SHARC labels.** `labels_from_sharc` applied to the SHARC layer's own output gives back the model's predictions, and it rejects non-degenerate multiplets.
* **Real SHARC** (`test_sharc_real.py`, skipped without `$SHARC`). A `SHARC_MACE.py` single point against `SharcCalculator`, read with SHARC's own `QM.out` parser. A `sharc.x` trajectory with energy conservation. A pysharc trajectory that must match `sharc.x`.

```bash
pytest tests/excited            # about 5 minutes on a CPU
pytest tests/excited -m "not slow"
source $SHARC/sharcvars.sh && pytest tests/excited/test_sharc_real.py
```
