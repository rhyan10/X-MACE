# Excited states with MACE (X-MACE)

X-MACE extends MACE to several electronic states. One network predicts, for S states:

| quantity | output | shape | how |
|---|---|---|---|
| energies | `energy` | `[n_graphs, S]` | per-atom invariant readouts, plus E0s and the ZBL core repulsion |
| forces | `forces` | `[n_atoms, S, 3]` | −∂E_s/∂r for every state |
| non-adiabatic couplings | `nacs` | `[n_atoms, P, 3]` | per-atom equivariant vectors (l = 1); smooth NACs, NAC × ΔE |
| spin–orbit couplings | `socs` | `[n_graphs, L]` | sums of per-atom invariant scalars |
| permanent and transition dipoles | `dipoles` | `[n_graphs, D, 3]` | sums of per-atom equivariant vectors |

P = S(S−1)/2 state pairs in the order (0,1), (0,2), …, (1,2), …. By default L = P and D = S + P (S permanent dipoles, then one transition dipole per pair).

There are two models:

* **`ExcitedMACE`** predicts all states directly.
* **`AutoencoderExcitedMACE`** predicts a permutation-invariant latent vector. A decoder turns it into a symmetric tridiagonal S × S matrix, and the energies are its eigenvalues. This keeps the states ordered and their crossings smooth near conical intersections. During training an encoder (a deep set over the states) maps the reference energies to the latent space. The loss asks the decoder to reconstruct them and the network to reproduce the encoded vector.

Both live in `mace/modules/excited.py` and work with the rest of MACE: the CLI, checkpoints, Stage Two, EMA, foundation models, TorchScript export, ASE and SHARC.

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
| `--compute_nacs` / `--compute_socs` / `--compute_dipoles` | off | bare flag or `=True` |
| `--nac_num` / `--soc_num` / `--dipole_num` | P / P / S + P | sizes of the outputs |
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

**Vectors in the last layer.** MACE reduces its last layer to scalars. When NACs or dipoles are predicted, the last layer keeps its l = 1 features, so every layer contributes to them. In X-MACE the last layer could not contribute to NACs.

**Transfer learning.** `--foundation_model=medium_off` (or `small`, `medium`, `large`, `*_off`, or a path to a MACE model) takes the architecture and weights of the backbone from a ground-state MACE model. With `--foundation_model_readout=True` (the default), every state starts from the foundation's energy readout and scale/shift. The model then reproduces the foundation's potential energy surface for every state before training. Where the last layer keeps l = 1 features, the foundation weights fill the scalar part.

## Evaluation and dynamics

```bash
mace_eval_configs --configs=test.xyz --model=ch2nh2.model --output=pred.xyz
```

`pred.xyz` gets `MACE_energies` `[S]`, `MACE_energy` (state 0) and `MACE_forces` `[n_atoms, S, 3]` in `atoms.info`. If the model predicts them, it also gets `MACE_smooth_nacs`, `MACE_nacs`, `MACE_socs` and `MACE_dipoles`.

```python
from mace.calculators import ExcitedMACECalculator

calc = ExcitedMACECalculator("ch2nh2.model", device="cuda", state=1)  # glob or list -> committee
atoms.calc = calc
atoms.get_potential_energy(), atoms.get_forces()          # state 1, for ASE MD
calc.results["energies"], calc.results["forces_states"]   # all states
calc.results["nacs"], calc.results["smooth_nacs"], calc.results["socs"], calc.results["dipoles"]
```

For a committee, each member's couplings are sign-aligned to the first model before averaging. `energies_var` and `forces_states_var` give the spread.

## Surface hopping with SHARC 4

`scripts/sharc/SHARC_MACE.py` is a SHARC 4 interface (`SHARC_FAST`). Copy or link it into `$SHARC/bin`, and put `scripts/sharc/MACE.template` into the `QM` directory:

```
model_file     /path/to/model.model     # glob pattern -> committee
properties     energy forces nacs        # add socs / dipoles; smooth_nacs passes NAC*dE
device         cuda
energy_unit    eV                        # units the model was trained in
distance_unit  Ang
```

`mace.calculators.sharc_calculator.SharcCalculator` does the work and can be used without SHARC. It converts to atomic units and returns `h` (energies, plus SOCs if requested), `grad`, `nacdr` and `dm`.

* **Spatial model.** A model with `n_energies` = singlets + triplets, singlets first. Each triplet's energy, gradient and permanent dipole are copied to its three M_S components. NACs and transition dipoles couple states of the same multiplicity and M_S only.
* **M_S-resolved model.** A model with `n_energies` = nmstates, in SHARC order.
* **Singlet-only run.** A singlet-only run may use a model with more states; the lowest ones are used.
* **SOCs.** These are the upper triangle of the nmstates × nmstates SOC matrix (X-MACE layout). Use nmstates(nmstates−1)/2 real values, or twice as many as (re, im) pairs.
* **Phases.** Couplings are sign-aligned with the previous step, so they change smoothly along a trajectory, and the interface reports all phases as +1.
* **Requests the model cannot answer.** These fail with a clear error, for example NACs from a model without NACs, or a mismatch between the number of states and SHARC's `states` line.

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

**Removed**

* **X-MACE's copy of MACE 0.3.6.** It changed `LinearReadoutBlock` and `compute_forces` for all models, removed BOTNet and the dipole models, and stopped exporting `MACE` and `ScaleShiftMACE`. X-MACE is now an addition: the ground-state MACE models are untouched.
* **Repository clutter.** Removed `build/`, `*.egg-info`, `__pycache__`, the committed `checkpoints/`, `logs/` and `results/` directories, the duplicated `SHARC_MACE.py` (`scripts/` and `tutorials/`), and the `model_type` special cases in the training loop. The autoencoder now computes its training outputs in its own forward pass.

**Not supported**

* **Data and hardware.** HDF5/LMDB training data (`mace_preprocess_data`), multi-head fine-tuning, stress/virials, cuEquivariance/OpenEquivariance acceleration and LAMMPS. Excited-state models train from extended XYZ files.

## Tests

`tests/excited` covers the data pipeline, the models, the losses and metrics, the calculators, SHARC and CLI training:

* **Data pipeline.** Layouts, smooth NACs and missing labels.
* **Models.** Invariance and equivariance under rotation, translation, inversion and atom permutation. Finite-difference forces for every state. The ZBL repulsion against a direct calculation. Batching, TorchScript, float32, and transfer from a foundation model, which must reproduce the ground state exactly.
* **Losses and metrics.** Phase invariance, per-frame weights, autoencoder terms, and optimizer coverage.
* **Calculators.** ASE, committees, MD energy conservation, and the SHARC layer in all state modes and units, including phase continuity.
* **CLI training.** All properties, the autoencoder with Stage Two, smooth NACs with missing labels, foundation models, float32 and `mace_eval_configs`.

```bash
pytest tests/excited            # about 3 minutes on a CPU
pytest tests/excited -m "not slow"
```
