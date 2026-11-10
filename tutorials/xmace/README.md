# X-MACE tutorials

Nine Jupyter notebooks in four parts: the physics of excited states (Part 1); building
a data set, training from scratch and fine-tuning foundation models (Part 2); using a
trained model, up to surface-hopping dynamics with SHARC (Part 3); and more data,
oscillator strengths and more models (Part 4). They are written for
new users who know some Python and quantum chemistry but have not used MACE or SHARC
before. Every notebook was run end to end; the outputs you see are real.

The example throughout is **SO2 with 4 singlets and 3 triplets**, with SHARC's linear
vibronic coupling (LVC) model standing in for the quantum chemistry. It is cheap, so
the whole series runs on a laptop CPU, yet it has everything an ab initio method gives
SHARC: energies, gradients, NACs, spin–orbit couplings and dipoles.

## Contents

| | notebook | what you learn | needs | time (2 CPU cores) |
|---|---|---|---|---|
| **Part 1 — Concepts** | | | | |
| | [01 Excited states and symmetry](01_excited_states_and_symmetry.ipynb) | what X-MACE predicts; smooth NACs; arbitrary phases; multiplicities and M_S states; SOC selection rules; Wigner–Eckart coupling vectors; rotation equivariance; polar and axial couplings in planar molecules | — | 1 min |
| **Part 2 — Data and training** | | | | |
| | [02 Building a dataset](02_building_a_dataset.ipynb) | checking SOC data; Wigner sampling; snapshots from dynamics; SHARC's `QM.out` quantities and units; converting them to X-MACE labels; inspecting the data | SHARC | 5 min |
| | [03 Training](03_training.ipynb) | every training option; reading the log; learning curves; error tables; phase-consistent parity plots; choosing loss weights | 02 | 30 min |
| | [04 Foundation models and fine-tuning](04_foundation_models_and_finetuning.ipynb) | complete `mace_run_train` commands (bash cells, also in [`04_commands.sh`](04_commands.sh)) for MACE-OFF, MACE-MP/MPA/OMAT, MACE-MH-1, MACE-POLAR-1 and MACE-OMOL compared with training from scratch; one command per strategy: full fine-tuning, frozen layers, backbone learning rates, LoRA, multi-head replay with pseudolabels, an extra ground-state head (YAML config), autoencoder, state-specific electrostatics; fine-tuning a trained model on additional data and transferring it to new states; the same on several GPUs and SLURM | 02, internet | 60 min |
| **Part 3 — Using models** | | | | |
| | [05 Using a trained model](05_using_a_trained_model.ipynb) | the ASE calculator; potential-energy and coupling scans against the reference; symmetry checks; `SharcCalculator`; ASE dynamics; `mace_eval_configs` | 03, SHARC | 3 min |
| | [06 Surface hopping with SHARC](06_surface_hopping_with_sharc.ipynb) | how SHARC calls X-MACE; `MACE.template`; single points; pysharc and `sharc.x` trajectories; energy conservation; populations compared with the reference | 03, SHARC | 15 min |
| **Part 4 — Going further** | | | | |
| | [07 More data](07_more_data.ipynb) | missing labels and per-frame weights; HDF5 for large data sets; free SOCs and data from the original X-MACE; doublets and quartets | 02 | 3 min |
| | [08 Oscillator strengths](08_oscillator_strengths.ipynb) | learning oscillator strengths directly; preparing and splitting data (also f stored under another key, Windows files) with `tools/prepare_osc_data.py`; models on oscillator strengths only (no forces computed) and on energies, forces and f together; an absorption spectrum; committees with uncertainties (`tools/osc_committee.py`); adding f to a trained model; a SLURM committee on MACE-POLAR-1-M (`tools/train_committee_osc.slurm`); checking the installation and troubleshooting | 02, 03 | 15 min |
| | [09 More models](09_more_models.ipynb) | committees and uncertainty; the autoencoder model; backbones, foundation models and heads, Stage Two and restarts (overview); the Python API | 03, SHARC | 12 min |

## Setup

1. Install X-MACE (from the repository root):

   ```bash
   pip install -e .
   pip install jupyter matplotlib
   ```

2. Install SHARC 4.1 with pysharc (https://github.com/sharc-md/sharc4) and load its
   environment, so that `$SHARC` points to SHARC's `bin` directory:

   ```bash
   source /path/to/sharc/bin/sharcvars.sh
   ```

   Apply the one-line fix described in `XMACE.md` ("pysharc input bug") before
   running pysharc with PyTorch models.

3. Start Jupyter **in this directory**, so that the notebooks find `tools/` and write
   their files to `work/`:

   ```bash
   cd tutorials/xmace
   jupyter lab
   ```

Run Parts 2 and 3 in order: 02 writes the data that 03 and 04 train on, and 05–09 use
the model from 03. Notebook 01 is independent, and Part 4 can be read in any order.
On a GPU, add `--device=cuda` in notebooks 03 and 04 and training takes minutes.

## Files

* `tools/so2_lvc.py` — helpers for the tutorials: the SO2 LVC model, Wigner sampling,
  reference calculations with SHARC's LVC interface, writing and running SHARC
  trajectories, reading `output.lis`.
* `tools/prepare_osc_data.py`, `tools/osc_committee.py`, `tools/train_committee_osc.slurm`
  — oscillator-strength data preparation, committee predictions and a SLURM committee
  job (Tutorial 8).
* `04_commands.sh` — all commands of Tutorial 4 as one script.
* `work/` — everything the notebooks generate (data, models, trajectories); it is not
  part of the repository and can be deleted at any time.

## Glossary

| term | meaning |
|---|---|
| adiabatic / MCH states | eigenstates of the electronic Hamiltonian without spin–orbit coupling (SHARC: "molecular Coulomb Hamiltonian"), sorted by energy within each multiplicity |
| diagonal states | eigenstates of the full Hamiltonian including SOCs; SHARC hops between these |
| spatial state | one electronic state of a given spin, without its M_S components (X-MACE predicts one energy per spatial state) |
| M_S-resolved states (nmstates) | all spin components; a triplet counts three times |
| NAC | non-adiabatic coupling vector ⟨ψ_I|∇ψ_J⟩, couples states of the same spin |
| smooth NAC | NAC × energy gap; what X-MACE learns |
| SOC | spin–orbit coupling matrix element between M_S states |
| SOC coupling vector | the vector **a**_IJ from which all SOC elements between two spatial states follow (Wigner–Eckart) |
| phase | the arbitrary sign of an electronic wavefunction |
| equivariant | transforms with the molecule under rotations (forces, NACs, dipoles) |
| Wigner sampling | initial conditions from the quantum distribution of the vibrational ground state |
| LVC | linear vibronic coupling model: a cheap model Hamiltonian fitted to quantum chemistry |
| committee | several models trained with different random seeds; their spread estimates the uncertainty |
