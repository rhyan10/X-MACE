# X-MACE tutorials

Six Jupyter notebooks that take you from the physics of excited states to
surface-hopping dynamics with SHARC driven by an X-MACE model. They are written for
new users who know some Python and quantum chemistry but have not used MACE or SHARC
before. Every notebook was run end to end; the outputs you see are real.

The example throughout is **SO2 with 4 singlets and 3 triplets**, with SHARC's linear
vibronic coupling (LVC) model standing in for the quantum chemistry. It is cheap, so
the whole series runs on a laptop CPU, yet it has everything an ab initio method gives
SHARC: energies, gradients, NACs, spin–orbit couplings and dipoles.

| notebook | what you learn | needs | time (2 CPU cores) |
|---|---|---|---|
| [01 Excited states and symmetry](01_excited_states_and_symmetry.ipynb) | what X-MACE predicts; smooth NACs; arbitrary phases; multiplicities and M_S states; SOC selection rules; Wigner–Eckart coupling vectors; rotation equivariance; polar and axial couplings in planar molecules | — | 1 min |
| [02 Building a dataset](02_building_a_dataset.ipynb) | checking SOC data; Wigner sampling; snapshots from dynamics; SHARC's `QM.out` quantities and units; converting them to X-MACE labels; inspecting the data | SHARC | 5 min |
| [03 Training](03_training.ipynb) | every training option; reading the log; learning curves; error tables; phase-consistent parity plots; choosing loss weights | 02 | 30 min |
| [04 Using a trained model](04_using_a_trained_model.ipynb) | the ASE calculator; potential-energy and coupling scans against the reference; symmetry checks; `SharcCalculator`; ASE dynamics; `mace_eval_configs` | 03, SHARC | 3 min |
| [05 Surface hopping with SHARC](05_surface_hopping_with_sharc.ipynb) | how SHARC calls X-MACE; `MACE.template`; single points; pysharc and `sharc.x` trajectories; energy conservation; populations compared with the reference | 03, SHARC | 15 min |
| [06 More models and data](06_more_models_and_data.ipynb) | committees and uncertainty; the autoencoder model; missing labels and weights; HDF5 for large data; free SOCs and X-MACE data; doublets and quartets; foundation models; the Python API | 03 | 15 min |

## Setup

1. Install this MACE branch (from the repository root):

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

Run the notebooks in order: 02 writes the data that 03 trains on, and 04–06 use the
model from 03. Notebook 01 is independent. On a GPU, add `--device=cuda` in notebook 03
and training takes a minute or two.

## Files

* `tools/so2_lvc.py` — helpers for the tutorials: the SO2 LVC model, Wigner sampling,
  reference calculations with SHARC's LVC interface, writing and running SHARC
  trajectories, reading `output.lis`.
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
