# X-MACE

**Excited-state MACE**: machine-learned potentials for several electronic states at
once — energies and forces of every state, non-adiabatic couplings (NACs),
spin–orbit couplings (SOCs), permanent and transition dipoles and oscillator
strengths — for non-adiabatic dynamics with SHARC or ASE and absorption spectra.

X-MACE is built on [MACE](https://github.com/ACEsuit/mace). Every state is the
energy of a ground-state MACE model (the *backbone*) plus a learned per-state
correction, so any MACE architecture or foundation model can carry excited states.

## Features

* **Two models.** `ExcitedMACE` (one readout per state) and
  `AutoencoderExcitedMACE` (states ordered via a permutation-invariant latent
  eigenvalue problem; smooth crossings).
* **Physics built in.** Smooth NACs (NAC × ΔE), phase-free losses and metrics, any
  spin multiplicities (singlets … quintets, doublets, quartets) with exactly
  degenerate M_S components, Wigner–Eckart spin–orbit couplings (the full
  M_S-resolved SOC matrix from one coupling vector per pair of states), polar and
  axial couplings for planar molecules, ZBL core repulsion.
* **Any backbone and foundation model.** `ScaleShiftMACE`, `MACE`, `MACELES`,
  `PolarMACE`, `MagneticScaleShiftMACE`, from scratch or from MACE-MP/MPA/OMAT/MATPES,
  MACE-OFF, MACE-OMOL, MACE-MH-1 and MACE-POLAR-1; shared or state-specific
  electrostatics.
* **Fine-tuning.** Full fine-tuning, frozen layers, per-group learning rates, LoRA,
  multi-head fine-tuning with replay and pseudolabels; a trained X-MACE model is
  itself a foundation (continue on more data, or transfer to new states).
* **Heads.** Several excited-state and ground-state data sets (different methods,
  numbers of states or multiplicities) in one model.
* **Scale.** Multi-GPU data-parallel training, sharded HDF5 data, restartable runs,
  and a recipe for training an excited-state foundation model
  ([`foundation/`](foundation/README.md)).
* **Dynamics.** ASE calculator (all states, committees), and a SHARC 4.1 interface
  (`scripts/sharc/SHARC_MACE.py`, `sharc.x` and pysharc).

## Installation

```bash
git clone https://github.com/rhyan10/X-MACE.git && cd X-MACE
pip install -e ".[test]"          # Python >= 3.11, PyTorch >= 2.5
```

Optional, for some backbones and options:

```bash
pip install "git+https://github.com/ChengUCB/les.git@fadfb0ca3b89ed371e5de6b383f4425defabd4a8"  # MACELES
pip install graph_longrange       # MACE-POLAR-1 / PolarMACE
pip install "sphericart[torch]"   # magnetic models
pip install schedulefree          # --optimizer=schedulefree
```

The Python package is `mace` (X-MACE keeps the MACE library intact), so it replaces
an installed `mace-torch`; use a separate environment if you need both.

## Quick start

Data: extended XYZ with `REF_energy` `[S]`, `REF_forces` `[n_atoms, S, 3]` and,
optionally, `REF_nacs`, `REF_socs`, `REF_dipoles` in `atoms.info`.

```bash
# train: 3 singlets + 2 triplets, NACs, SOCs, dipoles, starting from MACE-OFF
mace_run_train --name=mol --model=ExcitedMACE --states="3 0 2" \
    --compute_nacs --compute_socs --compute_dipoles \
    --train_file=train.xyz --valid_fraction=0.1 --test_file=test.xyz --E0s=average \
    --foundation_model=small_off --max_num_epochs=200 --device=cuda

# evaluate
mace_eval_configs --configs=test.xyz --model=mol.model --output=pred.xyz
```

```python
from mace.calculators import ExcitedMACECalculator
calc = ExcitedMACECalculator("mol.model", state=1)     # ASE: energy/forces of S1
atoms.calc = calc
atoms.get_potential_energy()
calc.results["energies"], calc.results["nacs"], calc.results["soc_matrix"]
```

Several GPUs:

```bash
torchrun --standalone --nproc_per_node=4 -m mace.cli.run_train --distributed --launcher=torchrun ...
```

## Documentation

* [`XMACE.md`](XMACE.md) — the full reference: data layout, multiplicities and SOCs,
  every option, backbones, heads, foundation models, fine-tuning, multi-GPU,
  evaluation, SHARC.
* [`tutorials/xmace/`](tutorials/xmace/README.md) — eight notebooks, from
  excited-state physics and data generation with SHARC to training, dynamics,
  foundation models, fine-tuning strategies and oscillator strengths.
* [`foundation/`](foundation/README.md) — training an excited-state foundation
  model on many data sets (scripts, configs, SLURM, dry run).

## Tests

```bash
pytest tests                       # ~20 min on a CPU
pytest tests -m "not slow"         # unit tests only, ~5 min
source $SHARC/sharcvars.sh && pytest tests/test_sharc_real.py   # with a SHARC installation
```

## Relation to MACE

X-MACE contains the MACE library (from ACEsuit/mace `develop`, commit `7215fadd`)
with the excited-state code added: `mace/modules/{excited,excited_loss,spin_states}.py`,
`mace/data/{excited,sharc_labels}.py`, `mace/calculators/{excited,sharc_calculator}.py`,
and hooks in the training,
evaluation and preprocessing scripts. Ground-state MACE models train and run as in
MACE.

## References

If you use X-MACE, please cite the X-MACE papers:

```bibtex
@article{barrett2025transferable,
  title={Transferable machine learning potential X-MACE for excited states using integrated DeepSets},
  author={Barrett, Rhyan and Ortner, Christoph and Westermayr, Julia},
  journal={arXiv preprint arXiv:2502.12870},
  year={2025}
}

@article{barrett2026transferable,
  title={Transferable excited-state dynamics enable screening of fluorescent protein chromophores},
  author={Barrett, Rhyan and Wesely, Sophia and Westermayr, Julia},
  journal={arXiv preprint arXiv:2604.12699},
  year={2026}
}
```

and MACE:

```bibtex
@inproceedings{Batatia2022mace,
  title={{MACE}: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author={Ilyes Batatia and David Peter Kovacs and Gregor N. C. Simm and Christoph Ortner and Gabor Csanyi},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022},
  url={https://openreview.net/forum?id=YPpSngE-ZU}
}

@misc{Batatia2022Design,
  title = {The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials},
  author = {Batatia, Ilyes and Batzner, Simon and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Musaelian, Albert and Simm, Gregor N. C. and Drautz, Ralf and Ortner, Christoph and Kozinsky, Boris and Cs{\'a}nyi, G{\'a}bor},
  year = {2022},
  eprint = {2205.06643},
  archiveprefix = {arXiv}
}
```

Foundation models come with their own licenses and references (see their release
pages); SHARC is distributed separately.

## License

MIT License (see [LICENSE.md](LICENSE.md)), as MACE.
