# X‑MACE with Dipole Elements  

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)

X‑MACE is a deep learning framework designed to model excited‐state potential energy surfaces with high accuracy, especially near conical intersections. It extends the Message Passing Atomic Cluster Expansion (MACE) architecture by integrating Deep Sets to learn smooth representations of inherently non‐smooth energy surfaces. An overview of the main parameters used in the MACE architecture can be found here.
(https://github.com/ACEsuit/mace).

Training the Dipoles Model
This model can be used to train only dipoles and producing charges.

```bash
python scripts/run_train.py --name="dipoles" --train_file="structures_with_dipoles.xyz" --seed=100 --valid_fraction=0.1 --E0s='average' --model="DipoleMACE" --r_max=5.0 --batch_size=10 --n_energies=0 --n_dipoles=5 --correlation=3 --max_num_epochs=100 --ema --lr=0.0001 --ema_decay=0.99 --default_dtype="float32" --device=cuda --hidden_irreps="128x0e + 128x1o" --MLP_irreps='128x0e' --num_radial_basis=8 --num_interactions=2 --dipoles_weight=100.0 --energy_weight=0.0 --forces_weight=0.0 --error_table="EnergyNacsDipoleMAE"
```

This is an example of training the dipoles using pseudo charges multiplied by position vectors (the molecules are centred at the centre of mass). This particular model can only train on dipoles. If you need to train energies and forces use the E-MACE and X-MACE models from the X_MACE_soc branch.

## License

This project is licensed under the MIT License

---
