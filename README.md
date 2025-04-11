# X‑MACE

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)   
[![Python Version](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)

**X‑MACE** is a deep learning framework designed to model excited‐state potential energy surfaces with high accuracy, especially near conical intersections. It extends the Message Passing Atomic Cluster Expansion (MACE) architecture by integrating Deep Sets to learn smooth representations of inherently non‐smooth energy surfaces. An overview of the main parameters used in the MACE architecture can be found [here](https://github.com/ACEsuit/mace).

---

## Overview

X‑MACE provides researchers with a powerful tool to train models that predict excited-state properties. The framework is tailored to handle multiple energy levels, forces, non-adiabatic couplings (NACs), and dipole moments, making it ideal for studying systems where excited-state dynamics are crucial.

Key features include:
- **Multi-level Energy Learning:** Use the `--n_energies` parameter to specify how many energy levels the model should learn.
- **Error Analysis:** Generate error tables for energies, forces, NACs, and dipole moments with the `--error_table` option and the associated table `EnergyNacsDipoleMAE`
- **Transfer Learning Capability:** Leverage pre-trained ground state representations by specifying the `--foundation_model` parameter.

---

## Documentation

Detailed documentation is in progress. In the meantime, usage examples provided below should help you get started. For more information on MACE-related parameters, please refer to the [MACE repository](https://github.com/ACEsuit/mace).

---

## System Requirements

### Hardware
- A standard computer with enough RAM for deep learning computations however a GPU enabled machine is strongly recommended. 

### Software
- **Operating System:** Linux, macOS, or Windows (best used with a conda environment)
- **Python:** 3.7 or higher
- **Dependencies:** X‑MACE relies on the typical deep learning and scientific computing stack in Python. All dependencies are installed during installation of the library 

---

## Installation

Ensure that Python 3.7+ is installed in your environment. To install X‑MACE and its dependencies clone the github repo and install locally. The installation should only take a few minutes on a normal computer. The following commands illustate this:

```bash
git clone https://github.com/your-repo/x-mace.git
cd x-mace
pip install .
```
It is also highly recommended to create a python environment beforehand, This can be done using the following commands:

```bash
# Clone the repository
git clone https://github.com/rhyan10/x-mace.git
cd x-mace

# Create and activate a new Python virtual environment using conda
conda create --name x-mace-env python=3.8 -y
conda activate x-mace-env

# Install dependencies and X‑MACE
pip install .
``` 

---

## Usage

The following commands allow training of the machine learning models mentioned in the paper with and without the autoencoder and with and without transfer learning. Output files containing the loss and validation errors can be seen in the results folder. Run time will depend on the size of the architecture as well as the number of GPUs available but normally will take less than 1 day. The following commands can be replaced with the relevant datasets and number of energy levels.  

### Training the X‑MACE Model (with Autoencoder)

```bash
python3 scripts/run_train.py --name="model1" --train_file="singlet_chromophores.xyz" --seed=100 --valid_fraction=0.1 --E0s='average' --model="AutoencoderExcitedMACE" --r_max=5.0 --batch_size=100 --n_energies=5 --correlation=3 --max_num_epochs=100 --ema --lr=0.0001 --ema_decay=0.99 --default_dtype="float32" --device=cuda --hidden_irreps="128x0e + 128x1o" --MLP_irreps='128x0e' --num_radial_basis=8 --num_interactions=2 --energy_weight=100.0 --error_table="EnergyNacsDipoleMAE"
```

### Training the E‑MACE Model (without Autoencoder)

```bash
python3 scripts/run_train.py --name="model1" --train_file="singlet_chromophores.xyz" --seed=100 --valid_fraction=0.1 --E0s='average' --model="ExcitedMACE" --r_max=5.0 --batch_size=100 --n_energies=5 --correlation=3 --max_num_epochs=100 --ema --lr=0.0001 --ema_decay=0.99 --default_dtype="float32" --device=cuda --hidden_irreps="128x0e + 128x1o" --MLP_irreps='128x0e' --num_radial_basis=8 --num_interactions=2 --energy_weight=100.0 --error_table="EnergyNacsDipoleMAE"
```

### Transfer Learning

```bash
python3 scripts/run_train.py --name="model1" --train_file="singlet_chromophores.xyz" --seed=100 --valid_fraction=0.1 --foundation_model="medium_off" --E0s='average' --model="ExcitedMACE" --r_max=5.0 --batch_size=100 --n_energies=5 --correlation=3 --max_num_epochs=100 --ema --lr=0.0001 --ema_decay=0.99 --default_dtype="float32" --device=cuda --hidden_irreps="128x0e + 128x1o" --MLP_irreps='128x0e' --num_radial_basis=8 --num_interactions=2 --energy_weight=100.0 --error_table="EnergyNacsDipoleMAE"
```

---

## Datasets

The datasets used for developing and benchmarking X‑MACE are available from publications references in the X-MACE publication. You can also access some of them directly via this link:  
[Datasets for X‑MACE Publication](https://figshare.com/articles/dataset/Datasets_for_X-MACE_Publication/28425173).

---

## License

This project is licensed under the MIT License

---

