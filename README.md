# X‑MACE with SOC elements

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)

**X‑MACE** is a deep learning framework designed to model excited‐state potential energy surfaces with high accuracy, especially near conical intersections. It extends the Message Passing Atomic Cluster Expansion (MACE) architecture by integrating Deep Sets to learn smooth representations of inherently non‐smooth energy surfaces. An overview of the main parameters used in the MACE architecture can be found [here](https://github.com/ACEsuit/mace).

---

## Overview

The framework is tailored to handle multiple energy levels, forces, non-adiabatic couplings (NACs), and dipole moments, making it useful for studying systems where excited-state dynamics are crucial.

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
git clone https://github.com/rhyan10/X-MACE.git
cd x-mace
git checkout X-MACE_socs
pip install .
```
It is also highly recommended to create a python environment beforehand, This can be done using the following commands:

```bash
# Clone the repository
git clone https://github.com/rhyan10/X-MACE.git
git checkout X-MACE_socs
cd x-mace

# Create and activate a new Python virtual environment using conda
conda create --name x-mace-env python=3.13.2 -y
conda activate x-mace-env

# Install dependencies and X-MACE
pip install .
```

---

## Usage

The following commands allow training of the machine learning models mentioned in the paper with and without the autoencoder and with and without transfer learning (note in this repo the autoencoder only trains energies and forces). All the commands can be used for E-MACE and the Autoencoder Model. Important to note that the EMACE model will expect energies so make sure they are contained in your dataset along with the properties. Output files containing the loss and validation errors can be seen in the results folder. Run time will depend on the size of the architecture as well as the number of GPUs available but normally will take less than 1 day. The following commands can be replaced with the relevant datasets and number of energy levels.

### Training the X‑MACE Model (with Autoencoder)

This model can be used to train only energies and forces. It contains a matrix diagonalization operation which may improve the energy predictions.

```bash
python scripts/run_train.py --name="energies_forces" --train_file="SINGLET_SOC_ALL.xyz" --seed=100 --valid_fraction=0.1 --E0s='average' --model="AutoencoderExcitedMACE" --r_max=5.0 --batch_size=10 --n_energies=4 --correlation=3 --max_num_epochs=100 --ema --lr=0.0001 --ema_decay=0.99 --default_dtype="float32" --device=cuda --hidden_irreps="128x0e + 128x1o" --MLP_irreps='128x0e' --num_radial_basis=8 --num_interactions=2 --energy_weight=100.0 --forces_weight=100.0 --error_table="EnergyNacsDipoleMAE"
```

### Training the E‑MACE Model (without Autoencoder)

This model just contains a multi energy surface readout with any matrix diagonalization.

```bash
python scripts/run_train.py --name="energies_forces" --train_file="SINGLET_SOC_ALL.xyz" --seed=100 --valid_fraction=0.1 --E0s='average' --model="ExcitedMACE" --r_max=5.0 --batch_size=10 --n_energies=4 --correlation=3 --max_num_epochs=100 --ema --lr=0.0001 --ema_decay=0.99 --default_dtype="float32" --device=cuda --hidden_irreps="32x0e + 32x1o" --MLP_irreps='32x0e' --num_radial_basis=8 --num_interactions=2 --energy_weight=100.0 --forces_weight=1.0 --error_table="EnergyNacsDipoleMAE"
```

### Transfer Learning

To add include the parameters of a previous foundational model use this command. It is based on the medium off model which is a good compromise in terms of model size and information gained.

```bash
python scripts/run_train.py --name="energies_forces" --train_file="SINGLET_SOC_ALL.xyz" --seed=100 --valid_fraction=0.1 --E0s='average' --model="ExcitedMACE" --r_max=5.0 --batch_size=10 --n_energies=4 --correlation=3 --max_num_epochs=100 --ema --lr=0.0001 --ema_decay=0.99 --default_dtype="float32" --device=cuda --hidden_irreps="128x0e + 128x1o" --MLP_irreps='128x0e' --num_radial_basis=8 --num_interactions=2 --energy_weight=100.0 --forces_weight=1.0 --error_table="EnergyNacsDipoleMAE" --foundation_model="medium_off"
```

### Training NACS and SOCS

### Training NACS

This will just train nac values, the nac_num is the number of vectorial quantities (like nacs) that the model will predict. Due to the way this MACE model is set up please make sure you also include energy values in the dataset otherwise there may be some problems.

```bash
python scripts/run_train.py --name="nacs" --train_file="SINGLET_SOC_ALL.xyz" --seed=100 --valid_fraction=0.1 --E0s='average' --model="ExcitedMACE" --r_max=5.0 --batch_size=10 --n_energies=4 --correlation=3 --max_num_epochs=100 --ema --lr=0.0001 --ema_decay=0.99 --default_dtype="float32" --device=cuda --hidden_irreps="32x0e + 32x1o" --MLP_irreps='32x0e' --num_radial_basis=8 --num_interactions=2 --energy_weight=100.0 --forces_weight=0.0 --nacs_weight=100.0 --error_table="EnergyNacsDipoleMAE" --compute_nacs --nac_num=6 --nacs_key="REF_nacs"
```

If you only need to train NACs keep all the properties in the dataset but set the weighting of it in the loss function to zero for example:

```bash
python scripts/run_train.py --name="nacs" --train_file="SINGLET_SOC_ALL.xyz" --seed=100 --valid_fraction=0.1 --E0s='average' --model="ExcitedMACE" --r_max=5.0 --batch_size=10 --n_energies=4 --correlation=3 --max_num_epochs=100 --ema --lr=0.0001 --ema_decay=0.99 --default_dtype="float32" --device=cuda --hidden_irreps="32x0e + 32x1o" --MLP_irreps='32x0e' --num_radial_basis=8 --num_interactions=2 --energy_weight=0.0 --forces_weight=0.0 --nacs_weight=100.0  --socs_weight=0.0  --error_table="EnergyNacsDipoleMAE" --compute_nacs --nac_num=6 --nacs_key="REF_nacs"
```

Here you still have the nac, socs and energy values in the dataset but the energy, forces and socs weights are set to zero so do not contribute.

### Training SOCS

This will just train soc values, the soc_num is the number of socs that the model will predict. Due to the way this MACE model is set up please make sure you also include energy values in the dataset otherwise there may be some problems.

```bash
python scripts/run_train.py --name="socs" --train_file="SINGLET_SOC_ALL.xyz" --seed=100 --valid_fraction=0.1 --E0s='average' --model="ExcitedMACE" --r_max=5.0 --batch_size=10 --n_energies=4 --correlation=3 --max_num_epochs=100 --ema --lr=0.0001 --ema_decay=0.99 --default_dtype="float32" --device=cuda --hidden_irreps="32x0e + 32x1o" --MLP_irreps='32x0e' --num_radial_basis=8 --num_interactions=2 --energy_weight=100.0 --forces_weight=100.0 --nacs_weight=100.0 --socs_weight=100.0 --error_table="EnergyNacsDipoleMAE" --compute_socs --soc_num=252 
```

If you only need to train NACs keep all the properties in the dataset but set the weighting of it in the loss function to zero for example:

```bash
python scripts/run_train.py --name="socs" --train_file="SINGLET_SOC_ALL.xyz" --seed=100 --valid_fraction=0.1 --E0s='average' --model="ExcitedMACE" --r_max=5.0 --batch_size=10 --n_energies=4 --correlation=3 --max_num_epochs=100 --ema --lr=0.0001 --ema_decay=0.99 --default_dtype="float32" --device=cuda --hidden_irreps="32x0e + 32x1o" --MLP_irreps='32x0e' --num_radial_basis=8 --num_interactions=2 --energy_weight=0.0 --forces_weight=0.0 --nacs_weight=0.0 --socs_weight=100.0 --error_table="EnergyNacsDipoleMAE" --compute_socs --soc_num=252
```

Here you still have the nac, socs and energy values in the dataset but the energy, forces and socs weights are set to zero so do not contribute.

### Training SOCS and NACS

You only need to add the flag together at the end like so

```bash
python scripts/run_train.py --name="socs" --train_file="SINGLET_SOC_ALL.xyz" --seed=100 --valid_fraction=0.1 --E0s='average' --model="ExcitedMACE" --r_max=5.0 --batch_size=10 --n_energies=4 --correlation=3 --max_num_epochs=100 --ema --lr=0.0001 --ema_decay=0.99 --default_dtype="float32" --device=cuda --hidden_irreps="32x0e + 32x1o" --MLP_irreps='32x0e' --num_radial_basis=8 --num_interactions=2 --energy_weight=100.0 --forces_weight=100.0 --nacs_weight=100.0 --socs_weight=100.0 --error_table="EnergyNacsDipoleMAE" --compute_socs --soc_num=252 --compute_nacs --nac_num=6
```

## Datasets

The datasets used for developing and benchmarking X‑MACE are available from publications references in the X-MACE publication. You can also access some of them directly via this link:
[Datasets for X‑MACE Publication](https://figshare.com/articles/dataset/Datasets_for_X-MACE_Publication/28425173).

---

## License

This project is licensed under the MIT License

---
