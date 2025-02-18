X-MACE is a deep learning framework designed to model excited-state potential energy surfaces with high accuracy, especially near conical intersections. It extends the Message Passing Atomic Cluster Expansion (MACE) architecture by integrating Deep Sets to learn smooth representations of inherently non-smooth energy surfaces.
An overview of the main parameters used in the MACE architecture can be found here https://github.com/ACEsuit/mace.

Installation

Ensure you have Python 3.7+ and the necessary dependencies installed. Installation of the library as well as dependecies can be performed using:

pip install .

in an appropriate conda environment

Usage
Training X-MACE model example:
python3 scripts/run_train.py --name="model1" --train_file="singlet_chromophores.xyz" --seed=100 --valid_fraction=0.1 --E0s='average' --model="AutoencoderExcitedMACE" --r_max=5.0 --batch_size=100 --n_energies=5 --correlation=3 --max_num_epochs=100 --ema --lr=0.0001 --ema_decay=0.99 --default_dtype="float32" --device=cuda --hidden_irreps="128x0e + 128x1o" --MLP_irreps='128x0e' --num_radial_basis=8 --num_interactions=2 --energy_weight=100.0 --error_table="EnergyNacsDipoleMAE"
The parameters that differ from the current MACE repository include --n_energies, which specifies the number of energy levels to be learned, and --error_table="EnergyNacsDipoleMAE", which generates an error table for energies, forces, NACs, and dipole moments, depending on the available dataset contents.

Training an E-MACE model example :
python3 scripts/run_train.py --name="model1" --train_file="singlet_chromophores.xyz" --seed=100 --valid_fraction=0.1 --E0s='average' --model="ExcitedMACE" --r_max=5.0 --batch_size=100 --n_energies=5 --correlation=3 --max_num_epochs=100 --ema --lr=0.0001 --ema_decay=0.99 --default_dtype="float32" --device=cuda --hidden_irreps="128x0e + 128x1o" --MLP_irreps='128x0e' --num_radial_basis=8 --num_interactions=2 --energy_weight=100.0 --error_table="EnergyNacsDipoleMAE"
In order to run the excited state variant of the model without the autoencoder the run command differs by the --model parameter being ExcitedMACE rather than AutoencoderExcitedMACE.

Transfer learning:
python3 scripts/run_train.py --name="model1" --train_file="singlet_chromophores.xyz" --seed=100 --valid_fraction=0.1 --foundation_model="medium_off" --E0s='average' --model="ExcitedMACE" --r_max=5.0 --batch_size=100 --n_energies=5 --correlation=3 --max_num_epochs=100 --ema --lr=0.0001 --ema_decay=0.99 --default_dtype="float32" --device=cuda --hidden_irreps="128x0e + 128x1o" --MLP_irreps='128x0e' --num_radial_basis=8 --num_interactions=2 --energy_weight=100.0 --error_table="EnergyNacsDipoleMAE"
In order to perform transfer learning from the ground state representations the parameter --foundation_model="medium_off" must be added with the correct parameters equivalent to those of the foundational model being used. The model here can be adapted from "medium_off". 

The datasets used for this publication can be found in the relevant publications references in the paper or through the link https://figshare.com/articles/dataset/Datasets_for_X-MACE_Publication/28425173. 
