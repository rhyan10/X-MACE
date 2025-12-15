from ase import Atoms
from ase.io import write
import numpy as np

# Number of structures you want
n_structures = 200
n_dipoles = 5
n_atoms = 4

# Create list of structures
structures = []
for i in range(n_structures):
    # Example: random coordinates for a triatomic molecule
    positions = np.random.rand(n_atoms, 3) * 2.0
    symbols = ['H', 'O', 'H', 'H']
    
    atoms = Atoms(symbols=symbols, positions=positions)
    
    # Number of dipoles i.e. n_dipoles - 1 excited plus ground state
    ref_dipoles = np.random.randn(n_dipoles, 3)
    
    # Store it as part of the Atoms.info dictionary
    atoms.info["REF_energy"] = np.ones((1,3))
    atoms.info["REF_dipoles"] = ref_dipoles.tolist()  # convert to list for JSON compatibility
    
    structures.append(atoms)

# Write all to an extended XYZ file
write("structures_with_dipoles.xyz", structures)

