import ase.io
import numpy as np

db = ase.io.read("CH3I_valence_eV.xyz", ":")
mols = []
for mol in db:
    mol.info["REF_scalar"] = np.random.uniform(-1, 1, (1, 1))
    mol.info["REF_socs"] = np.random.uniform(-1, 1, (8, 3))
    mol.info["REF_dipoles"] = np.random.uniform(-1, 1, (11, 3))
    mol.info["REF_nacs"] = np.random.uniform(-1, 1, (6, 3))
    mol.info["REF_oscillator"] = np.random.uniform(-1, 1, (4,))
    mols.append(mol)

ase.io.write("test.xyz", mols)
