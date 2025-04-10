import ase.io
import numpy as np

db = ase.io.read("db_train_1.xyz", ":")

print(db[0].info["REF_energies"].shape)

db2 = ase.io.read("test.xyz", ":")

print(db2[0].info["REF_energy"].shape)

correct_mols = []
for mol in db:
    mol.info["REF_scalar"] = np.full((len(mol), 1),  mol.info["REF_scalar"])
    mol.info["REF_energy"] = mol.info["REF_energies"].reshape(1,21)
    correct_mols.append(mol)

ase.io.write("db_train_1_c.xyz", correct_mols)


