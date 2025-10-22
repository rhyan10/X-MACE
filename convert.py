import ase.io
import numpy as np
import re, sys, json

fn = "tyrosine_diagonal_adaptive.xyz"
out = []
with open(fn, 'r', encoding='utf-8', errors='ignore') as f:
    for _ in range(5000):
        n = f.readline()
        if not n: break
        try: nat = int(n.strip())
        except: continue
        c = f.readline() or ""
        m = re.search(r'energy\s*=\s*("[^"]+"|\S+)', c)
        if m:
            raw = m.group(1).strip('"')
            if "_JSON" in raw:
                obj = json.loads(raw.split(None, 1)[1])
            else:
                obj = json.loads(raw) if raw[:1] in "[{" else [float(raw)]
            arr = obj[0] if isinstance(obj, list) and obj and isinstance(obj[0], list) else obj
            out.append(np.array(arr, float))
        for _ in range(nat): f.readline()

energies = np.stack(out)

from ase import Atoms

db2 = ase.io.read("tyrosine_diagonal_adaptive.xyz", ":5000")

new_db = []
for j, mol in enumerate(db2):
    new_atoms = Atoms(numbers=mol.numbers, positions=mol.positions*0.529177210903)
    mol.info["REF_energy"] = energies[j].reshape((1,13)) * 27.211386
    mol.info["REF_forces"] = mol.info["forces"] * 51.422
    mol.info["REF_socs"] = mol.info["socs"] * 27.211386
    new_db.append(mol)

ase.io.write("tyrosine_correct_l.xyz", new_db)
