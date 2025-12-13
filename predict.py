import ase.io
from mace.calculators.mace import MACECalculator
import pickle
from tqdm import tqdm

db = ase.io.read("structures_with_dipoles.xyz", ":")
predictions = []
reference = []

model_path = "dipoles.model"

calc = MACECalculator(model_paths=model_path, n_energies=1,device="cuda", default_dtype="float32")

results_list = []

for mol in tqdm(db):
    calc.calculate(mol)
    energy = calc.results["energy"]
    forces = calc.results["forces"]
    dipoles = calc.results["dipoles"]
    charges = calc.results["charges"]


