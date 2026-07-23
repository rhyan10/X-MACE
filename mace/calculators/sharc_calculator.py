from glob import glob
from pathlib import Path
from typing import Union
import ase
import numpy as np
import torch
import os
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
import ase.io
from mace import data
from mace.modules.utils import extract_invariant
from mace.tools import torch_geometric, torch_tools, utils
from typing import Dict, Union, List
from mace.calculators import MACECalculator

symbols = ["", "H", "He", "Li", "Be", "B", "C", "N", "O", "F"]


class SharcCalculator:

    def __init__(
        self,
        atom_types,
        model_paths: str,
        device: str,
        energy_unit: str,
        distance_unit: str,
        n_states: Dict[str, int] = None,
        properties: List[str] = None,
    ):

        distance_units = {"Ang": 0.529177249, "Bohr": 1.0}
        energy_units = {"eV": 0.0367493, "Hartree": 1.0}
        self.energy_unit_conversion = energy_units[energy_unit]
        self.distance_unit_conversion = distance_units[distance_unit]
        # Load model and setup molecule
        self.model_paths = model_paths
        # SHARC's template parser nests list keywords: [["energy","forces","smooth_nacs"]]
        if properties and isinstance(properties[0], list):
            properties = properties[0]
        self.properties = properties or []
        self.nac_key = next((p for p in self.properties if "nac" in p), "smooth_nacs")
        self.molecule = ase.Atoms(symbols=atom_types)
        self.atom_types = atom_types
        self.n_states = n_states
        self.n_total_states = n_states["n_singlets"] + 3 * n_states["n_triplets"]
        self.nac_idx = np.triu_indices(self.n_states["n_singlets"], 1)
        self.soc_idx = np.triu_indices(self.n_total_states, 1)
        self.n_atoms = len(atom_types)
        self.calc = MACECalculator(model_paths=model_paths, n_energies=self.n_total_states, device=device)

    def calculate(
        self, sharc_coords: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, List[np.ndarray]]:
        """
        Calculate properties from new positions
        If multiple models are used, the average values between
        the two predictions with the lowest NAC MAE will be returned
        """
        mace_output = []
        # Update molecule positions and perform calculation
        self.molecule.set_positions(
            np.array(sharc_coords) * self.distance_unit_conversion
        )
        self.calc.calculate(self.molecule)
        mace_output = self.calc.results
        states_n = self.n_states["n_singlets"] + self.n_states["n_triplets"]
        mace_output["energy"] = mace_output["energy"][0][:states_n] * self.energy_unit_conversion
        mace_output["forces"] = mace_output["forces"][:,:states_n,:] * self.energy_unit_conversion * self.distance_unit_conversion
        qm_h = self.get_qm(mace_output)
        return qm_h

    def get_qm(self, mace_output: List[np.ndarray]) -> Dict[str, List[np.ndarray]]:
        """
        Calculate QM string for SHARC
        with predictions from model
        """

        states = self.n_total_states
        n_singlets = self.n_states["n_singlets"]
        n_triplets = self.n_states["n_triplets"]
        qm_out = {}
        # Convert energy array to complex diagonal matrix

        qm_out["h"] = np.diag(np.array(mace_output["energy"], dtype=complex)).tolist()
        # Reshape force array from [atoms, states, coords] to [states, atoms, coords]
        qm_out["grad"] = np.einsum("ijk->jik", -mace_output["forces"]).tolist()

        if self.nac_key in self.properties:
            nacs_v = np.einsum("ijk->jik", mace_output[self.nac_key])
            nacs_m = np.zeros((states, states, self.n_atoms, 3))

            if n_triplets == 0:
                nacs_m[self.nac_idx] = nacs_v
                nacs_m -= np.transpose(nacs_m, axes=(1, 0, 2, 3))
            else:
                nacs_singlet = np.zeros((n_singlets, n_singlets, self.n_atoms, 3))
                nacs_singlet[self.nac_idx] = nacs_v[
                    0 : int(n_singlets * (n_singlets - 1) / 2)
                ]
                nacs_singlet -= nacs_singlet.T

                nacs_m[0:n_singlets, 0:n_singlets] = nacs_singlet

                nacs_trip_sub = np.zeros((n_triplets, n_triplets, self.n_atoms, 3))
                sub_idx = np.triu_indices(n_triplets, 1)
                nacs_trip_sub[sub_idx] = nacs_v[int(n_singlets * (n_singlets - 1) / 2) :]
                nacs_trip_sub -= nacs_trip_sub.T

                nacs_trip = np.zeros((3 * n_triplets, 3 * n_triplets, self.n_atoms, 3))
                for i in range(3):
                    for j in range(i, 3):
                        nacs_trip[
                            i * n_triplets : (i + 1) * n_triplets,
                            j * n_triplets : (j + 1) * n_triplets,
                        ] = nacs_trip_sub

                trip_idx = np.tril_indices(3 * n_triplets)
                nacs_trip[trip_idx] = 0
                nacs_trip -= nacs_trip.T

                nacs_m[n_singlets:, n_singlets:] = nacs_trip

            qm_out["nacdr"] = nacs_m.tolist()
        
        if "socs" in self.properties:
            soc_m = np.zeros((states, states), dtype=complex)
            soc_m[self.soc_idx] = mace_output["socs"]
            soc_m += soc_m.T
            qm_out["h"] += soc_m

        return qm_out
