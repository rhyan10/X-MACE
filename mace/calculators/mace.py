from typing import Dict, List, Union

import ase
import numpy as np
import torch

from mace.calculators import MACECalculator


class SharcCalculator:
    """Bridge between a trained (X-)MACE model and the SHARC dynamics driver.

    SHARC hands over coordinates each timestep and expects back the electronic
    Hamiltonian, gradients and (optionally) nonadiabatic and spin-orbit couplings,
    all in atomic units. This class runs the model and repackages its output into
    the dictionary layout SHARC expects.
    """

    # Conversion factors into atomic units (Hartree / Bohr).
    DISTANCE_TO_BOHR = {"Ang": 0.529177249, "Bohr": 1.0}
    ENERGY_TO_HARTREE = {"eV": 0.0367493, "Hartree": 1.0}

    def __init__(
        self,
        atom_types,
        model_path: str,
        device: str,
        energy_unit: str,
        distance_unit: str,
        n_states: Dict[str, int] = None,
        properties: List[str] = None,
        nac_key: str = None,
    ):
        """
        atom_types     : element symbols, one per atom
        model_path     : path to the trained .model file
        device         : "cpu" or "cuda"
        energy_unit    : units the model was trained in ("eV" or "Hartree")
        distance_unit  : units the model was trained in ("Ang" or "Bohr")
        n_states       : {"n_singlets": int, "n_triplets": int}
        properties     : model outputs to request, e.g. ["energy", "forces", "smooth_nacs"]
        nac_key        : which entry of `properties` holds the NAC vectors. If not
                         given, it is inferred (any property containing "nac"),
                         falling back to "smooth_nacs".
        """
        self.energy_unit_conversion = self.ENERGY_TO_HARTREE[energy_unit]
        self.distance_unit_conversion = self.DISTANCE_TO_BOHR[distance_unit]

        self.properties = properties or []
        self.atom_types = atom_types
        self.n_atoms = len(atom_types)
        self.molecule = ase.Atoms(symbols=atom_types)

        # Datasets disagree on the NAC key ("smooth_nacs" vs "REF_smooth_nacs" ...),
        # so infer it rather than hard-coding one.
        self.nac_key = nac_key or next(
            (p for p in self.properties if "nac" in p.lower()), "smooth_nacs"
        )

        self.n_states = n_states
        # Each triplet appears three times in the SHARC basis (Ms = -1, 0, +1).
        self.n_total_states = n_states["n_singlets"] + 3 * n_states["n_triplets"]
        self.nac_idx = np.triu_indices(n_states["n_singlets"], 1)
        self.soc_idx = np.triu_indices(self.n_total_states, 1)

        self.calc = MACECalculator(
            model_paths=model_path,
            n_energies=self.n_total_states,
            device=device,
        )

    def calculate(
        self, sharc_coords: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, List[np.ndarray]]:
        """Run the model at the given geometry and return SHARC-formatted output."""
        # SHARC works in Bohr; convert to whatever the model was trained on.
        self.molecule.set_positions(
            np.array(sharc_coords) * self.distance_unit_conversion
        )
        self.calc.calculate(self.molecule)
        mace_output = self.calc.results

        # Trim to the number of distinct electronic states and convert to a.u.
        n_distinct = self.n_states["n_singlets"] + self.n_states["n_triplets"]
        mace_output["energy"] = (
            mace_output["energy"][0][:n_distinct] * self.energy_unit_conversion
        )
        mace_output["forces"] = (
            mace_output["forces"][:, :n_distinct, :]
            * self.energy_unit_conversion
            * self.distance_unit_conversion
        )
        return self.get_qm(mace_output)

    def get_qm(self, mace_output: Dict[str, np.ndarray]) -> Dict[str, List]:
        """Repackage model predictions into the dictionary SHARC consumes."""
        states = self.n_total_states
        n_singlets = self.n_states["n_singlets"]
        n_triplets = self.n_states["n_triplets"]
        qm_out = {}

        # Hamiltonian: state energies on the diagonal.
        h = np.diag(np.array(mace_output["energy"], dtype=complex))

        # Gradients are -forces, reshaped [atoms, states, xyz] -> [states, atoms, xyz].
        qm_out["grad"] = np.einsum("ijk->jik", -mace_output["forces"]).tolist()

        if self.nac_key in self.properties:
            # NACs come per state pair; rebuild the antisymmetric state-by-state matrix.
            nacs_v = np.einsum("ijk->jik", mace_output[self.nac_key])
            nacs_m = np.zeros((states, states, self.n_atoms, 3))

            if n_triplets == 0:
                nacs_m[self.nac_idx] = nacs_v
                nacs_m -= np.transpose(nacs_m, axes=(1, 0, 2, 3))
            else:
                n_singlet_pairs = n_singlets * (n_singlets - 1) // 2

                nacs_singlet = np.zeros((n_singlets, n_singlets, self.n_atoms, 3))
                nacs_singlet[self.nac_idx] = nacs_v[:n_singlet_pairs]
                nacs_singlet -= np.transpose(nacs_singlet, axes=(1, 0, 2, 3))
                nacs_m[:n_singlets, :n_singlets] = nacs_singlet

                nacs_trip_sub = np.zeros((n_triplets, n_triplets, self.n_atoms, 3))
                nacs_trip_sub[np.triu_indices(n_triplets, 1)] = nacs_v[n_singlet_pairs:]
                nacs_trip_sub -= np.transpose(nacs_trip_sub, axes=(1, 0, 2, 3))

                # Replicate the triplet block across the three Ms components.
                nacs_trip = np.zeros(
                    (3 * n_triplets, 3 * n_triplets, self.n_atoms, 3)
                )
                for i in range(3):
                    for j in range(i, 3):
                        nacs_trip[
                            i * n_triplets : (i + 1) * n_triplets,
                            j * n_triplets : (j + 1) * n_triplets,
                        ] = nacs_trip_sub

                nacs_trip[np.tril_indices(3 * n_triplets)] = 0
                nacs_trip -= np.transpose(nacs_trip, axes=(1, 0, 2, 3))
                nacs_m[n_singlets:, n_singlets:] = nacs_trip

            qm_out["nacdr"] = nacs_m.tolist()

        if "socs" in self.properties:
            # Spin-orbit couplings fill the off-diagonal of the Hamiltonian.
            soc_m = np.zeros((states, states), dtype=complex)
            soc_m[self.soc_idx] = mace_output["socs"]
            soc_m += soc_m.T
            h = h + soc_m

        qm_out["h"] = h.tolist()
        return qm_out
