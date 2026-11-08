###########################################################################################
# SHARC interface layer for excited-state MACE models (X-MACE)
#
# Converts model predictions (eV, Angstrom) into the QM.out quantities SHARC 4 needs,
# in atomic units: h (energies + SOCs), grad, nacdr and dm, for singlets and triplets.
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from typing import Dict, List, Optional, Sequence, Union

import ase
import numpy as np

from .excited import ExcitedMACECalculator, coupling_signs

BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_EV = 27.211386245988

ENERGY_UNITS_TO_HARTREE = {"eV": 1.0 / HARTREE_TO_EV, "Hartree": 1.0}
DISTANCE_UNITS_PER_BOHR = {
    "Ang": BOHR_TO_ANGSTROM,
    "Angstrom": BOHR_TO_ANGSTROM,
    "Bohr": 1.0,
}

NAC_PROPERTIES = ("nacs", "smooth_nacs")


class SharcCalculator:
    """Predict the SHARC QM.out quantities with an excited-state MACE model.

    atom_types     element symbols in SHARC order
    model_paths    model file(s) (several: committee)
    n_states       {"n_singlets": ns, "n_triplets": nt}; SHARC has nmstates = ns + 3 nt
    properties     any of "energy", "forces", "nacs" (physical NACs) or "smooth_nacs"
                   (NAC * dE, as the model predicts them), "socs", "dipoles"
    energy_unit, distance_unit   units the model was trained in

    Two kinds of models are supported:
      spatial      n_energies = ns + nt: singlets, then triplets. Every M_S component
                   of a triplet gets the triplet's energy and gradient; NACs and dipoles
                   couple states of the same multiplicity and M_S only.
      M_S-resolved n_energies = ns + 3 nt, in SHARC order.
    A singlet-only run may use a model with more states; the lowest ns are used.
    SOCs are the upper triangle of the nmstates x nmstates SOC matrix (X-MACE layout):
    nmstates(nmstates-1)/2 real values, or nmstates(nmstates-1) values as (re, im).

    Couplings have an arbitrary sign; they are aligned with the previous call
    (phase continuity), so SHARC can be told that all phases are +1.
    """

    def __init__(
        self,
        atom_types: Sequence[str],
        model_paths: Union[str, Sequence[str]],
        device: str = "cpu",
        energy_unit: str = "eV",
        distance_unit: str = "Ang",
        n_states: Optional[Dict[str, int]] = None,
        properties: Optional[List] = None,
        nac_to_au: Optional[float] = None,
        soc_to_au: Optional[float] = None,
        dipole_to_au: Optional[float] = None,
        calculator: Optional[ExcitedMACECalculator] = None,
    ):
        if energy_unit not in ENERGY_UNITS_TO_HARTREE:
            raise ValueError(
                f"energy_unit must be one of {list(ENERGY_UNITS_TO_HARTREE)}"
            )
        if distance_unit not in DISTANCE_UNITS_PER_BOHR:
            raise ValueError(
                f"distance_unit must be one of {list(DISTANCE_UNITS_PER_BOHR)}"
            )
        # SHARC's template parser nests list keywords: [["energy", "forces", "nacs"]]
        if properties and isinstance(properties[0], (list, tuple)):
            properties = list(properties[0])
        self.properties = list(properties or ["energy", "forces"])
        unknown = set(self.properties) - {
            "energy",
            "forces",
            "socs",
            "dipoles",
            *NAC_PROPERTIES,
        }
        if unknown:
            raise ValueError(f"Unknown properties {sorted(unknown)}")
        self.nac_key = next((p for p in self.properties if p in NAC_PROPERTIES), None)

        n_states = n_states or {"n_singlets": 1, "n_triplets": 0}
        self.n_singlets = int(n_states.get("n_singlets", 0))
        self.n_triplets = int(n_states.get("n_triplets", 0))
        self.nmstates = self.n_singlets + 3 * self.n_triplets
        if self.nmstates < 1:
            raise ValueError("No electronic states requested")

        self.energy_to_au = ENERGY_UNITS_TO_HARTREE[energy_unit]
        self.length_per_bohr = DISTANCE_UNITS_PER_BOHR[
            distance_unit
        ]  # model length / bohr
        self.force_to_au = self.energy_to_au * self.length_per_bohr
        if nac_to_au is None:
            # physical NACs are 1/length; smooth NACs are energy/length
            nac_to_au = self.length_per_bohr
            if self.nac_key == "smooth_nacs":
                nac_to_au = self.force_to_au
        self.nac_to_au = nac_to_au
        self.soc_to_au = self.energy_to_au if soc_to_au is None else soc_to_au
        self.dipole_to_au = (
            1.0 / self.length_per_bohr if dipole_to_au is None else dipole_to_au
        )

        self.molecule = ase.Atoms(symbols=list(atom_types))
        self.n_atoms = len(self.molecule)
        self.calc = calculator or ExcitedMACECalculator(
            model_paths=model_paths, device=device
        )
        self.n_model_states = self.calc.n_states
        self._setup_state_map()
        self._check_properties()
        self._previous: Dict[str, np.ndarray] = {}

    # ----------------------------------------------------------------------------------
    def _setup_state_map(self):
        ns, nt, nm = self.n_singlets, self.n_triplets, self.nmstates
        n_model = self.n_model_states
        if n_model == nm:
            self.mode = "ms_resolved" if nt > 0 else "singlets"
            self.used_states = list(range(nm))
            self.state_map = [[k] for k in range(nm)]
            self.multiplicity = [1] * ns + [3] * (3 * nt)
        elif nt > 0 and n_model == ns + nt:
            self.mode = "spatial"
            self.used_states = list(range(ns + nt))
            self.state_map = [[i] for i in range(ns)] + [
                [ns + m * nt + a for m in range(3)] for a in range(nt)
            ]
            self.multiplicity = [1] * ns + [3] * nt
        elif nt == 0 and n_model > ns:
            self.mode = "singlets"
            self.used_states = list(range(ns))
            self.state_map = [[i] for i in range(ns)]
            self.multiplicity = [1] * ns
            logging.info(f"Model predicts {n_model} states; using the lowest {ns}")
        else:
            raise ValueError(
                f"The model predicts {n_model} states, but SHARC asks for {ns} singlets and "
                f"{nt} triplets: use a model with {ns + nt} (spatial) or {nm} "
                "(M_S-resolved) states"
            )
        self.model_pairs = list(zip(*np.triu_indices(self.n_model_states, k=1)))

    def _check_properties(self):
        model = self.calc.models[0]
        if self.nac_key is not None:
            if not model.compute_nacs:
                raise ValueError(
                    "NACs requested, but the model was trained without NACs"
                )
            if self.nac_key == "nacs" and not self.calc.nacs_per_pair:
                raise ValueError(
                    "Physical NACs need one NAC per state pair; this model has a custom "
                    "NAC layout, use 'smooth_nacs'"
                )
        if "dipoles" in self.properties and not model.compute_dipoles:
            raise ValueError(
                "Dipoles requested, but the model was trained without dipoles"
            )
        if "socs" in self.properties:
            if self.nmstates == self.n_singlets:
                return  # singlets only: no SOCs
            if not model.compute_socs:
                raise ValueError(
                    "SOCs requested, but the model was trained without SOCs"
                )
            n_upper = self.nmstates * (self.nmstates - 1) // 2
            if int(model.soc_num) not in (n_upper, 2 * n_upper):
                raise ValueError(
                    f"SOCs: the model predicts {int(model.soc_num)} values, SHARC needs the "
                    f"upper triangle of a {self.nmstates}x{self.nmstates} matrix: "
                    f"{n_upper} real values or {2 * n_upper} (re, im) values"
                )

    # ----------------------------------------------------------------------------------
    def _align_phases(self, results: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for key, axis in (("smooth_nacs", 1), ("nacs", 1)):
            if key in results:
                if key in self._previous:
                    signs = coupling_signs(self._previous[key], results[key], axis)
                    results["smooth_nacs"] = (
                        results["smooth_nacs"] * signs[None, :, None]
                    )
                    if "nacs" in results:
                        results["nacs"] = results["nacs"] * signs[None, :, None]
                break
        if "socs" in results and "socs" in self._previous:
            if np.sum((results["socs"] + self._previous["socs"]) ** 2) < np.sum(
                (results["socs"] - self._previous["socs"]) ** 2
            ):
                results["socs"] = -results["socs"]
        if "dipoles" in results:
            n_perm = self.n_model_states
            if results["dipoles"].shape[0] > n_perm and "dipoles" in self._previous:
                transition = results["dipoles"][n_perm:]
                signs = coupling_signs(
                    self._previous["dipoles"][n_perm:], transition, 0
                )
                results["dipoles"] = np.concatenate(
                    [results["dipoles"][:n_perm], transition * signs[:, None]]
                )
        for key in ("smooth_nacs", "nacs", "socs", "dipoles"):
            if key in results:
                self._previous[key] = results[key].copy()
        return results

    def reset_phases(self):
        self._previous = {}

    def predict(self, sharc_coords) -> Dict[str, np.ndarray]:
        """Model predictions (model units, phase aligned) for coordinates in bohr."""
        self.molecule.set_positions(
            np.asarray(sharc_coords, dtype=float) * self.length_per_bohr
        )
        return self._align_phases(self.calc.predict(self.molecule))

    def calculate(self, sharc_coords) -> Dict[str, np.ndarray]:
        """QM.out quantities in atomic units: h [nm, nm] (complex), grad [nm, N, 3],
        and if requested nacdr [nm, nm, N, 3] and dm [3, nm, nm]."""
        results = self.predict(sharc_coords)
        return self.get_qm(results)

    # ----------------------------------------------------------------------------------
    def get_qm(self, results: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        nm, n_atoms = self.nmstates, self.n_atoms
        used = self.used_states
        energies = results["energies"][used] * self.energy_to_au
        grads = -results["forces_states"][:, used, :] * self.force_to_au  # [N, S, 3]

        h = np.zeros((nm, nm), dtype=complex)
        grad = np.zeros((nm, n_atoms, 3))
        for k, targets in enumerate(self.state_map):
            for t in targets:
                h[t, t] = energies[k]
                grad[t] = grads[:, k, :]
        qm_out: Dict[str, np.ndarray] = {"h": h, "grad": grad}

        if self.nac_key is not None:
            nacs = results[self.nac_key] * self.nac_to_au  # [N, P_model, 3]
            nacdr = np.zeros((nm, nm, n_atoms, 3))
            for p, (k, l) in enumerate(self.model_pairs):
                if k >= len(self.state_map) or l >= len(self.state_map):
                    continue
                if self.multiplicity[k] != self.multiplicity[l]:
                    continue
                for tk, tl in zip(self.state_map[k], self.state_map[l]):
                    nacdr[tk, tl] = nacs[:, p, :]
                    nacdr[tl, tk] = -nacs[:, p, :]
            qm_out["nacdr"] = nacdr

        if "dipoles" in self.properties:
            dipoles = results["dipoles"] * self.dipole_to_au  # [D, 3]
            dm = np.zeros((3, nm, nm))
            n_model = self.n_model_states
            for k, targets in enumerate(self.state_map):
                for t in targets:
                    dm[:, t, t] = dipoles[k]
            if dipoles.shape[0] >= n_model + len(self.model_pairs):
                for p, (k, l) in enumerate(self.model_pairs):
                    if k >= len(self.state_map) or l >= len(self.state_map):
                        continue
                    if self.multiplicity[k] != self.multiplicity[l]:
                        continue
                    for tk, tl in zip(self.state_map[k], self.state_map[l]):
                        dm[:, tk, tl] = dipoles[n_model + p]
                        dm[:, tl, tk] = dipoles[n_model + p]
            qm_out["dm"] = dm

        if "socs" in self.properties and nm > self.n_singlets:
            socs = results["socs"] * self.soc_to_au
            iu = np.triu_indices(nm, k=1)
            if socs.size == len(iu[0]):
                values = socs.astype(complex)
            else:
                values = socs[0::2] + 1j * socs[1::2]
            soc_matrix = np.zeros((nm, nm), dtype=complex)
            soc_matrix[iu] = values
            qm_out["h"] = h + soc_matrix + soc_matrix.conj().T
        return qm_out
