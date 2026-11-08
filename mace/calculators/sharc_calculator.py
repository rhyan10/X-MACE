###########################################################################################
# SHARC interface layer for excited-state MACE models (X-MACE)
#
# Converts model predictions (eV, Angstrom) into the QM.out quantities SHARC 4 needs,
# in atomic units: h (energies + SOCs), grad, nacdr and dm, for any multiplicities
# (singlets, doublets, triplets, quartets, quintets, ...).
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import ase
import numpy as np

from mace.modules.spin_states import (
    describe_states,
    ms_resolved_map,
    n_mstates,
    parse_states,
    soc_matrix_from_vectors,
    spin_orbit_pairs,
)

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


def sharc_states(
    states: Optional[Sequence[int]] = None, n_states: Optional[Dict[str, int]] = None
) -> List[int]:
    """SHARC states list from `states` ([n_singlets, n_doublets, ...] or '3 0 2') or the
    older {"n_singlets": ns, "n_triplets": nt} dictionary."""
    if states is not None:
        return parse_states(states)
    n_states = n_states or {"n_singlets": 1}
    names = (
        "n_singlets",
        "n_doublets",
        "n_triplets",
        "n_quartets",
        "n_quintets",
        "n_sextets",
        "n_septets",
    )
    unknown = set(n_states) - set(names)
    if unknown:
        raise ValueError(f"Unknown keys {sorted(unknown)} in n_states")
    return parse_states([int(n_states.get(name, 0)) for name in names])


class SharcCalculator:
    """Predict the SHARC QM.out quantities with an excited-state MACE model.

    atom_types     element symbols in SHARC order
    model_paths    model file(s) (several: committee)
    states         states per multiplicity requested by SHARC, [n_singlets, n_doublets,
                   n_triplets, ...] (spatial states; nmstates = sum (m+1) n_m)
    n_states       alternatively {"n_singlets": ns, "n_doublets": nd, "n_triplets": nt,...}
    properties     any of "energy", "forces", "nacs" (physical NACs) or "smooth_nacs"
                   (NAC * dE, as the model predicts them), "socs", "dipoles"
    energy_unit, distance_unit   units the model was trained in

    Models trained with --states (multiplicities) give, for every multiplicity, as many
    states as SHARC asks for or more (the lowest ones are used). The symmetry of the
    M_S components is built in:
      * all M_S components of a spatial state share its energy, gradient and
        permanent dipole;
      * NACs and transition dipoles couple states of equal multiplicity and equal M_S;
      * SOCs (soc_mode="spin_orbit") follow from one coupling vector per pair of spatial
        states by the Wigner-Eckart theorem, so the SOC matrix is Hermitian, obeys the
        spin selection rules and transforms correctly under rotations.
    Older models without multiplicities are read as follows: n_energies states of one
    multiplicity (singlets, or the only multiplicity SHARC asks for); n_singlets +
    n_triplets states (spatial, singlets then triplets); or nmstates states in SHARC
    order (M_S-resolved). Their SOCs are the upper triangle of the nmstates x nmstates
    matrix (X-MACE layout): nm(nm-1)/2 real values, or nm(nm-1) values as (re, im).

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
        states: Optional[Sequence[int]] = None,
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

        self.states = sharc_states(states, n_states)
        if not self.states:
            raise ValueError("No electronic states requested")
        self.nmstates = n_mstates(self.states)
        # kept for scripts written for the singlet/triplet version
        self.n_singlets = self.states[0]
        self.n_triplets = self.states[2] if len(self.states) > 2 else 0

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
        self.layout = self.calc.layout
        self.n_model_states = self.calc.n_states
        self._setup_state_map()
        self._check_properties()
        self._previous: Dict[str, np.ndarray] = {}

    # ----------------------------------------------------------------------------------
    def _model_states(self) -> Tuple[List[int], str]:
        """Multiplicity layout of the model and how its states map onto SHARC's."""
        model_states = list(self.layout.states)
        n_model, nm = self.n_model_states, self.nmstates
        sharc = self.states
        if len(model_states) > 1:
            return model_states, "multiplicities"
        # model without multiplicities: n_model states of one multiplicity
        multiplicities = [m for m, n in enumerate(sharc) if n > 0]
        if len(multiplicities) == 1:
            m = multiplicities[0]
            return [0] * m + [n_model], "multiplicities"
        if len(sharc) == 3 and sharc[1] == 0 and n_model == sharc[0] + sharc[2]:
            return [sharc[0], 0, sharc[2]], "multiplicities"  # X-MACE spatial
        if n_model == nm:
            return model_states, "ms_resolved"  # X-MACE: every M_S a model state
        raise ValueError(
            f"The model predicts {n_model} states without multiplicities, but SHARC asks "
            f"for {describe_states(sharc)}: train the model with --states"
        )

    def _setup_state_map(self):
        model_states, self.mode = self._model_states()
        self.model_states = model_states
        sharc_map = ms_resolved_map(self.states)  # per SHARC spatial state
        sharc_mult = [m + 1 for m, n in enumerate(self.states) for _ in range(n)]
        # slots[k]: SHARC (index, M_S) components of model state k (empty: unused)
        self.slots: List[List[Tuple[int, float]]] = [
            [] for _ in range(self.n_model_states)
        ]
        self.multiplicity = [0] * self.n_model_states
        if self.mode == "ms_resolved":
            # model state k is SHARC state k (multiplicity, M_S, state order)
            for components, mult in zip(sharc_map, sharc_mult):
                for index, ms in components:
                    self.slots[index] = [(index, ms)]
                    self.multiplicity[index] = mult
            self.sharc_to_model = list(range(self.nmstates))
            logging.info(
                "Model without multiplicities predicts every M_S component (X-MACE layout)"
            )
            return
        offsets = np.cumsum([0] + list(model_states))
        sharc_offsets = np.cumsum([0] + list(self.states))
        self.sharc_to_model = []  # SHARC spatial state -> model state
        for m, n in enumerate(self.states):
            available = model_states[m] if m < len(model_states) else 0
            if n > available:
                raise ValueError(
                    f"SHARC asks for {describe_states(self.states)}, but the model "
                    f"predicts {describe_states(model_states)}"
                )
            for a in range(n):
                k = int(offsets[m]) + a
                self.sharc_to_model.append(k)
                self.slots[k] = sharc_map[int(sharc_offsets[m]) + a]
                self.multiplicity[k] = m + 1
        if any(
            model_states[m] > (self.states[m] if m < len(self.states) else 0)
            for m in range(len(model_states))
        ):
            logging.info(
                f"Model predicts {describe_states(model_states)}; SHARC uses the lowest "
                f"{describe_states(self.states)}"
            )

    def _check_properties(self):
        model = self.calc.models[0]
        layout = self.layout
        if self.nac_key is not None:
            if not model.compute_nacs:
                raise ValueError(
                    "NACs requested, but the model was trained without NACs"
                )
            if layout.nac_pairs is None:
                raise ValueError(
                    f"The model predicts {int(model.nac_num)} NACs, not one per pair of "
                    "states of equal multiplicity, so SHARC cannot tell which states "
                    "they couple"
                )
        if "dipoles" in self.properties and not model.compute_dipoles:
            raise ValueError(
                "Dipoles requested, but the model was trained without dipoles"
            )
        self.soc_source = None
        if "socs" in self.properties and spin_orbit_pairs(self.states):
            if layout.soc_mode == "spin_orbit":
                if self.mode != "multiplicities":
                    raise ValueError("spin_orbit SOCs need a model with multiplicities")
                self.soc_source = "spin_orbit"
            elif bool(model.compute_socs):
                n_upper = self.nmstates * (self.nmstates - 1) // 2
                if int(model.soc_num) not in (n_upper, 2 * n_upper):
                    raise ValueError(
                        f"SOCs: the model predicts {int(model.soc_num)} values, SHARC "
                        f"needs the upper triangle of a {self.nmstates}x{self.nmstates} "
                        f"matrix: {n_upper} real values or {2 * n_upper} (re, im) values, "
                        "or a model trained with --soc_mode=spin_orbit"
                    )
                self.soc_source = "free"
            else:
                raise ValueError(
                    "SOCs requested, but the model was trained without SOCs"
                )

    # ----------------------------------------------------------------------------------
    def _align_phases(self, results: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for key in ("smooth_nacs", "nacs"):
            if key in results:
                if key in self._previous:
                    signs = coupling_signs(self._previous[key], results[key], 1)
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
        if "soc_vectors" in results and "soc_vectors" in self._previous:
            signs = coupling_signs(
                self._previous["soc_vectors"], results["soc_vectors"], 0
            )
            results["soc_vectors"] = results["soc_vectors"] * signs[:, None]
        if "dipoles" in results:
            n_perm = self.layout.n_permanent
            if results["dipoles"].shape[0] > n_perm and "dipoles" in self._previous:
                transition = results["dipoles"][n_perm:]
                signs = coupling_signs(
                    self._previous["dipoles"][n_perm:], transition, 0
                )
                results["dipoles"] = np.concatenate(
                    [results["dipoles"][:n_perm], transition * signs[:, None]]
                )
        for key in ("smooth_nacs", "nacs", "socs", "soc_vectors", "dipoles"):
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
    def _pair_components(self, k: int, l: int):
        """SHARC index pairs (k, M), (l, M) coupled by a NAC or transition dipole."""
        if self.multiplicity[k] != self.multiplicity[l]:
            return []
        return [
            (tk, tl)
            for tk, mk in self.slots[k]
            for tl, ml in self.slots[l]
            if abs(mk - ml) < 1e-9
        ]

    def _soc_matrix(self, results: Dict[str, np.ndarray]) -> np.ndarray:
        nm = self.nmstates
        if self.soc_source == "spin_orbit":
            model_pairs = {pair: q for q, pair in enumerate(self.layout.soc_pairs)}
            vectors = np.zeros((len(spin_orbit_pairs(self.states)), 3))
            for p, (i, j) in enumerate(spin_orbit_pairs(self.states)):
                k, l = self.sharc_to_model[i], self.sharc_to_model[j]
                vectors[p] = results["soc_vectors"][model_pairs[(k, l)]]
            return soc_matrix_from_vectors(vectors * self.soc_to_au, self.states)
        socs = results["socs"] * self.soc_to_au
        iu = np.triu_indices(nm, k=1)
        if socs.size == len(iu[0]):
            values = socs.astype(complex)
        else:
            values = socs[0::2] + 1j * socs[1::2]
        soc_matrix = np.zeros((nm, nm), dtype=complex)
        soc_matrix[iu] = values
        return soc_matrix + soc_matrix.conj().T

    def get_qm(self, results: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        nm, n_atoms = self.nmstates, self.n_atoms
        energies = results["energies"] * self.energy_to_au
        grads = -results["forces_states"] * self.force_to_au  # [N, S, 3]

        h = np.zeros((nm, nm), dtype=complex)
        grad = np.zeros((nm, n_atoms, 3))
        for k, components in enumerate(self.slots):
            for t, _ in components:
                h[t, t] = energies[k]
                grad[t] = grads[:, k, :]
        qm_out: Dict[str, np.ndarray] = {"h": h, "grad": grad}

        if self.nac_key is not None:
            nacs = results[self.nac_key] * self.nac_to_au  # [N, P_model, 3]
            nacdr = np.zeros((nm, nm, n_atoms, 3))
            for p, (k, l) in enumerate(self.layout.nac_pairs):
                for tk, tl in self._pair_components(k, l):
                    nacdr[tk, tl] = nacs[:, p, :]
                    nacdr[tl, tk] = -nacs[:, p, :]
            qm_out["nacdr"] = nacdr

        if "dipoles" in self.properties:
            dipoles = results["dipoles"] * self.dipole_to_au  # [D, 3]
            dm = np.zeros((3, nm, nm))
            n_permanent = self.layout.n_permanent
            if n_permanent == self.n_model_states:
                for k, components in enumerate(self.slots):
                    for t, _ in components:
                        dm[:, t, t] = dipoles[k]
            for p, (k, l) in enumerate(self.layout.dipole_pairs):
                for tk, tl in self._pair_components(k, l):
                    dm[:, tk, tl] = dipoles[n_permanent + p]
                    dm[:, tl, tk] = dipoles[n_permanent + p]
            qm_out["dm"] = dm

        if self.soc_source is not None:
            qm_out["h"] = h + self._soc_matrix(results)
        return qm_out
