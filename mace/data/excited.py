###########################################################################################
# Excited-state labels (X-MACE): multi-state energies and forces, non-adiabatic
# couplings (NACs), spin-orbit couplings (SOCs) and permanent/transition dipoles.
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import Configuration

EXCITED_MODELS = ("ExcitedMACE", "AutoencoderExcitedMACE")


def num_state_pairs(n_states: int) -> int:
    """Number of state pairs P = S(S-1)/2."""
    return n_states * (n_states - 1) // 2


def state_pair_indices(n_states: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pair order used for NACs and transition dipoles: (0,1), (0,2), ..., (1,2), ..."""
    return np.triu_indices(n_states, k=1)


def energy_gaps(energies: np.ndarray) -> np.ndarray:
    """|E_j - E_i| for every state pair (i < j), in the order of state_pair_indices."""
    energies = np.asarray(energies, dtype=float).reshape(-1)
    i, j = state_pair_indices(energies.size)
    return np.abs(energies[j] - energies[i])


@dataclass
class ExcitedStateSpec:
    """What an excited-state model predicts.

    n_states   S, number of electronic states (energies and forces for each)
    n_nacs     number of NAC vectors per atom (0: no NACs); S(S-1)/2 for all pairs
    n_socs     number of SOC values per configuration (0: no SOCs)
    n_dipoles  number of dipole vectors per configuration (0: no dipoles);
               S + S(S-1)/2 for S permanent dipoles followed by the transition dipoles
    nacs_format  "raw": NACs in the data are converted to smooth NACs (NAC * dE)
                 "smooth": NACs in the data are already smooth
    """

    n_states: int
    n_nacs: int = 0
    n_socs: int = 0
    n_dipoles: int = 0
    nacs_format: str = "raw"

    def __post_init__(self):
        if self.n_states < 1:
            raise ValueError(f"n_states must be at least 1, got {self.n_states}")
        if self.nacs_format not in ("raw", "smooth"):
            raise ValueError(
                f"nacs_format must be 'raw' or 'smooth', got {self.nacs_format!r}"
            )
        if (
            self.n_nacs > 0
            and self.nacs_format == "raw"
            and self.n_nacs != num_state_pairs(self.n_states)
        ):
            raise ValueError(
                f"Converting raw NACs to smooth NACs needs one NAC per state pair "
                f"({num_state_pairs(self.n_states)} for {self.n_states} states), but "
                f"nac_num={self.n_nacs}. Store smooth NACs and use --nacs_format=smooth "
                "for a custom NAC layout."
            )


def resolve_nacs_format(nacs_format: str, nacs_key: str) -> str:
    """'auto' picks 'smooth' when the key name says so (e.g. 'smooth_nacs'), else 'raw'."""
    if nacs_format != "auto":
        return nacs_format
    return "smooth" if "smooth" in nacs_key.lower() else "raw"


def _as_energies(value, n_states: int, index: int) -> np.ndarray:
    energies = np.asarray(value, dtype=float).reshape(-1)
    if energies.size != n_states:
        raise ValueError(
            f"Configuration {index}: expected {n_states} energies (n_energies), "
            f"found {energies.size}"
        )
    return energies


def _as_forces(value, n_atoms: int, n_states: int, index: int) -> np.ndarray:
    forces = np.asarray(value, dtype=float)
    if forces.shape == (n_atoms, n_states, 3):
        return forces
    if forces.shape == (n_atoms, 3) and n_states == 1:
        return forces.reshape(n_atoms, 1, 3)
    if forces.shape == (n_states, n_atoms, 3):
        return np.transpose(forces, (1, 0, 2))
    if forces.shape == (n_atoms, 3 * n_states):
        return forces.reshape(n_atoms, n_states, 3)
    raise ValueError(
        f"Configuration {index}: forces have shape {forces.shape}; expected "
        f"({n_atoms}, {n_states}, 3), ({n_states}, {n_atoms}, 3) or ({n_atoms}, {3 * n_states})"
    )


def _as_nacs(value, n_atoms: int, n_nacs: int, index: int) -> np.ndarray:
    nacs = np.asarray(value, dtype=float)
    if nacs.shape == (n_atoms, n_nacs, 3):
        return nacs
    if nacs.shape == (n_nacs, n_atoms, 3):
        return np.transpose(nacs, (1, 0, 2))
    if nacs.shape == (n_atoms, 3 * n_nacs):
        return nacs.reshape(n_atoms, n_nacs, 3)
    if nacs.shape == (n_atoms, 3) and n_nacs == 1:
        return nacs.reshape(n_atoms, 1, 3)
    raise ValueError(
        f"Configuration {index}: NACs have shape {nacs.shape}; expected "
        f"({n_atoms}, {n_nacs}, 3), ({n_nacs}, {n_atoms}, 3) or ({n_atoms}, {3 * n_nacs})"
    )


def _as_socs(value, n_socs: int, index: int) -> np.ndarray:
    socs = np.asarray(value)
    if np.iscomplexobj(socs):
        raise ValueError(
            f"Configuration {index}: complex SOCs are not supported; store real and "
            "imaginary parts as separate real numbers and set --soc_num accordingly"
        )
    socs = socs.astype(float).reshape(-1)
    if socs.size != n_socs:
        raise ValueError(
            f"Configuration {index}: expected {n_socs} SOC values (soc_num), found {socs.size}"
        )
    return socs


def _as_dipoles(value, n_dipoles: int, index: int) -> np.ndarray:
    dipoles = np.asarray(value, dtype=float)
    if dipoles.size != 3 * n_dipoles:
        raise ValueError(
            f"Configuration {index}: expected {n_dipoles} dipole vectors (dipole_num), "
            f"found an array of shape {dipoles.shape}"
        )
    return dipoles.reshape(n_dipoles, 3)


def standardise_excited_configs(
    configs: List[Configuration], spec: ExcitedStateSpec
) -> Dict[str, int]:
    """Bring the labels of every configuration into the shapes ExcitedMACE expects.

    energy  [S]          forces  [n_atoms, S, 3]
    nacs    [n_atoms, n_nacs, 3] (smooth)   socs [n_socs]   dipoles [n_dipoles, 3]

    Missing labels are filled with zeros and get weight 0, so that frame is
    skipped for that term only. Raw NACs are multiplied by the energy gap of
    their state pair (smooth NACs); frames with NACs but without energies cannot
    be converted and get NAC weight 0. Returns the number of labelled frames per
    property. Works in place.
    """
    n_states = spec.n_states
    counts = {"energy": 0, "forces": 0, "nacs": 0, "socs": 0, "dipoles": 0}
    for index, config in enumerate(configs):
        props, weights = config.properties, config.property_weights
        n_atoms = len(config.atomic_numbers)

        if props.get("energy") is None:
            props["energy"] = np.zeros(n_states)
            weights["energy"] = 0.0
        else:
            props["energy"] = _as_energies(props["energy"], n_states, index)
            counts["energy"] += 1

        if props.get("forces") is None:
            props["forces"] = np.zeros((n_atoms, n_states, 3))
            weights["forces"] = 0.0
        else:
            props["forces"] = _as_forces(props["forces"], n_atoms, n_states, index)
            counts["forces"] += 1

        for name, enabled in (
            ("nacs", spec.n_nacs),
            ("socs", spec.n_socs),
            ("dipoles", spec.n_dipoles),
        ):
            if not enabled:
                props.pop(name, None)
                weights.pop(name, None)

        if spec.n_nacs:
            nacs = props.get("nacs")
            if nacs is not None:
                nacs = _as_nacs(nacs, n_atoms, spec.n_nacs, index)
                if spec.nacs_format == "raw":
                    if weights.get("energy", 1.0) == 0.0:
                        logging.warning(
                            f"Configuration {index} has NACs but no energies, so the NACs "
                            "cannot be converted to smooth NACs; its NAC weight is set to 0"
                        )
                        nacs = None
                    else:
                        nacs = nacs * energy_gaps(props["energy"])[None, :, None]
            if nacs is None:
                props["nacs"] = np.zeros((n_atoms, spec.n_nacs, 3))
                weights["nacs"] = 0.0
            else:
                props["nacs"] = nacs
                weights.setdefault("nacs", 1.0)
                counts["nacs"] += 1

        if spec.n_socs:
            if props.get("socs") is None:
                props["socs"] = np.zeros(spec.n_socs)
                weights["socs"] = 0.0
            else:
                props["socs"] = _as_socs(props["socs"], spec.n_socs, index)
                weights.setdefault("socs", 1.0)
                counts["socs"] += 1

        if spec.n_dipoles:
            if props.get("dipoles") is None:
                props["dipoles"] = np.zeros((spec.n_dipoles, 3))
                weights["dipoles"] = 0.0
            else:
                props["dipoles"] = _as_dipoles(props["dipoles"], spec.n_dipoles, index)
                weights.setdefault("dipoles", 1.0)
                counts["dipoles"] += 1
    return counts


def excited_spec_from_args(args) -> Optional[ExcitedStateSpec]:
    """Build the ExcitedStateSpec from command line arguments.

    None for other models. Without a --model argument (mace_preprocess_data),
    --n_energies switches the excited-state labels on."""
    if hasattr(args, "model"):
        if args.model not in EXCITED_MODELS:
            return None
    elif getattr(args, "n_energies", None) is None:
        return None
    n_states = args.n_energies if args.n_energies is not None else 1
    n_pairs = num_state_pairs(n_states)
    n_nacs = 0
    if args.compute_nacs:
        n_nacs = args.nac_num if args.nac_num is not None else n_pairs
        if n_nacs < 1:
            raise ValueError(
                "--compute_nacs needs at least one NAC: use n_energies >= 2 or set --nac_num"
            )
    n_socs = 0
    if args.compute_socs:
        n_socs = args.soc_num if args.soc_num is not None else n_pairs
        if n_socs < 1:
            raise ValueError(
                "--compute_socs needs at least one SOC: use n_energies >= 2 or set --soc_num"
            )
    n_dipoles = 0
    if args.compute_dipoles:
        n_dipoles = (
            args.dipole_num if args.dipole_num is not None else n_states + n_pairs
        )
    return ExcitedStateSpec(
        n_states=n_states,
        n_nacs=n_nacs,
        n_socs=n_socs,
        n_dipoles=n_dipoles,
        nacs_format=resolve_nacs_format(args.nacs_format, args.nacs_key),
    )


def add_excited_state_keys(key_specification, args) -> None:
    """Read NACs (per atom), SOCs and dipoles with the keys given on the command line."""
    key_specification.update(
        info_keys={"socs": args.socs_key, "dipoles": args.dipoles_key},
        arrays_keys={"nacs": args.nacs_key},
    )
