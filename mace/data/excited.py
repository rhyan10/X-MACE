###########################################################################################
# Excited-state labels (X-MACE): multi-state energies and forces, non-adiabatic
# couplings (NACs), spin-orbit couplings (SOCs), permanent/transition dipoles and
# oscillator strengths.
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from mace.modules.spin_states import (
    default_oscillator_strength_num,
    describe_states,
    n_mstates,
    same_multiplicity_pairs,
    soc_vectors_from_matrix,
    spin_orbit_pairs,
    state_pairs,
)

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

    n_states   S, number of spatial electronic states (energies and forces for each)
    n_nacs     number of NAC vectors per atom (0: no NACs)
    n_socs     number of SOC values per configuration ("free" SOCs, 0: none)
    n_dipoles  number of dipole vectors per configuration (0: no dipoles)
    n_osc      number of oscillator strengths per configuration, from the ground
               state (state 0) to states 1..n_osc (0: none)
    nacs_format  "raw": NACs in the data are converted to smooth NACs (NAC * dE)
                 "smooth": NACs in the data are already smooth
    states     states per multiplicity as in SHARC, [n_singlets, n_doublets,
               n_triplets, ...] (None: S states of one multiplicity)
    soc_mode   "free": SOCs are n_socs invariant numbers (X-MACE layout)
               "spin_orbit": one axial coupling vector per spin-orbit-coupled pair of
               spatial states, giving the M_S-resolved SOC matrix by Wigner-Eckart

    With states, NACs and transition dipoles couple states of the same multiplicity
    only: by default n_nacs is the number of such pairs and n_dipoles = S + n_nacs.
    """

    n_states: int
    n_nacs: int = 0
    n_socs: int = 0
    n_dipoles: int = 0
    nacs_format: str = "raw"
    states: Optional[List[int]] = None
    soc_mode: str = "free"
    n_osc: int = 0

    def __post_init__(self):
        if self.n_states < 1:
            raise ValueError(f"n_states must be at least 1, got {self.n_states}")
        if not 0 <= self.n_osc <= self.n_states - 1:
            raise ValueError(
                f"{self.n_osc} oscillator strengths (from the ground state) need "
                f"n_energies >= {self.n_osc + 1}, got {self.n_states}"
            )
        if self.nacs_format not in ("raw", "smooth"):
            raise ValueError(
                f"nacs_format must be 'raw' or 'smooth', got {self.nacs_format!r}"
            )
        if self.soc_mode not in ("free", "spin_orbit"):
            raise ValueError(
                f"soc_mode must be 'free' or 'spin_orbit', got {self.soc_mode!r}"
            )
        if self.states is not None:
            self.states = list(self.states)
            if sum(self.states) != self.n_states:
                raise ValueError(
                    f"states {self.states} describe {sum(self.states)} states, "
                    f"but n_energies={self.n_states}"
                )
        if self.soc_mode == "spin_orbit":
            if self.states is None:
                raise ValueError(
                    "soc_mode='spin_orbit' needs the states per multiplicity"
                )
            if not spin_orbit_pairs(self.states):
                raise ValueError(f"No spin-orbit-coupled states in {self.states}")
        if self.n_nacs > 0 and self.nacs_format == "raw" and self.nac_pairs is None:
            raise ValueError(
                f"Converting raw NACs to smooth NACs needs one NAC per coupled state pair "
                f"({len(self.coupled_pairs)}), but nac_num={self.n_nacs}. Store smooth NACs "
                "and use --nacs_format=smooth for a custom NAC layout."
            )

    @property
    def layout(self) -> List[int]:
        return self.states if self.states is not None else [self.n_states]

    @property
    def coupled_pairs(self) -> List[Tuple[int, int]]:
        """State pairs that NACs and transition dipoles can couple."""
        if self.states is None:
            return state_pairs(self.n_states)
        return same_multiplicity_pairs(self.states)

    @property
    def nac_pairs(self) -> Optional[List[Tuple[int, int]]]:
        """The state pair of every NAC, or None for a custom layout."""
        pairs = self.coupled_pairs
        return pairs if self.n_nacs == len(pairs) else None

    @property
    def n_permanent_dipoles(self) -> int:
        """Default layout: S permanent dipoles, then one transition dipole per pair."""
        if self.n_dipoles == self.n_states + len(self.coupled_pairs):
            return self.n_states
        return self.n_dipoles

    @property
    def soc_pairs(self) -> List[Tuple[int, int]]:
        return spin_orbit_pairs(self.states) if self.soc_mode == "spin_orbit" else []

    @property
    def n_soc_vectors(self) -> int:
        return len(self.soc_pairs)

    def describe(self) -> str:
        if (
            self.n_states == 1
            and self.n_nacs == 0
            and self.n_socs == 0
            and self.n_soc_vectors == 0
            and self.n_dipoles == 0
            and self.n_osc == 0
        ):
            return "ground state (energy and forces)"
        text = f"{self.n_states} states"
        if self.states is not None:
            text += f" ({describe_states(self.states)})"
        socs = (
            f"SOCs={self.n_soc_vectors} coupling vectors (spin_orbit)"
            if self.soc_mode == "spin_orbit"
            else f"SOCs={self.n_socs}"
        )
        text = (
            f"{text}, NACs={self.n_nacs} ({self.nacs_format}), {socs}, "
            f"dipoles={self.n_dipoles}"
        )
        if self.n_osc:
            text += f", oscillator strengths={self.n_osc} (S0 -> S1..S{self.n_osc})"
        return text


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


def _select_pairs(spec: ExcitedStateSpec) -> Optional[np.ndarray]:
    """Indices of the coupled pairs among all state pairs (data stored for all pairs)."""
    if spec.states is None:
        return None
    every = state_pairs(spec.n_states)
    return np.array([every.index(pair) for pair in spec.coupled_pairs], dtype=int)


def _as_nacs(value, n_atoms: int, spec: ExcitedStateSpec, index: int) -> np.ndarray:
    n_nacs = spec.n_nacs
    nacs = np.asarray(value, dtype=float)
    selected = _select_pairs(spec)
    n_all = len(state_pairs(spec.n_states))
    if selected is not None and spec.nac_pairs is not None and n_all != n_nacs:
        # NACs given for every pair of states: keep the same-multiplicity pairs
        if nacs.shape == (n_atoms, n_all, 3):
            return nacs[:, selected]
        if nacs.shape == (n_all, n_atoms, 3):
            return np.transpose(nacs[selected], (1, 0, 2))
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


def soc_matrix_from_label(value, states: List[int], index: int = 0) -> np.ndarray:
    """M_S-resolved SOC matrix [nm, nm] from the layouts accepted for SOC labels:
    complex [nm, nm]; real [nm, nm] (real SOCs); [nm, nm, 2] or [2, nm, nm] (re, im);
    the upper triangle (i < j, row by row) as nm(nm-1)/2 real or complex values, or as
    nm(nm-1) values (re, im) (X-MACE layout for SHARC)."""
    nm = n_mstates(states)
    array = np.asarray(value)
    n_upper = nm * (nm - 1) // 2
    if array.shape == (nm, nm):
        matrix = array.astype(complex)
    elif array.shape == (nm, nm, 2):
        matrix = array[..., 0] + 1j * array[..., 1]
    elif array.shape == (2, nm, nm):
        matrix = array[0] + 1j * array[1]
    elif array.size in (n_upper, 2 * n_upper):
        flat = array.reshape(-1)
        if np.iscomplexobj(flat) or flat.size == n_upper:
            upper = flat.astype(complex)
        else:
            upper = flat[0::2] + 1j * flat[1::2]
        matrix = np.zeros((nm, nm), dtype=complex)
        matrix[np.triu_indices(nm, k=1)] = upper
        matrix = matrix + matrix.conj().T
    else:
        raise ValueError(
            f"Configuration {index}: SOCs of shape {array.shape} do not describe an "
            f"M_S-resolved {nm}x{nm} SOC matrix for states {states}"
        )
    return matrix


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


def _as_dipoles(value, spec: ExcitedStateSpec, index: int) -> np.ndarray:
    n_dipoles = spec.n_dipoles
    dipoles = np.asarray(value, dtype=float)
    selected = _select_pairs(spec)
    n_all = spec.n_states + len(state_pairs(spec.n_states))
    if (
        selected is not None
        and dipoles.size == 3 * n_all
        and n_all != n_dipoles
        and spec.n_permanent_dipoles == spec.n_states
    ):
        # dipoles given for every pair of states: keep the same-multiplicity pairs
        dipoles = dipoles.reshape(n_all, 3)
        return np.concatenate(
            [dipoles[: spec.n_states], dipoles[spec.n_states :][selected]]
        )
    if dipoles.size != 3 * n_dipoles:
        raise ValueError(
            f"Configuration {index}: expected {n_dipoles} dipole vectors (dipole_num), "
            f"found an array of shape {dipoles.shape}"
        )
    return dipoles.reshape(n_dipoles, 3)


def _as_oscillator_strengths(value, n_osc: int, index: int) -> np.ndarray:
    strengths = np.asarray(value, dtype=float).reshape(-1)
    if strengths.size == n_osc + 1 and abs(strengths[0]) < 1e-12:
        strengths = strengths[1:]  # given with the ground state's own (zero) entry
    if strengths.size != n_osc:
        raise ValueError(
            f"Configuration {index}: expected {n_osc} oscillator strengths (from the "
            f"ground state to states 1..{n_osc}), found {strengths.size}"
        )
    return strengths


def standardise_excited_configs(
    configs: List[Configuration], spec: ExcitedStateSpec
) -> Dict[str, int]:  # pylint: disable=too-many-branches
    """Bring the labels of every configuration into the shapes ExcitedMACE expects.

    energy  [S]          forces  [n_atoms, S, 3]
    nacs    [n_atoms, n_nacs, 3] (smooth)   socs [n_socs]   dipoles [n_dipoles, 3]
    oscillator_strengths [n_osc]
    soc_vectors [n_pairs, 3]  (soc_mode="spin_orbit", fitted to the SOC matrix)

    Missing labels are filled with zeros and get weight 0, so that frame is
    skipped for that term only. Raw NACs are multiplied by the energy gap of
    their state pair (smooth NACs); frames with NACs but without energies cannot
    be converted and get NAC weight 0. Returns the number of labelled frames per
    property. Works in place.
    """
    n_states = spec.n_states
    counts = {
        "energy": 0,
        "forces": 0,
        "nacs": 0,
        "socs": 0,
        "dipoles": 0,
        "oscillator_strengths": 0,
    }
    worst_residual = 0.0
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
            ("socs", spec.n_socs or spec.n_soc_vectors),
            ("dipoles", spec.n_dipoles),
            ("oscillator_strengths", spec.n_osc),
        ):
            if not enabled:
                props.pop(name, None)
                weights.pop(name, None)

        if spec.n_nacs:
            nacs = props.get("nacs")
            if nacs is not None:
                nacs = _as_nacs(nacs, n_atoms, spec, index)
                if spec.nacs_format == "raw":
                    if weights.get("energy", 1.0) == 0.0:
                        logging.warning(
                            f"Configuration {index} has NACs but no energies, so the NACs "
                            "cannot be converted to smooth NACs; its NAC weight is set to 0"
                        )
                        nacs = None
                    else:
                        energies = props["energy"]
                        gaps = np.array(
                            [abs(energies[j] - energies[i]) for i, j in spec.nac_pairs]
                        )
                        nacs = nacs * gaps[None, :, None]
            if nacs is None:
                props["nacs"] = np.zeros((n_atoms, spec.n_nacs, 3))
                weights["nacs"] = 0.0
            else:
                props["nacs"] = nacs
                weights.setdefault("nacs", 1.0)
                counts["nacs"] += 1

        if spec.n_soc_vectors:
            socs = props.pop("socs", None)
            soc_weight = weights.pop("socs", 1.0)
            if socs is None:
                props["soc_vectors"] = np.zeros((spec.n_soc_vectors, 3))
                weights["soc_vectors"] = 0.0
            else:
                matrix = soc_matrix_from_label(socs, spec.states, index)
                vectors, residual = soc_vectors_from_matrix(matrix, spec.states)
                worst_residual = max(worst_residual, residual)
                props["soc_vectors"] = vectors
                weights["soc_vectors"] = soc_weight
                counts["socs"] += 1
        elif spec.n_socs:
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
                props["dipoles"] = _as_dipoles(props["dipoles"], spec, index)
                weights.setdefault("dipoles", 1.0)
                counts["dipoles"] += 1

        if spec.n_osc:
            if props.get("oscillator_strengths") is None:
                props["oscillator_strengths"] = np.zeros(spec.n_osc)
                weights["oscillator_strengths"] = 0.0
            else:
                props["oscillator_strengths"] = _as_oscillator_strengths(
                    props["oscillator_strengths"], spec.n_osc, index
                )
                weights.setdefault("oscillator_strengths", 1.0)
                counts["oscillator_strengths"] += 1
    if worst_residual > 1e-3:
        logging.warning(
            f"The SOC matrices deviate from the Wigner-Eckart form by up to "
            f"{100 * worst_residual:.1f}% (relative): check that they are M_S-resolved "
            "in SHARC order (multiplicity, M_S = -S..S, state) with standard phases"
        )
    return counts


def excited_spec_from_args(args) -> Optional[ExcitedStateSpec]:
    """Build the ExcitedStateSpec from command line arguments.

    None for other models. Without a --model argument (mace_prepare_data),
    --n_energies or --states switches the excited-state labels on."""
    states = getattr(args, "states", None)
    if hasattr(args, "model"):
        if args.model not in EXCITED_MODELS:
            return None
    elif getattr(args, "n_energies", None) is None and states is None:
        return None
    if states is not None:
        from mace.modules.spin_states import (  # pylint: disable=import-outside-toplevel
            parse_states,
        )

        states = parse_states(states)
        if args.n_energies is not None and args.n_energies != sum(states):
            raise ValueError(
                f"--states {states} describes {sum(states)} states, but --n_energies="
                f"{args.n_energies}"
            )
        n_states = sum(states)
    else:
        n_states = args.n_energies if args.n_energies is not None else 1
    coupled = same_multiplicity_pairs(states) if states else state_pairs(n_states)
    n_pairs = len(coupled)
    n_nacs = 0
    if args.compute_nacs:
        n_nacs = args.nac_num if args.nac_num is not None else n_pairs
        if n_nacs < 1:
            raise ValueError(
                "--compute_nacs needs at least one NAC: use two states of one "
                "multiplicity or set --nac_num"
            )
    soc_mode = getattr(args, "soc_mode", "auto")
    if soc_mode == "auto":
        soc_mode = "spin_orbit" if states and spin_orbit_pairs(states) else "free"
    n_socs = 0
    if args.compute_socs and soc_mode == "free":
        n_socs = args.soc_num if args.soc_num is not None else num_state_pairs(n_states)
        if n_socs < 1:
            raise ValueError(
                "--compute_socs needs at least one SOC: use n_energies >= 2 or set --soc_num"
            )
    if not args.compute_socs:
        soc_mode = "free"
    n_dipoles = 0
    if args.compute_dipoles:
        n_dipoles = (
            args.dipole_num if args.dipole_num is not None else n_states + n_pairs
        )
    n_osc = 0
    if getattr(args, "compute_oscillator_strengths", False):
        osc_num = getattr(args, "osc_num", None)
        n_osc = (
            osc_num
            if osc_num is not None
            else default_oscillator_strength_num(states or [n_states])
        )
        if n_osc < 1:
            raise ValueError(
                "--compute_oscillator_strengths needs at least two states of the ground "
                "state's multiplicity (or --osc_num)"
            )
    return ExcitedStateSpec(
        n_states=n_states,
        n_nacs=n_nacs,
        n_socs=n_socs,
        n_dipoles=n_dipoles,
        nacs_format=resolve_nacs_format(args.nacs_format, args.nacs_key),
        states=states,
        soc_mode=soc_mode,
        n_osc=n_osc,
    )


def add_excited_state_keys(key_specification, args) -> None:
    """Read NACs (per atom), SOCs, dipoles and oscillator strengths with the keys given
    on the command line."""
    key_specification.update(
        info_keys={
            "socs": args.socs_key,
            "dipoles": args.dipoles_key,
            "oscillator_strengths": getattr(
                args, "oscillator_strengths_key", "REF_oscillator_strengths"
            ),
        },
        arrays_keys={"nacs": args.nacs_key},
    )


# --------------------------------------------------------------------------------------
# Several heads: excited-state and ground-state data sets in one model
# --------------------------------------------------------------------------------------

EXCITED_HEAD_KEYS = (
    "n_energies",
    "n_states",
    "states",
    "compute_nacs",
    "compute_socs",
    "compute_dipoles",
    "compute_oscillator_strengths",
    "nac_num",
    "soc_num",
    "dipole_num",
    "osc_num",
    "soc_mode",
    "nacs_format",
    "ground_state",
)


def ground_state_spec() -> ExcitedStateSpec:
    """A ground-state head: one state, no couplings."""
    return ExcitedStateSpec(n_states=1)


def spec_layout(spec: ExcitedStateSpec) -> Dict:
    """The model's description of a head (mace.modules.excited.excited_head_layout)."""
    return {
        "n_states": spec.n_states,
        "states": list(spec.layout),
        "nac_num": spec.n_nacs,
        "soc_num": spec.n_socs,
        "soc_vector_num": spec.n_soc_vectors,
        "dipole_num": spec.n_dipoles,
        "n_permanent_dipoles": spec.n_permanent_dipoles if spec.n_dipoles else 0,
        "osc_num": spec.n_osc,
    }


def excited_specs_for_heads(
    args, heads: Dict[str, Dict]
) -> Dict[str, ExcitedStateSpec]:
    """The ExcitedStateSpec of every head.

    pt_head (replay data of a foundation model) and heads with ``ground_state: True``
    or ``n_energies: 1`` are ground-state heads. A head that sets any of the excited
    options (n_energies, states, compute_nacs, ...) in its dictionary uses them, with
    the command line as default; other heads use the command-line options."""
    import argparse  # pylint: disable=import-outside-toplevel

    specs: Dict[str, ExcitedStateSpec] = {}
    for name, head in heads.items():
        head = head or {}
        own = {k: head[k] for k in EXCITED_HEAD_KEYS if k in head}
        if "n_states" in own:
            own["n_energies"] = own.pop("n_states")
        ground = (
            name == "pt_head"
            or bool(own.get("ground_state", False))
            or (own.get("n_energies") == 1 and own.get("states") is None)
        )
        if ground:
            specs[name] = ground_state_spec()
            continue
        values = dict(vars(args))
        for key in ("nacs_key", "socs_key", "dipoles_key", "oscillator_strengths_key"):
            if key in head:
                values[key] = head[key]
        values.update({k: v for k, v in own.items() if k != "ground_state"})
        if "states" in own and "n_energies" not in own:
            values["n_energies"] = None
        spec = excited_spec_from_args(argparse.Namespace(**values))
        specs[name] = spec if spec is not None else ground_state_spec()
    return specs


@dataclass
class PaddedLayout:
    """Largest number of states and couplings over all heads: the shapes every label
    is padded to, so that frames of different heads can share a batch."""

    n_states: int
    n_nacs: int
    n_socs: int
    n_soc_vectors: int
    n_dipoles: int
    n_osc: int = 0

    @classmethod
    def from_specs(cls, specs) -> "PaddedLayout":
        specs = list(specs)
        return cls(
            n_states=max(s.n_states for s in specs),
            n_nacs=max(s.n_nacs for s in specs),
            n_socs=max(s.n_socs for s in specs),
            n_soc_vectors=max(s.n_soc_vectors for s in specs),
            n_dipoles=max(s.n_dipoles for s in specs),
            n_osc=max(s.n_osc for s in specs),
        )


def attach_excited_padding(
    dataset, spec: ExcitedStateSpec, padded: PaddedLayout, weight: float = 1.0
) -> int:
    """Make preprocessed (HDF5) datasets pad their frames' labels to the shared layout
    of all heads as they are read (and scale their configuration weights by weight).
    Returns the number of datasets that will do so; other dataset types are left
    unchanged (their labels must already have the layout)."""
    # pylint: disable=import-outside-toplevel
    from torch.utils.data import ConcatDataset

    from mace.data.hdf5_dataset import HDF5Dataset

    if isinstance(dataset, ConcatDataset):
        return sum(
            attach_excited_padding(d, spec, padded, weight) for d in dataset.datasets
        )
    if (
        isinstance(dataset, (list, tuple))
        and dataset
        and not hasattr(dataset[0], "positions")
    ):
        return sum(attach_excited_padding(d, spec, padded, weight) for d in dataset)
    if isinstance(dataset, HDF5Dataset):
        dataset.excited_padding = (spec, padded)
        dataset.config_weight_scale = float(weight)
        return 1
    return 0


def _pad(array: np.ndarray, axis: int, size: int) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    missing = size - array.shape[axis]
    if missing <= 0:
        return array
    widths = [(0, 0)] * array.ndim
    widths[axis] = (0, missing)
    return np.pad(array, widths)


def pad_excited_configs(
    configs: List[Configuration], spec: ExcitedStateSpec, padded: PaddedLayout
) -> None:
    """Pad the standardised labels of one head to the shared layout (in place).

    Padded entries are zero; the model and the loss ignore them through per-head
    masks. Labels the head does not have but other heads do are added as zeros with
    weight 0."""
    for config in configs:
        props, weights = config.properties, config.property_weights
        n_atoms = len(config.atomic_numbers)
        props["energy"] = _pad(np.reshape(props["energy"], -1), 0, padded.n_states)
        forces = np.asarray(props["forces"], dtype=float).reshape(n_atoms, -1, 3)
        props["forces"] = _pad(forces, 1, padded.n_states)
        for name, size, shape in (
            ("nacs", padded.n_nacs, (n_atoms, 0, 3)),
            ("socs", padded.n_socs, (0,)),
            ("soc_vectors", padded.n_soc_vectors, (0, 3)),
            ("dipoles", padded.n_dipoles, (0, 3)),
            ("oscillator_strengths", padded.n_osc, (0,)),
        ):
            if size == 0:
                props.pop(name, None)
                weights.pop(name, None)
                continue
            if props.get(name) is None:
                props[name] = np.zeros(shape)
                weights[name] = 0.0
            axis = 1 if name == "nacs" else 0
            props[name] = _pad(props[name], axis, size)
