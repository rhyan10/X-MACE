###########################################################################################
# Turn SHARC quantities (QM.out / QMout objects, atomic units, M_S-resolved) into
# X-MACE training labels (one entry per spatial state, eV and Angstrom)
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Dict, Optional, Sequence

import ase
import numpy as np

from mace.modules.spin_states import (
    ms_resolved_map,
    n_mstates,
    parse_states,
    same_multiplicity_pairs,
)

BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_INV_CM = 219474.6313632

SOC_UNITS_PER_HARTREE = {
    "Hartree": 1.0,
    "eV": HARTREE_TO_EV,
    "cm-1": HARTREE_TO_INV_CM,
}


def labels_from_sharc(
    states: Sequence[int],
    h: np.ndarray,
    grad: Optional[np.ndarray] = None,
    nacdr: Optional[np.ndarray] = None,
    dm: Optional[np.ndarray] = None,
    soc_unit: str = "cm-1",
    degeneracy_tolerance: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """X-MACE labels from SHARC's M_S-resolved quantities in atomic units.

    states  SHARC states per multiplicity, e.g. [4, 0, 3]
    h       [nm, nm] Hamiltonian (MCH): energies on the diagonal, SOCs off it (Hartree)
    grad    [nm, n_atoms, 3] gradients (Hartree / bohr)
    nacdr   [nm, nm, n_atoms, 3] non-adiabatic couplings (1 / bohr)
    dm      [3, nm, nm] dipole matrices (e bohr)

    Returns, per spatial state (model order: singlets, doublets, triplets, ...):
      energies [S] (eV), forces [n_atoms, S, 3] (eV / A),
      nacs     [n_atoms, P, 3] raw NACs (1 / A) of the same-multiplicity pairs,
      socs     [nm, nm] complex M_S-resolved SOC matrix (soc_unit),
      dipoles  [S + P, 3] permanent then transition dipoles (e A).
    The M_S components of every spatial state must be degenerate (checked), so the
    first component carries the label.
    """
    states = parse_states(states)
    nm = n_mstates(states)
    h = np.asarray(h)
    if h.shape != (nm, nm):
        raise ValueError(f"h has shape {h.shape}, expected ({nm}, {nm}) for {states}")
    if soc_unit not in SOC_UNITS_PER_HARTREE:
        raise ValueError(f"soc_unit must be one of {list(SOC_UNITS_PER_HARTREE)}")
    ms_map = ms_resolved_map(states)
    first = [components[0][0] for components in ms_map]

    diagonal = np.real(np.diag(h))
    for k, components in enumerate(ms_map):
        spread = np.ptp([diagonal[index] for index, _ in components])
        if spread > degeneracy_tolerance:
            raise ValueError(
                f"The M_S components of spatial state {k} differ in energy by {spread:.2e} "
                "Hartree: h is not in the MCH (spin-free) representation"
            )
    labels: Dict[str, np.ndarray] = {"energies": diagonal[first] * HARTREE_TO_EV}

    soc = h - np.diag(np.diag(h))
    labels["socs"] = soc.astype(complex) * SOC_UNITS_PER_HARTREE[soc_unit]

    if grad is not None:
        grad = np.asarray(grad, dtype=float)
        forces = -grad[first] * HARTREE_TO_EV / BOHR_TO_ANGSTROM  # [S, N, 3]
        labels["forces"] = np.transpose(forces, (1, 0, 2))

    pairs = same_multiplicity_pairs(states)
    if nacdr is not None:
        nacdr = np.asarray(nacdr, dtype=float)
        nacs = np.stack([nacdr[first[i], first[j]] for i, j in pairs], axis=1)
        labels["nacs"] = (
            nacs / BOHR_TO_ANGSTROM if pairs else np.zeros((nacdr.shape[2], 0, 3))
        )
    if dm is not None:
        dm = np.real(np.asarray(dm))
        permanent = [dm[:, a, a] for a in first]
        transition = [dm[:, first[i], first[j]] for i, j in pairs]
        labels["dipoles"] = np.array(permanent + transition) * BOHR_TO_ANGSTROM
    return labels


def soc_to_upper_triangle(socs: np.ndarray) -> np.ndarray:
    """Complex SOC matrix -> real (re, im)-interleaved upper triangle, the layout that
    extended XYZ files can store and --socs_key reads."""
    upper = socs[np.triu_indices(socs.shape[0], k=1)]
    return np.stack([upper.real, upper.imag], axis=-1).reshape(-1)


def atoms_from_sharc(
    symbols: Sequence[str],
    coords_bohr: np.ndarray,
    states: Sequence[int],
    h: np.ndarray,
    grad: Optional[np.ndarray] = None,
    nacdr: Optional[np.ndarray] = None,
    dm: Optional[np.ndarray] = None,
    soc_unit: str = "cm-1",
    prefix: str = "REF_",
) -> ase.Atoms:
    """An ase.Atoms frame with X-MACE labels (prefix + energy, forces, nacs, socs,
    dipoles) from SHARC quantities; write a list of them with ase.io.write."""
    labels = labels_from_sharc(states, h, grad, nacdr, dm, soc_unit=soc_unit)
    atoms = ase.Atoms(
        symbols=list(symbols), positions=np.asarray(coords_bohr) * BOHR_TO_ANGSTROM
    )
    atoms.info[prefix + "energy"] = labels["energies"]
    if "forces" in labels:
        atoms.info[prefix + "forces"] = labels["forces"]
    if "nacs" in labels:
        atoms.info[prefix + "nacs"] = labels["nacs"]
    if np.any(labels["socs"] != 0):
        atoms.info[prefix + "socs"] = soc_to_upper_triangle(labels["socs"])
    if "dipoles" in labels:
        atoms.info[prefix + "dipoles"] = labels["dipoles"]
    return atoms
