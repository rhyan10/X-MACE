"""Synthetic excited-state data for the X-MACE tests.

Every label comes from a closed-form, smooth, symmetry-respecting function of
the positions, so a model can learn it and "the loss went down" means
something:

* energies   E_s = sum_pairs k_s (r_ij - r0)^2 + c_s, with state-dependent k_s, c_s
* forces     -dE_s/dr (analytic)
* NACs       raw NACs per pair p: a_p (r_i - centroid) / dE_p, so the smooth NACs
             (NAC * dE) are a_p (r_i - centroid): equivariant vectors
* SOCs       invariants b_l * sum_pairs r_ij
* dipoles    d_k * sum_i q_i (r_i - centroid): equivariant vectors

The layouts match X-MACE data sets: everything is in atoms.info, forces as
[n_atoms, n_states, 3] and NACs as [n_atoms, n_pairs, 3].
"""

from __future__ import annotations

from typing import List, Optional

import ase
import numpy as np

SYMBOLS = ["C", "H", "H", "O"]
CHARGES = {"C": 0.3, "H": 0.1, "O": -0.5}


def _base_positions() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [-0.4, 1.0, 0.0],
            [-0.3, -0.6, 1.0],
        ]
    )


def excited_labels(positions: np.ndarray, n_states: int, symbols=None):
    symbols = symbols or SYMBOLS
    n_atoms = len(positions)
    k = 0.5 + 0.25 * np.arange(n_states)
    c = 1.0 * np.arange(n_states)  # state offsets in eV
    r0 = 1.3
    energies = np.array(c, dtype=float)
    forces = np.zeros((n_atoms, n_states, 3))
    distance_sum = 0.0
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            rij = positions[i] - positions[j]
            d = np.linalg.norm(rij)
            distance_sum += d
            for s in range(n_states):
                energies[s] += k[s] * (d - r0) ** 2
                grad = 2 * k[s] * (d - r0) * rij / d  # dE/dr_i
                forces[i, s] -= grad
                forces[j, s] += grad
    centred = positions - positions.mean(axis=0)
    iu, ju = np.triu_indices(n_states, k=1)
    n_pairs = len(iu)
    a = 0.2 * (1 + np.arange(n_pairs))
    gaps = np.abs(energies[ju] - energies[iu])
    smooth_nacs = a[None, :, None] * centred[:, None, :]  # [n_atoms, n_pairs, 3]
    raw_nacs = smooth_nacs / gaps[None, :, None]
    socs = 0.01 * (1 + np.arange(max(n_pairs, 1))) * distance_sum
    q = np.array([CHARGES[s] for s in symbols])
    dipole = (q[:, None] * centred).sum(axis=0)
    n_dipoles = n_states + n_pairs
    dipoles = (0.5 + 0.1 * np.arange(n_dipoles))[:, None] * dipole[None, :]
    return {
        "energies": energies,
        "forces": forces,
        "raw_nacs": raw_nacs,
        "smooth_nacs": smooth_nacs,
        "socs": socs,
        "dipoles": dipoles,
    }


def make_excited_configs(
    n_configs: int,
    n_states: int = 3,
    seed: int = 0,
    displacement: float = 0.1,
    with_nacs: bool = True,
    with_socs: bool = True,
    with_dipoles: bool = True,
    smooth_key: Optional[str] = None,
    random_phases: bool = False,
) -> List[ase.Atoms]:
    """Molecules with excited-state labels in atoms.info (X-MACE layout).

    random_phases flips the sign of every coupling at random, as quantum
    chemistry codes do, to test the phase-free losses."""
    rng = np.random.default_rng(seed)
    atoms_list = []
    for _ in range(n_configs):
        positions = _base_positions() + displacement * rng.standard_normal((4, 3))
        labels = excited_labels(positions, n_states)
        atoms = ase.Atoms(SYMBOLS, positions=positions)
        atoms.info["REF_energy"] = labels["energies"]
        atoms.info["REF_forces"] = labels["forces"]
        n_pairs = labels["raw_nacs"].shape[1]
        if with_nacs and n_pairs:
            nacs = labels["smooth_nacs"] if smooth_key else labels["raw_nacs"]
            if random_phases:
                nacs = nacs * rng.choice([-1.0, 1.0], size=(1, n_pairs, 1))
            atoms.info[smooth_key or "REF_nacs"] = nacs
        if with_socs:
            socs = labels["socs"]
            if random_phases:
                socs = socs * rng.choice([-1.0, 1.0])
            atoms.info["REF_socs"] = socs
        if with_dipoles:
            dipoles = labels["dipoles"].copy()
            if random_phases:
                dipoles[n_states:] *= rng.choice([-1.0, 1.0], size=(n_pairs, 1))
            atoms.info["REF_dipoles"] = dipoles
        atoms_list.append(atoms)
    return atoms_list
