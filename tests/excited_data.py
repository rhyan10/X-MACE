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
    with_oscillator_strengths: bool = False,
) -> List[ase.Atoms]:
    """Molecules with excited-state labels in atoms.info (X-MACE layout).

    with_oscillator_strengths adds REF_oscillator_strengths [S - 1], f_0k =
    2/3 dE_0k |mu_0k|^2 from the energies and transition dipoles (smooth, >= 0).

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
        if with_oscillator_strengths:
            atoms.info["REF_oscillator_strengths"] = oscillator_strengths(
                labels, n_states
            )
        atoms_list.append(atoms)
    return atoms_list


def oscillator_strengths(labels, n_states: int) -> np.ndarray:
    """f_0k = 2/3 (E_k - E_0) |mu_0k|^2 for k = 1..S-1 (pair order (0,1), (0,2), ...)."""
    energies = np.asarray(labels["energies"])
    transition = np.asarray(labels["dipoles"])[n_states:]
    return np.array(
        [
            2.0
            / 3.0
            * abs(energies[k] - energies[0])
            * float(transition[k - 1] @ transition[k - 1])
            for k in range(1, n_states)
        ]
    )


MASSES = {"C": 12.0, "H": 1.0, "O": 16.0}


def axial_vector(positions: np.ndarray, power: int = 1, symbols=None) -> np.ndarray:
    """An axial (pseudo)vector of the molecule, sum_ij q_i m_j |c_i - c_j|^power
    (c_i x c_j): rotation-equivariant, invariant under inversion and permutations of
    equal atoms."""
    symbols = symbols or SYMBOLS
    q = np.array([CHARGES[s] for s in symbols])
    m = np.array([MASSES[s] for s in symbols]) / 10.0
    centred = positions - positions.mean(axis=0)
    vector = np.zeros(3)
    for i in range(len(positions)):
        for j in range(len(positions)):
            d = np.linalg.norm(centred[i] - centred[j])
            vector += q[i] * m[j] * d**power * np.cross(centred[i], centred[j])
    return vector


def multiplicity_labels(positions: np.ndarray, states, rng=None):
    """Labels for states of several multiplicities ([n_singlets, n_doublets, ...]).

    Energies, forces, NACs and dipoles as in excited_labels (NACs/transition dipoles
    given for every pair, only same-multiplicity pairs are physical); the SOC matrix
    comes from axial coupling vectors by Wigner-Eckart. With rng every spatial state
    gets a random sign, as in quantum chemistry outputs."""
    from mace.modules.spin_states import (  # pylint: disable=import-outside-toplevel
        ms_resolved_map,
        soc_matrix_from_vectors,
        spin_orbit_pairs,
    )

    n_states = int(sum(states))
    labels = excited_labels(positions, n_states)
    pairs = spin_orbit_pairs(states)
    base, other = axial_vector(positions), axial_vector(positions, power=2)
    vectors = np.array(
        [(0.2 + 0.1 * p) * base + 0.05 * (p + 1) * other for p in range(len(pairs))]
    ).reshape(len(pairs), 3)
    signs = np.ones(n_states) if rng is None else rng.choice([-1.0, 1.0], size=n_states)
    for p, (i, j) in enumerate(pairs):
        vectors[p] *= signs[i] * signs[j]
    labels["soc_vectors"] = vectors
    labels["soc_matrix"] = soc_matrix_from_vectors(vectors, states)
    iu, ju = np.triu_indices(n_states, k=1)
    labels["raw_nacs"] = labels["raw_nacs"] * (signs[iu] * signs[ju])[None, :, None]
    labels["smooth_nacs"] = (
        labels["smooth_nacs"] * (signs[iu] * signs[ju])[None, :, None]
    )
    labels["dipoles"][n_states:] *= (signs[iu] * signs[ju])[:, None]
    labels["ms_map"] = ms_resolved_map(states)
    return labels


def make_multiplicity_configs(
    n_configs: int, states, seed: int = 0, displacement: float = 0.1, random_phases=True
) -> List[ase.Atoms]:
    """Molecules with labels for several multiplicities; SOCs as the upper triangle of
    the M_S-resolved SOC matrix, (re, im) interleaved."""
    rng = np.random.default_rng(seed)
    atoms_list = []
    for _ in range(n_configs):
        positions = _base_positions() + displacement * rng.standard_normal((4, 3))
        labels = multiplicity_labels(positions, states, rng if random_phases else None)
        atoms = ase.Atoms(SYMBOLS, positions=positions)
        atoms.info["REF_energy"] = labels["energies"]
        atoms.info["REF_forces"] = labels["forces"]
        atoms.info["REF_nacs"] = labels["raw_nacs"]
        upper = labels["soc_matrix"][
            np.triu_indices(labels["soc_matrix"].shape[0], k=1)
        ]
        atoms.info["REF_socs"] = np.stack([upper.real, upper.imag], axis=-1).reshape(-1)
        atoms.info["REF_dipoles"] = labels["dipoles"]
        atoms_list.append(atoms)
    return atoms_list
