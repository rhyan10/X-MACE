###########################################################################################
# Spin multiplicities and spin-orbit couplings for excited-state models
#
# Electronic states are given per multiplicity as in SHARC: states = [n_singlets,
# n_doublets, n_triplets, n_quartets, n_quintets, ...]. The model predicts spatial
# (spin-free) states in this order. SHARC's M_S-resolved basis repeats every state of
# multiplicity m = 2S + 1 for M_S = -S, ..., S:
#     for m: for M_S: for state of multiplicity m
#
# Symmetry constraints used throughout:
#   * energies, gradients and permanent dipoles do not depend on M_S
#   * NACs and transition dipoles couple states of the same multiplicity (and M_S) only
#   * spin-orbit couplings follow the Wigner-Eckart theorem: H_SO = sum_k A_k (x) S_k is a
#     rank-1 spin tensor, so every pair of spatial states I (spin S) and J (spin S'),
#     S - S' in {-1, 0, 1} and not both singlets, has one coupling vector A^IJ = i a^IJ with a
#     real a^IJ (real spatial wave functions, time-reversal symmetry). a^IJ is an axial
#     vector of the molecular frame, and
#         <I S M | H_SO | J S' M'> = (-1)^q A_q c(S, S') <S' M' 1 -q | S M>,  -q = M - M'
#     with spherical components A_0 = A_z, A_{+-1} = -+(A_x +- i A_y)/sqrt(2) and
#     c(S, S) = sqrt(S(S+1)), c = 1 otherwise. For S = S' this is exactly a . (i S).
#   * the diagonal blocks <I|H_SO|I> vanish (real wave functions)
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from functools import lru_cache
from math import factorial, sqrt
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

MULTIPLICITY_NAMES = {
    1: "singlet",
    2: "doublet",
    3: "triplet",
    4: "quartet",
    5: "quintet",
}


def parse_states(states) -> List[int]:
    """'2 0 2', [2, 0, 2] or '2,0,2' -> [2, 0, 2] (trailing zeros removed)."""
    if states is None:
        return []
    if isinstance(states, str):
        values = [
            int(x)
            for x in states.replace(",", " ")
            .replace("[", " ")
            .replace("]", " ")
            .split()
        ]
    else:
        values = [int(x) for x in states]
    if any(v < 0 for v in values):
        raise ValueError(f"Numbers of states must be >= 0, got {values}")
    while values and values[-1] == 0:
        values.pop()
    if not values or sum(values) == 0:
        raise ValueError("At least one electronic state is needed")
    return values


def spatial_multiplicities(states: Sequence[int]) -> List[int]:
    """Multiplicity of every spatial state, in model order."""
    return [m + 1 for m, n in enumerate(states) for _ in range(n)]


def n_mstates(states: Sequence[int]) -> int:
    return sum((m + 1) * n for m, n in enumerate(states))


def ms_resolved_map(states: Sequence[int]) -> List[List[Tuple[int, float]]]:
    """For every spatial state the list of (M_S-resolved index, M_S), in SHARC order."""
    result: List[List[Tuple[int, float]]] = []
    offset = 0
    for m, n in enumerate(states):
        mult = m + 1
        spin = (mult - 1) / 2
        for j in range(n):
            result.append([(offset + k * n + j, -spin + k) for k in range(mult)])
        offset += mult * n
    return result


def state_pairs(n_states: int) -> List[Tuple[int, int]]:
    """All pairs (i < j) in the order (0,1), (0,2), ..., (1,2), ..."""
    i, j = np.triu_indices(n_states, k=1)
    return list(zip(i.tolist(), j.tolist()))


def same_multiplicity_pairs(states: Sequence[int]) -> List[Tuple[int, int]]:
    """Pairs coupled by NACs and transition dipoles: equal multiplicity."""
    mult = spatial_multiplicities(states)
    return [(i, j) for i, j in state_pairs(len(mult)) if mult[i] == mult[j]]


def default_oscillator_strength_num(states: Sequence[int]) -> int:
    """Number of oscillator strengths from the ground state (state 0, the lowest state
    of the first multiplicity present) to the other states of its multiplicity."""
    first = next((int(n) for n in states if int(n) > 0), 0)
    return max(first - 1, 0)


def oscillator_strength_pairs(n_osc: int) -> List[Tuple[int, int]]:
    """The transitions of the oscillator strengths: (0, 1), (0, 2), ..., (0, n_osc)."""
    return [(0, k) for k in range(1, int(n_osc) + 1)]


def spin_orbit_pairs(states: Sequence[int]) -> List[Tuple[int, int]]:
    """Pairs coupled by spin-orbit coupling: S_i - S_j in {-1, 0, 1}, not both singlets.

    States whose multiplicities differ by one (e.g. singlet and doublet) have different
    numbers of electrons and do not couple."""
    mult = spatial_multiplicities(states)
    return [
        (i, j)
        for i, j in state_pairs(len(mult))
        if abs(mult[i] - mult[j]) in (0, 2) and not (mult[i] == 1 and mult[j] == 1)
    ]


# --------------------------------------------------------------------------------------
# Clebsch-Gordan coefficients (Condon-Shortley phase convention)
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def clebsch_gordan(
    j1: float, m1: float, j2: float, m2: float, j: float, m: float
) -> float:
    """<j1 m1 j2 m2 | j m> for integer or half-integer arguments (Racah formula)."""
    tj1, tm1, tj2, tm2, tj, tm = (int(round(2 * x)) for x in (j1, m1, j2, m2, j, m))
    if tm1 + tm2 != tm:
        return 0.0
    if not abs(tj1 - tj2) <= tj <= tj1 + tj2 or (tj1 + tj2 + tj) % 2:
        return 0.0
    if abs(tm1) > tj1 or abs(tm2) > tj2 or abs(tm) > tj:
        return 0.0
    if (tj1 + tm1) % 2 or (tj2 + tm2) % 2 or (tj + tm) % 2:
        return 0.0

    def f(two_x: int) -> int:
        return factorial(two_x // 2)

    prefactor = sqrt(
        (tj + 1)
        * f(tj + tj1 - tj2)
        * f(tj - tj1 + tj2)
        * f(tj1 + tj2 - tj)
        / f(tj1 + tj2 + tj + 2)
    )
    prefactor *= sqrt(
        f(tj + tm)
        * f(tj - tm)
        * f(tj1 - tm1)
        * f(tj1 + tm1)
        * f(tj2 - tm2)
        * f(tj2 + tm2)
    )
    total = 0.0
    for k in range(0, 2 * (tj1 + tj2 + tj) + 1, 2):
        terms = (
            k,
            tj1 + tj2 - tj - k,
            tj1 - tm1 - k,
            tj2 + tm2 - k,
            tj - tj2 + tm1 + k,
            tj - tj1 - tm2 + k,
        )
        if min(terms) < 0:
            continue
        denominator = 1
        for t in terms:
            denominator *= f(t)
        total += (-1) ** (k // 2) / denominator
    return prefactor * total


def spin_matrices(spin: float) -> np.ndarray:
    """S_x, S_y, S_z [3, 2S+1, 2S+1] in the basis M = -S, ..., S (Condon-Shortley)."""
    dim = int(round(2 * spin)) + 1
    ms = -spin + np.arange(dim)
    s_plus = np.zeros((dim, dim))
    for k in range(dim - 1):  # <M+1|S+|M>
        s_plus[k + 1, k] = sqrt(spin * (spin + 1) - ms[k] * (ms[k] + 1))
    s_minus = s_plus.T
    sx = (s_plus + s_minus) / 2
    sy = (s_plus - s_minus) / 2j
    sz = np.diag(ms)
    return np.array([sx, sy, sz], dtype=complex)


@lru_cache(maxsize=None)
def _soc_coefficients_cached(states: Tuple[int, ...]) -> np.ndarray:
    states = list(states)
    pairs = spin_orbit_pairs(states)
    mult = spatial_multiplicities(states)
    ms_map = ms_resolved_map(states)
    nm = n_mstates(states)
    coeff = np.zeros((nm, nm, max(len(pairs), 1), 3), dtype=complex)
    # spherical components of A = i a:  A_q = sum_k U[q, k] a_k  (q = -1, 0, +1)
    u = np.array(
        [
            [
                1j / sqrt(2),
                1 / sqrt(2),
                0.0,
            ],  # A_{-1} = (A_x - i A_y)/sqrt2 = (i a_x + a_y)/sqrt2
            [0.0, 0.0, 1j],  # A_0 = i a_z
            [
                -1j / sqrt(2),
                1 / sqrt(2),
                0.0,
            ],  # A_{+1} = -(A_x + i A_y)/sqrt2 = (-i a_x + a_y)/sqrt2
        ]
    )
    for p, (i, j) in enumerate(pairs):
        s_i, s_j = (mult[i] - 1) / 2, (mult[j] - 1) / 2
        c = sqrt(s_i * (s_i + 1)) if mult[i] == mult[j] else 1.0
        for row, m_i in ms_map[i]:
            for col, m_j in ms_map[j]:
                minus_q = m_i - m_j
                if abs(minus_q) > 1:
                    continue
                q = -minus_q
                cg = clebsch_gordan(s_j, m_j, 1, minus_q, s_i, m_i)
                if cg == 0.0:
                    continue
                value = (-1) ** int(round(q)) * c * cg * u[int(round(q)) + 1]
                coeff[row, col, p] += value
                coeff[col, row, p] += np.conj(value)
    return coeff


def soc_coefficients(states: Sequence[int]) -> np.ndarray:
    """C [nm, nm, P, 3] (complex) with H_SO = sum_{p,k} C[:, :, p, k] a[p, k]."""
    return _soc_coefficients_cached(tuple(int(s) for s in states))


def soc_pair_weights(states: Sequence[int]) -> np.ndarray:
    """||H_SO||_F^2 contributed by a unit coupling vector of every pair [P] (isotropic)."""
    coeff = soc_coefficients(states)
    pairs = spin_orbit_pairs(states)
    if not pairs:
        return np.zeros(0)
    return np.real(np.einsum("abpk,abpk->p", coeff, coeff.conj())) / 3.0


def soc_matrix_from_vectors(vectors, states: Sequence[int]):
    """Hermitian M_S-resolved SOC matrix [..., nm, nm] from coupling vectors [..., P, 3].

    Works with numpy arrays and (differentiable) torch tensors."""
    coeff = soc_coefficients(states)
    if not spin_orbit_pairs(states):
        coeff = coeff[:, :, :0]
    if isinstance(vectors, torch.Tensor):
        c = torch.as_tensor(coeff, device=vectors.device)
        c = c.to(
            torch.complex128 if vectors.dtype == torch.float64 else torch.complex64
        )
        return torch.einsum("abpk,...pk->...ab", c, vectors.to(c.dtype))
    return np.einsum("abpk,...pk->...ab", coeff, np.asarray(vectors))


def soc_matrix_entries(states: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Rows and columns of the upper-triangle SOC matrix elements allowed by the
    selection rules (all others vanish by symmetry)."""
    coeff = soc_coefficients(states)
    allowed = np.abs(coeff).sum(axis=(2, 3)) > 1e-12
    rows, cols = np.nonzero(np.triu(allowed, k=1))
    return rows, cols


def soc_entries_from_vectors(vectors, states: Sequence[int]):
    """Real and imaginary parts of the allowed SOC matrix elements [..., 2K] given by
    coupling vectors [..., P, 3] (numpy or torch); used for SOC error metrics."""
    rows, cols = soc_matrix_entries(states)
    matrix = soc_matrix_from_vectors(vectors, states)
    entries = matrix[..., rows, cols]
    if isinstance(entries, torch.Tensor):
        return torch.cat([entries.real, entries.imag], dim=-1)
    return np.concatenate([entries.real, entries.imag], axis=-1)


def soc_vectors_from_matrix(
    matrix: np.ndarray, states: Sequence[int]
) -> Tuple[np.ndarray, float]:
    """Least-squares coupling vectors [P, 3] of an M_S-resolved SOC matrix [nm, nm].

    Returns the vectors and the relative residual ||H - H(a)|| / ||H|| (off-diagonal
    blocks between different spatial states only); a large residual means the matrix
    does not follow the Wigner-Eckart form (other phase convention or basis)."""
    states = list(states)
    pairs = spin_orbit_pairs(states)
    coeff = soc_coefficients(states)[:, :, : len(pairs)]
    matrix = np.asarray(matrix, dtype=complex)
    nm = n_mstates(states)
    if matrix.shape != (nm, nm):
        raise ValueError(
            f"SOC matrix must be {nm}x{nm} for states {states}, got {matrix.shape}"
        )
    if not pairs:
        return np.zeros((0, 3)), 0.0
    mask = _coupling_mask(states)
    design = coeff[mask].reshape(-1, len(pairs) * 3)
    target = matrix[mask]
    # real unknowns: stack real and imaginary parts
    a_mat = np.concatenate([design.real, design.imag])
    b_vec = np.concatenate([target.real, target.imag])
    solution, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    residual = np.linalg.norm(a_mat @ solution - b_vec)
    norm = np.linalg.norm(b_vec)
    return solution.reshape(len(pairs), 3), float(residual / norm) if norm > 0 else 0.0


@lru_cache(maxsize=None)
def _coupling_mask_cached(states: Tuple[int, ...]) -> np.ndarray:
    ms_map = ms_resolved_map(list(states))
    nm = n_mstates(list(states))
    owner = np.zeros(nm, dtype=int)
    for k, entries in enumerate(ms_map):
        for idx, _ in entries:
            owner[idx] = k
    return owner[:, None] != owner[None, :]


def _coupling_mask(states: Sequence[int]) -> np.ndarray:
    """True for matrix elements between different spatial states."""
    return _coupling_mask_cached(tuple(int(s) for s in states))


def spin_rotation(states: Sequence[int], rotation: np.ndarray) -> np.ndarray:
    """Unitary [nm, nm] rotating the spin of every multiplet by the rotation matrix R."""
    from scipy.linalg import expm  # pylint: disable=import-outside-toplevel
    from scipy.spatial.transform import (  # pylint: disable=import-outside-toplevel
        Rotation,
    )

    rotvec = Rotation.from_matrix(rotation).as_rotvec()
    nm = n_mstates(states)
    unitary = np.zeros((nm, nm), dtype=complex)
    for k, entries in enumerate(ms_resolved_map(states)):
        mult = len(entries)
        spin_ops = spin_matrices((mult - 1) / 2)
        d_matrix = expm(-1j * np.einsum("k,kab->ab", rotvec, spin_ops))
        idx = [e[0] for e in entries]
        unitary[np.ix_(idx, idx)] = d_matrix
    return unitary


def describe_states(states: Sequence[int]) -> str:
    parts = []
    for m, n in enumerate(states):
        if n:
            name = MULTIPLICITY_NAMES.get(m + 1, f"multiplicity-{m + 1}")
            parts.append(f"{n} {name}{'s' if n > 1 else ''}")
    return ", ".join(parts)


def expand_states(states: Optional[Sequence[int]], n_energies: int) -> List[int]:
    """Multiplicity layout of a model: given, or n_energies singlets."""
    return list(states) if states else [n_energies]
