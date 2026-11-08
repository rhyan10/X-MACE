"""Spin-state bookkeeping and the Wigner-Eckart form of spin-orbit couplings."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import mace  # noqa: F401  pylint: disable=unused-import
from mace.modules.spin_states import (
    clebsch_gordan,
    describe_states,
    ms_resolved_map,
    n_mstates,
    parse_states,
    same_multiplicity_pairs,
    soc_coefficients,
    soc_entries_from_vectors,
    soc_matrix_entries,
    soc_matrix_from_vectors,
    soc_pair_weights,
    soc_vectors_from_matrix,
    spatial_multiplicities,
    spin_matrices,
    spin_orbit_pairs,
    spin_rotation,
)

LAYOUTS = [[2, 0, 2], [0, 2, 0, 1], [1, 0, 1, 0, 1], [0, 3], [1, 1, 1], [2, 0, 1, 0, 2]]


def test_parse_and_describe():
    assert parse_states("3 0 2") == [3, 0, 2]
    assert parse_states("3,0,2,0") == [3, 0, 2]
    assert parse_states([0, 2, 0, 1]) == [0, 2, 0, 1]
    assert describe_states([2, 0, 1]) == "2 singlets, 1 triplet"
    assert n_mstates([2, 1, 2, 1, 1]) == 2 + 2 + 6 + 4 + 5
    assert spatial_multiplicities([1, 2, 0, 1]) == [1, 2, 2, 4]


def test_ms_resolved_order_is_sharc_order():
    # SHARC: for each multiplicity, for M_S = -S..S, for each state
    ms_map = ms_resolved_map([1, 0, 2])
    assert ms_map[0] == [(0, 0.0)]
    assert ms_map[1] == [(1, -1.0), (3, 0.0), (5, 1.0)]
    assert ms_map[2] == [(2, -1.0), (4, 0.0), (6, 1.0)]
    doublets = ms_resolved_map([0, 2])
    assert doublets == [[(0, -0.5), (2, 0.5)], [(1, -0.5), (3, 0.5)]]


def test_selection_rules():
    # singlets S0 S1, doublet D0, triplet T0, quintet Q0
    states = [2, 1, 1, 0, 1]
    assert same_multiplicity_pairs(states) == [(0, 1)]
    # no singlet-singlet, no singlet-doublet (|dS| = 1/2), no singlet-quintet (dS = 2)
    assert spin_orbit_pairs(states) == [(0, 3), (1, 3), (3, 4)]
    assert spin_orbit_pairs([3]) == []
    assert spin_orbit_pairs([0, 2, 0, 1]) == [(0, 1), (0, 2), (1, 2)]


@pytest.mark.parametrize(
    "args",
    [
        (1, 1, 1, -1, 1, 0),
        (0.5, 0.5, 1, -1, 0.5, -0.5),
        (2, -1, 1, 1, 2, 0),
        (1.5, 0.5, 1, 0, 1.5, 0.5),
        (1, 0, 1, 0, 2, 0),
        (2, 2, 1, -1, 1, 1),
    ],
)
def test_clebsch_gordan_against_sympy(args):
    sympy_cg = pytest.importorskip("sympy.physics.quantum.cg")
    from sympy import Rational, nsimplify  # pylint: disable=import-outside-toplevel

    rational = [nsimplify(Rational(x).limit_denominator(2)) for x in args]
    expected = float(sympy_cg.CG(*rational).doit())
    assert abs(clebsch_gordan(*args) - expected) < 1e-12


@pytest.mark.parametrize("spin", [0.5, 1.0, 1.5, 2.0])
def test_spin_matrices_commutators(spin):
    sx, sy, sz = spin_matrices(spin)
    np.testing.assert_allclose(sx @ sy - sy @ sx, 1j * sz, atol=1e-12)
    casimir = sx @ sx + sy @ sy + sz @ sz
    np.testing.assert_allclose(casimir, spin * (spin + 1) * np.eye(len(sz)), atol=1e-12)


@pytest.mark.parametrize("mult", [2, 3, 4, 5])
def test_same_spin_block_is_a_dot_i_s(mult):
    # two states of one multiplicity: the coupling block is (a . i S) up to the norm
    states = [0] * (mult - 1) + [2]
    a = np.array([0.3, -0.2, 0.7])
    h = soc_matrix_from_vectors(a[None, :], states)
    ms_map = ms_resolved_map(states)
    rows = [index for index, _ in ms_map[0]]
    cols = [index for index, _ in ms_map[1]]
    block = h[np.ix_(rows, cols)]
    spin = (mult - 1) / 2
    expected = 1j * np.einsum("k,kab->ab", a, spin_matrices(spin))
    np.testing.assert_allclose(block, expected, atol=1e-12)
    # diagonal blocks (same spatial state) vanish
    np.testing.assert_allclose(h[np.ix_(rows, rows)], 0.0, atol=1e-14)


@pytest.mark.parametrize("states", LAYOUTS)
def test_soc_matrix_symmetries(states):
    rng = np.random.default_rng(0)
    pairs = spin_orbit_pairs(states)
    vectors = rng.standard_normal((len(pairs), 3))
    h = soc_matrix_from_vectors(vectors, states)
    nm = n_mstates(states)
    assert h.shape == (nm, nm)
    np.testing.assert_allclose(h, h.conj().T, atol=1e-14)  # Hermitian
    mult = []
    for m, n in enumerate(states):
        mult += [m + 1] * (n * (m + 1))
    ms = np.zeros(nm)
    for comps in ms_resolved_map(states):
        for index, value in comps:
            ms[index] = value
    for a in range(nm):
        for b in range(nm):
            allowed = abs(mult[a] - mult[b]) <= 2 and abs(ms[a] - ms[b]) <= 1 + 1e-9
            allowed &= not (mult[a] == 1 and mult[b] == 1)
            if not allowed:
                assert h[a, b] == 0
    # round trip
    back, residual = soc_vectors_from_matrix(h, states)
    np.testing.assert_allclose(back, vectors, atol=1e-10)
    assert residual < 1e-12


@pytest.mark.parametrize("states", LAYOUTS)
def test_soc_matrix_rotates_with_the_spins(states):
    # H(R a) = U(R) H(a) U(R)^dagger: the coupling vectors are vectors under rotation
    rng = np.random.default_rng(1)
    vectors = rng.standard_normal((len(spin_orbit_pairs(states)), 3))
    rotation = Rotation.random(random_state=3).as_matrix()
    u = spin_rotation(states, rotation)
    np.testing.assert_allclose(u @ u.conj().T, np.eye(len(u)), atol=1e-12)
    h = soc_matrix_from_vectors(vectors, states)
    h_rotated = soc_matrix_from_vectors(vectors @ rotation.T, states)
    np.testing.assert_allclose(h_rotated, u @ h @ u.conj().T, atol=1e-12)


def test_matrices_that_are_not_wigner_eckart_are_detected():
    states = [1, 0, 1]
    rng = np.random.default_rng(2)
    h = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    h = h + h.conj().T
    _, residual = soc_vectors_from_matrix(h, states)
    assert residual > 0.1


def test_torch_and_numpy_agree_and_metrics_entries():
    torch = pytest.importorskip("torch")
    states = [1, 0, 2]
    rng = np.random.default_rng(4)
    vectors = rng.standard_normal((len(spin_orbit_pairs(states)), 3))
    h_np = soc_matrix_from_vectors(vectors, states)
    h_t = soc_matrix_from_vectors(torch.tensor(vectors), states).numpy()
    np.testing.assert_allclose(h_np, h_t, atol=1e-12)
    rows, cols = soc_matrix_entries(states)
    assert np.all(rows < cols)
    entries = soc_entries_from_vectors(vectors, states)
    np.testing.assert_allclose(entries[: len(rows)], h_np[rows, cols].real)
    np.testing.assert_allclose(entries[len(rows) :], h_np[rows, cols].imag)
    # all other upper-triangle elements vanish
    mask = np.zeros_like(h_np, dtype=bool)
    mask[rows, cols] = True
    assert np.all(h_np[np.triu(~mask, k=1)] == 0)


def test_pair_weights_are_frobenius_norms():
    states = [1, 0, 1, 0, 1]
    coeff = soc_coefficients(states)
    weights = soc_pair_weights(states)
    for p in range(len(weights)):
        norms = [np.sum(np.abs(coeff[:, :, p, k]) ** 2) for k in range(3)]
        assert abs(weights[p] - np.mean(norms)) < 1e-12
