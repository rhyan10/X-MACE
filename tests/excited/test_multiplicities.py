"""Excited states of several multiplicities (doublets, triplets, quartets, quintets)
and spin-orbit couplings with the spin symmetry built in."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from e3nn import o3
from e3nn.util import jit
from scipy.spatial.transform import Rotation

from mace import data, modules
from mace.calculators import ExcitedMACECalculator
from mace.calculators.excited import ExcitedModelLayout, evaluate_excited_configs
from mace.calculators.sharc_calculator import (
    BOHR_TO_ANGSTROM,
    HARTREE_TO_EV,
    SharcCalculator,
)
from mace.data.excited import (
    ExcitedStateSpec,
    excited_spec_from_args,
    soc_matrix_from_label,
    standardise_excited_configs,
)
from mace.modules.spin_states import (
    n_mstates,
    same_multiplicity_pairs,
    soc_matrix_from_vectors,
    spin_orbit_pairs,
    spin_rotation,
    state_pairs,
)
from mace.tools.train import MACELoss

sys.path.insert(0, str(Path(__file__).parent))
from conftest import (  # noqa: E402  pylint: disable=wrong-import-position
    KEYSPEC,
    atoms_to_batch,
    build_excited,
    model_config,
)
from excited_data import (  # noqa: E402  pylint: disable=wrong-import-position
    make_multiplicity_configs,
    multiplicity_labels,
)

SINGLET_TRIPLET_QUINTET = [2, 0, 2, 0, 1]
DOUBLET_QUARTET = [0, 2, 0, 1]


def spin_spec(states, **kwargs):
    n_states = sum(states)
    coupled = same_multiplicity_pairs(states)
    values = dict(
        n_states=n_states,
        n_nacs=len(coupled),
        n_dipoles=n_states + len(coupled),
        states=states,
        soc_mode="spin_orbit",
    )
    values.update(kwargs)
    return ExcitedStateSpec(**values)


def spin_model(states, **kwargs):
    return build_excited(
        n_states=sum(states), states=states, soc_mode="spin_orbit", **kwargs
    )


# ------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------


@pytest.mark.parametrize("states", [SINGLET_TRIPLET_QUINTET, DOUBLET_QUARTET])
def test_standardise_selects_pairs_and_fits_soc_vectors(states):
    atoms_list = make_multiplicity_configs(3, states, seed=1)
    configs = [data.config_from_atoms(a, key_specification=KEYSPEC) for a in atoms_list]
    spec = spin_spec(states)
    counts = standardise_excited_configs(configs, spec)
    assert counts["socs"] == 3 and counts["nacs"] == 3
    n_states = sum(states)
    every = state_pairs(n_states)
    for atoms, config in zip(atoms_list, configs):
        props = config.properties
        assert "socs" not in props
        assert props["soc_vectors"].shape == (len(spin_orbit_pairs(states)), 3)
        # the fitted vectors reproduce the SOC matrix exactly
        matrix = soc_matrix_from_label(atoms.info["REF_socs"], states)
        np.testing.assert_allclose(
            soc_matrix_from_vectors(props["soc_vectors"], states), matrix, atol=1e-12
        )
        # NACs and transition dipoles of same-multiplicity pairs only, smooth NACs
        coupled = same_multiplicity_pairs(states)
        assert props["nacs"].shape == (4, len(coupled), 3)
        energies = atoms.info["REF_energy"]
        for p, (i, j) in enumerate(coupled):
            raw = atoms.info["REF_nacs"][:, every.index((i, j))]
            np.testing.assert_allclose(
                props["nacs"][:, p], raw * abs(energies[j] - energies[i])
            )
            np.testing.assert_allclose(
                props["dipoles"][n_states + p],
                atoms.info["REF_dipoles"][n_states + every.index((i, j))],
            )


@pytest.mark.parametrize(
    "layout",
    [
        "complex",
        "real_imag_last",
        "real_imag_first",
        "upper_complex",
        "upper_interleaved",
    ],
)
def test_soc_label_layouts(layout):
    states = [1, 0, 1]
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((1, 3))
    matrix = soc_matrix_from_vectors(vectors, states)
    upper = matrix[np.triu_indices(4, k=1)]
    value = {
        "complex": matrix,
        "real_imag_last": np.stack([matrix.real, matrix.imag], axis=-1),
        "real_imag_first": np.stack([matrix.real, matrix.imag]),
        "upper_complex": upper,
        "upper_interleaved": np.stack([upper.real, upper.imag], -1).reshape(-1),
    }[layout]
    np.testing.assert_allclose(soc_matrix_from_label(value, states), matrix)


def test_soc_label_errors_and_residual_warning(caplog):
    with pytest.raises(ValueError, match="M_S-resolved"):
        soc_matrix_from_label(np.zeros(7), [1, 0, 1])
    atoms_list = make_multiplicity_configs(2, [1, 0, 1], seed=0)
    nm = 4
    rng = np.random.default_rng(5)
    bad = rng.standard_normal((nm, nm))
    atoms_list[0].info["REF_socs"] = (bad + bad.T).astype(complex)
    configs = [data.config_from_atoms(a, key_specification=KEYSPEC) for a in atoms_list]
    with caplog.at_level("WARNING"):
        standardise_excited_configs(configs, spin_spec([1, 0, 1]))
    assert "Wigner-Eckart" in caplog.text


def _args(**kwargs):
    values = dict(
        model="ExcitedMACE",
        n_energies=None,
        states=None,
        compute_nacs=True,
        compute_socs=True,
        compute_dipoles=True,
        nac_num=None,
        soc_num=None,
        dipole_num=None,
        nacs_format="raw",
        nacs_key="REF_nacs",
        soc_mode="auto",
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_spec_from_args():
    spec = excited_spec_from_args(_args(states="2 0 2 0 1"))
    assert spec.states == [2, 0, 2, 0, 1] and spec.n_states == 5
    assert spec.soc_mode == "spin_orbit" and spec.n_soc_vectors == 7
    assert spec.n_nacs == 2  # S0-S1, T0-T1
    assert spec.n_dipoles == 5 + 2 and spec.n_permanent_dipoles == 5
    singlets = excited_spec_from_args(_args(states="3"))
    assert singlets.soc_mode == "free" and singlets.n_nacs == 3
    no_soc = excited_spec_from_args(
        _args(states="1 0 1", compute_socs=False, compute_nacs=False)
    )
    assert no_soc.soc_mode == "free" and no_soc.n_socs == 0
    with pytest.raises(ValueError, match="n_energies"):
        excited_spec_from_args(_args(states="1 0 2", n_energies=4))
    with pytest.raises(ValueError, match="one multiplicity"):
        excited_spec_from_args(_args(states="1 0 1"))
    with pytest.raises(ValueError, match="spin-orbit"):
        ExcitedStateSpec(n_states=2, states=[1, 1], soc_mode="spin_orbit")
    assert "1 quintet" in spec.describe() and "7 coupling vectors" in spec.describe()


# ------------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------------


def _predict(model, atoms_list):
    batch = atoms_to_batch(atoms_list)
    return model(batch.to_dict(), training=False, compute_force=True)


@pytest.mark.parametrize("states", [SINGLET_TRIPLET_QUINTET, DOUBLET_QUARTET])
def test_model_layout_and_symmetry(states, molecules):
    torch.manual_seed(0)
    model = spin_model(states)
    n_states = sum(states)
    coupled = same_multiplicity_pairs(states)
    assert model.nac_num == len(coupled)
    assert model.dipole_num == n_states + len(coupled)
    assert model.soc_vector_num == len(spin_orbit_pairs(states))
    assert model.multiplicity_states == states and model.soc_mode == "spin_orbit"
    assert int(model.soc_num) == 0
    out = _predict(model, molecules[:2])
    assert out["soc_vectors"].shape == (2, model.soc_vector_num, 3)
    assert out["socs"] is None

    rotation = torch.tensor(Rotation.random(random_state=5).as_matrix())
    rotated = [m.copy() for m in molecules[:2]]
    inverted = [m.copy() for m in molecules[:2]]
    for atoms in rotated:
        atoms.positions = atoms.positions @ rotation.numpy().T
    for atoms in inverted:
        atoms.positions = -atoms.positions
    out_rot = _predict(model, rotated)
    # SOC vectors, NACs and dipoles rotate with the molecule
    torch.testing.assert_close(out_rot["soc_vectors"], out["soc_vectors"] @ rotation.T)
    torch.testing.assert_close(out_rot["dipoles"], out["dipoles"] @ rotation.T)
    torch.testing.assert_close(out_rot["energy"], out["energy"])
    # coupling_parity="fixed": SOC vectors axial (unchanged by inversion), dipoles polar
    fixed = spin_model(states, coupling_parity="fixed")
    out_fixed, out_fixed_inv = _predict(fixed, molecules[:2]), _predict(fixed, inverted)
    torch.testing.assert_close(out_fixed_inv["soc_vectors"], out_fixed["soc_vectors"])
    torch.testing.assert_close(out_fixed_inv["dipoles"], -out_fixed["dipoles"])
    # default "both": the SOC vectors also have a polar part
    out_inv = _predict(model, inverted)
    assert (out_inv["soc_vectors"] - out["soc_vectors"]).abs().max() > 1e-6


def test_model_scripts_and_errors(molecules):
    torch.manual_seed(1)
    model = spin_model([1, 0, 2])
    scripted = jit.compile(model)
    batch = atoms_to_batch(molecules[:1])
    expected = model(batch.to_dict(), training=False, compute_force=True)
    got = scripted(batch.to_dict(), training=False, compute_force=True)
    torch.testing.assert_close(got["soc_vectors"], expected["soc_vectors"])
    with pytest.raises(ValueError, match="describe"):
        build_excited(n_states=3, states=[1, 0, 1])
    with pytest.raises(ValueError, match="needs the states"):
        build_excited(n_states=2, soc_mode="spin_orbit")
    with pytest.raises(ValueError, match="No spin-orbit"):
        build_excited(n_states=2, states=[1, 1], soc_mode="spin_orbit")
    # a backbone without l=1 features: the adapter layer provides them
    scalar = modules.ExcitedMACE(
        **model_config(hidden_irreps=o3.Irreps("8x0e")),
        n_energies=2,
        compute_socs=True,
        states=[1, 0, 1],
        soc_mode="spin_orbit",
    )
    out = scalar(atoms_to_batch(molecules[:1]).to_dict())
    assert out["soc_vectors"].abs().max() > 0


def test_autoencoder_with_multiplicities(molecules):
    torch.manual_seed(2)
    model = build_excited(
        n_states=3, states=[1, 0, 2], soc_mode="spin_orbit", autoencoder=True
    )
    out = _predict(model, molecules[:2])
    assert out["soc_vectors"].shape == (2, 3, 3)
    assert out["nacs"].shape[1] == 1  # T0-T1


# ------------------------------------------------------------------------------------
# loss and metrics
# ------------------------------------------------------------------------------------


def _labelled_batch(states, seed=0, random_phases=True):
    atoms_list = make_multiplicity_configs(
        4, states, seed=seed, random_phases=random_phases
    )
    return atoms_to_batch(atoms_list, spec=spin_spec(states))


def test_soc_loss_is_phase_free_and_metrics_are_matrix_elements():
    states = [1, 0, 2]
    batch = _labelled_batch(states, random_phases=False)
    weights = modules.spin_states.soc_pair_weights(states)
    loss_fn = modules.WeightedExcitedStatesLoss(
        energy_weight=0.0,
        forces_weight=0.0,
        nacs_weight=0.0,
        dipoles_weight=0.0,
        socs_weight=1.0,
        soc_pair_weights=weights / weights.mean(),
        soc_states=states,
    )
    exact = {"soc_vectors": batch.soc_vectors.clone()}
    assert loss_fn(batch, exact) < 1e-5  # only the sqrt epsilon
    # the phase of every state is arbitrary: a prediction with other signs is as good
    signs = torch.tensor(
        [[1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [1.0, 1.0, 1.0], [-1.0, -1.0, 1.0]]
    )
    flipped = {"soc_vectors": batch.soc_vectors * signs.unsqueeze(-1)}
    assert loss_fn(batch, flipped) < 1e-5
    noise = 0.01 * torch.randn_like(
        batch.soc_vectors, generator=torch.Generator().manual_seed(0)
    )
    noisy = {"soc_vectors": (batch.soc_vectors + noise) * signs.unsqueeze(-1)}
    assert loss_fn(batch, noisy) > 1e-3

    metric = MACELoss(loss_fn=loss_fn)
    metric.update(batch, noisy)
    _, aux = metric.compute()
    matrix_ref = soc_matrix_from_vectors(batch.soc_vectors, states)
    matrix_pred = soc_matrix_from_vectors(batch.soc_vectors + noise, states)
    rows, cols = modules.spin_states.soc_matrix_entries(states)
    diff = (matrix_ref - matrix_pred)[:, rows, cols]
    expected_rmse = np.sqrt(
        np.mean(np.concatenate([diff.real.numpy(), diff.imag.numpy()], axis=-1) ** 2)
    )
    assert abs(aux["rmse_socs"] - expected_rmse) < 1e-10


def test_spin_orbit_model_learns_couplings():
    states = [1, 0, 2]
    torch.manual_seed(3)
    model = spin_model(states)
    batch = _labelled_batch(states, seed=4)
    spec = spin_spec(states)
    weights = modules.spin_states.soc_pair_weights(states)
    loss_fn = modules.WeightedExcitedStatesLoss(
        nacs_weight=1.0,
        socs_weight=10.0,
        n_permanent_dipoles=spec.n_permanent_dipoles,
        soc_pair_weights=weights / weights.mean(),
        soc_states=states,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    losses = []
    for _ in range(60):
        optimizer.zero_grad()
        out = model(batch.to_dict(), training=True, compute_force=True)
        loss = loss_fn(batch, out)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]


# ------------------------------------------------------------------------------------
# calculators
# ------------------------------------------------------------------------------------


@pytest.fixture(name="spin_model_files")
def fixture_spin_model_files(tmp_path):
    torch.manual_seed(7)
    paths = {}
    for name, states in (("stq", SINGLET_TRIPLET_QUINTET), ("dq", DOUBLET_QUARTET)):
        path = tmp_path / f"{name}.model"
        torch.save(spin_model(states), path)
        paths[name] = path
    return paths


def test_calculator_soc_matrix_and_committee(spin_model_files, molecules, tmp_path):
    calc = ExcitedMACECalculator(model_paths=str(spin_model_files["stq"]))
    layout = calc.layout
    assert layout == ExcitedModelLayout.from_model(calc.models[0])
    assert layout.states == SINGLET_TRIPLET_QUINTET
    assert layout.nac_pairs == same_multiplicity_pairs(SINGLET_TRIPLET_QUINTET)
    results = calc.predict(molecules[0])
    nm = n_mstates(SINGLET_TRIPLET_QUINTET)
    assert results["soc_matrix"].shape == (nm, nm)
    np.testing.assert_allclose(
        results["soc_matrix"],
        soc_matrix_from_vectors(results["soc_vectors"], SINGLET_TRIPLET_QUINTET),
    )
    # physical NACs use the gap of their own pair
    energies = results["energies"]
    for p, (i, j) in enumerate(layout.nac_pairs):
        np.testing.assert_allclose(
            results["nacs"][:, p] * abs(energies[j] - energies[i]),
            results["smooth_nacs"][:, p],
        )
    # committee of the same model with flipped signs: aligned before averaging
    model = calc.models[0]
    committee = ExcitedMACECalculator(model_paths=None, models=[model, model])
    np.testing.assert_allclose(
        committee.predict(molecules[0])["soc_vectors"], results["soc_vectors"]
    )
    atoms_list = [m.copy() for m in molecules[:2]]
    evaluate_excited_configs(model, atoms_list)
    assert atoms_list[0].info["MACE_soc_vectors"].shape == (7, 3)
    np.testing.assert_allclose(
        soc_matrix_from_label(atoms_list[0].info["MACE_socs"], SINGLET_TRIPLET_QUINTET),
        results["soc_matrix"],
        atol=1e-10,
    )
    ase_io = pytest.importorskip("ase.io")
    ase_io.write(tmp_path / "out.xyz", atoms_list)
    back = ase_io.read(tmp_path / "out.xyz", ":")
    np.testing.assert_allclose(
        np.asarray(back[0].info["MACE_socs"]).reshape(-1),
        atoms_list[0].info["MACE_socs"],
    )


PROPERTIES = ["energy", "forces", "nacs", "socs", "dipoles"]


def _sharc(path, states):
    return SharcCalculator(
        atom_types=["C", "H", "H", "O"],
        model_paths=str(path),
        states=states,
        properties=PROPERTIES,
    )


@pytest.mark.parametrize(
    "name,states", [("stq", SINGLET_TRIPLET_QUINTET), ("dq", DOUBLET_QUARTET)]
)
def test_sharc_multiplicities(spin_model_files, molecules, name, states):
    sharc = _sharc(spin_model_files[name], states)
    assert sharc.mode == "multiplicities" and sharc.soc_source == "spin_orbit"
    coords = molecules[0].positions / BOHR_TO_ANGSTROM
    qm = sharc.calculate(coords)
    results = sharc.calc.predict(molecules[0])
    nm = n_mstates(states)
    h, grad, nacdr, dm = qm["h"], qm["grad"], qm["nacdr"], qm["dm"]
    assert h.shape == (nm, nm) and grad.shape == (nm, 4, 3)
    np.testing.assert_allclose(h, h.conj().T, atol=1e-15)
    from mace.modules.spin_states import ms_resolved_map

    ms_map = ms_resolved_map(states)
    mult = [m + 1 for m, n in enumerate(states) for _ in range(n)]
    energies = results["energies"] / HARTREE_TO_EV
    for k, comps in enumerate(ms_map):
        for index, _ in comps:
            # M_S components are degenerate: energy, gradient, permanent dipole
            assert abs(h[index, index].real - energies[k]) < 1e-12
            np.testing.assert_allclose(grad[index], grad[comps[0][0]])
            np.testing.assert_allclose(
                dm[:, index, index], dm[:, comps[0][0], comps[0][0]]
            )
    # NACs and transition dipoles: same multiplicity and same M_S only
    for k, comps_k in enumerate(ms_map):
        for l, comps_l in enumerate(ms_map):
            for a, ma in comps_k:
                for b, mb in comps_l:
                    if mult[k] != mult[l] or ma != mb or k == l:
                        assert np.all(nacdr[a, b] == 0)
                        if k != l:
                            assert np.all(dm[:, a, b] == 0)
    assert np.any(nacdr != 0)
    np.testing.assert_allclose(nacdr, -np.transpose(nacdr, (1, 0, 2, 3)))
    # SOC block equals the Wigner-Eckart matrix of the predicted vectors (in Hartree)
    soc = h - np.diag(np.diag(h))
    np.testing.assert_allclose(
        soc,
        soc_matrix_from_vectors(results["soc_vectors"], states) / HARTREE_TO_EV,
        atol=1e-15,
    )
    # rotating the molecule rotates the spins: H_SO(R x) = U(R) H_SO(x) U(R)^dagger
    rotation = Rotation.random(random_state=11).as_matrix()
    sharc.reset_phases()
    rotated = sharc.calculate(coords @ rotation.T)
    u = spin_rotation(states, rotation)
    soc_rotated = rotated["h"] - np.diag(np.diag(rotated["h"]))
    np.testing.assert_allclose(soc_rotated, u @ soc @ u.conj().T, atol=1e-12)
    np.testing.assert_allclose(rotated["grad"], grad @ rotation.T, atol=1e-12)


def test_sharc_uses_the_lowest_states_and_phase_continuity(spin_model_files, molecules):
    full = _sharc(spin_model_files["stq"], SINGLET_TRIPLET_QUINTET)
    fewer = _sharc(spin_model_files["stq"], [1, 0, 2])  # no quintet, one singlet
    coords = molecules[0].positions / BOHR_TO_ANGSTROM
    qm_full, qm_few = full.calculate(coords), fewer.calculate(coords)
    # S0 T0 T1 of the full calculation, in SHARC order
    keep = [0, 2, 3, 4, 5, 6, 7]
    np.testing.assert_allclose(
        qm_few["h"], qm_full["h"][np.ix_(keep, keep)], atol=1e-15
    )
    np.testing.assert_allclose(qm_few["nacdr"], qm_full["nacdr"][np.ix_(keep, keep)])
    with pytest.raises(ValueError, match="predicts"):
        _sharc(spin_model_files["stq"], [3, 0, 2])
    with pytest.raises(ValueError, match="predicts"):
        _sharc(spin_model_files["dq"], [0, 2, 0, 1, 0, 1])
    # phases follow the previous step: a small move keeps the signs
    first = fewer.calculate(coords)
    moved = fewer.calculate(coords + 1e-3)
    assert np.sum(first["nacdr"] * moved["nacdr"]) > 0
    vectors_before = fewer._previous["soc_vectors"].copy()
    fewer.calculate(coords + 2e-3)
    assert np.all(np.sum(fewer._previous["soc_vectors"] * vectors_before, axis=-1) > 0)


def test_sharc_finite_difference_gradients(spin_model_files, molecules):
    sharc = _sharc(spin_model_files["dq"], DOUBLET_QUARTET)
    coords = molecules[0].positions / BOHR_TO_ANGSTROM
    grad = sharc.calculate(coords)["grad"]
    step = 1e-4
    for atom, axis in ((0, 0), (3, 2)):
        plus, minus = coords.copy(), coords.copy()
        plus[atom, axis] += step
        minus[atom, axis] -= step
        e_plus = np.diag(sharc.calculate(plus)["h"]).real
        e_minus = np.diag(sharc.calculate(minus)["h"]).real
        np.testing.assert_allclose(
            (e_plus - e_minus) / (2 * step), grad[:, atom, axis], atol=1e-7
        )


def test_labels_from_sharc_round_trip(spin_model_files, molecules):
    """SharcCalculator output (SHARC units, M_S resolved) turned back into labels
    gives the model's own predictions."""
    from mace.data.sharc_labels import atoms_from_sharc, labels_from_sharc

    states = SINGLET_TRIPLET_QUINTET
    sharc = _sharc(spin_model_files["stq"], states)
    coords = molecules[0].positions / BOHR_TO_ANGSTROM
    qm = sharc.calculate(coords)
    results = sharc.calc.predict(molecules[0])
    labels = labels_from_sharc(
        states, qm["h"], qm["grad"], qm["nacdr"], qm["dm"], soc_unit="eV"
    )
    np.testing.assert_allclose(labels["energies"], results["energies"], atol=1e-10)
    np.testing.assert_allclose(labels["forces"], results["forces_states"], atol=1e-10)
    np.testing.assert_allclose(labels["nacs"], results["nacs"], atol=1e-10)
    np.testing.assert_allclose(labels["dipoles"], results["dipoles"], atol=1e-10)
    np.testing.assert_allclose(labels["socs"], results["soc_matrix"], atol=1e-12)
    atoms = atoms_from_sharc(
        ["C", "H", "H", "O"], coords, states, qm["h"], qm["grad"], qm["nacdr"], qm["dm"]
    )
    np.testing.assert_allclose(atoms.positions, molecules[0].positions)
    assert atoms.info["REF_socs"].shape == (13 * 12,)
    bad = qm["h"].copy()
    bad[3, 3] += 1e-3  # an M_S component off the multiplet energy
    with pytest.raises(ValueError, match="MCH"):
        labels_from_sharc(states, bad)


def test_planar_molecules_need_both_parities(molecules):
    """For a planar geometry, polar functions of the positions lie in the plane and
    axial ones are perpendicular to it. Couplings between states of opposite
    reflection symmetry need the other direction (e.g. out-of-plane n->pi*
    transition dipoles, in-plane SOC vectors), which coupling_parity="both" gives."""
    planar = molecules[0].copy()
    planar.positions[:, 2] = 0.0  # molecule in the xy plane, normal along z
    torch.manual_seed(5)
    states = [2, 0, 2]
    fixed = spin_model(states, coupling_parity="fixed")
    both = spin_model(states, coupling_parity="both")
    out_fixed, out_both = _predict(fixed, [planar]), _predict(both, [planar])
    n_states = sum(states)
    transition_fixed = out_fixed["dipoles"][0, n_states:]
    transition_both = out_both["dipoles"][0, n_states:]
    assert transition_fixed[:, 2].abs().max() < 1e-10  # polar: in plane only
    assert transition_both[:, 2].abs().max() > 1e-6  # out of plane too
    assert out_fixed["soc_vectors"][0, :, :2].abs().max() < 1e-10  # axial: normal only
    assert out_both["soc_vectors"][0, :, :2].abs().max() > 1e-6  # in plane too
    assert out_fixed["nacs"][..., 2].abs().max() < 1e-10
    assert out_both["nacs"][..., 2].abs().max() > 1e-6
    # permanent dipoles of a planar molecule lie in its plane in both cases
    assert out_both["dipoles"][0, :n_states, 2].abs().max() < 1e-10


def test_autoencoder_orders_states_within_each_multiplicity(molecules):
    torch.manual_seed(6)
    states = [2, 0, 3]
    model = build_excited(n_states=5, states=states, soc_mode="spin_orbit", autoencoder=True)
    assert model.perm_decoders[0].block_sizes == [2, 3]
    assert model.perm_encoders[0].n_groups == 2
    # the energies of each multiplicity are sorted (one shift per multiplicity)
    model.delta_shift[0] = torch.tensor([0.3, -0.2, 1.0, -1.0, 0.5])
    energies = _predict(model, molecules[:3])["energy"].detach()
    assert torch.all(energies[:, 1] >= energies[:, 0] - 1e-9)
    assert torch.all(torch.diff(energies[:, 2:], dim=1) >= -1e-9)
    # the encoder distinguishes multiplicities: swapping a singlet and a triplet energy
    # changes the encoding, swapping two triplets does not
    reference = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])
    swap_triplets = reference[:, [0, 1, 3, 2, 4]]
    swap_mult = reference[:, [2, 1, 0, 3, 4]]
    enc = model.perm_encoders[0]
    torch.testing.assert_close(enc(swap_triplets), enc(reference))
    assert (enc(swap_mult) - enc(reference)).abs().max() > 1e-6
    from e3nn.util import jit

    jit.compile(model)
