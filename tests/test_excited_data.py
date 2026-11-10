"""Excited-state labels: parsing, standardisation, smooth NACs, missing labels, batching."""

import ase
import ase.io
import numpy as np
import pytest
import torch

from mace import data
from mace.data.excited import (
    ExcitedStateSpec,
    energy_gaps,
    num_state_pairs,
    resolve_nacs_format,
    standardise_excited_configs,
)
from mace.tools import AtomicNumberTable

from conftest import KEYSPEC, atoms_to_batch, full_spec  # pylint: disable=import-error
from excited_data import excited_labels, make_excited_configs  # pylint: disable=import-error


def _config(atoms):
    return data.config_from_atoms(atoms, key_specification=KEYSPEC)


def test_forces_and_nacs_are_read_from_info(molecules):
    config = _config(molecules[0])
    assert config.properties["forces"].shape == (4, 3, 3)
    assert config.properties["nacs"].shape == (4, 3, 3)
    assert config.property_weights["forces"] == 1.0


def test_extxyz_round_trip_keeps_shapes(tmp_path, molecules):
    ase.io.write(tmp_path / "x.xyz", molecules)
    atoms = ase.io.read(tmp_path / "x.xyz", 0)
    config = _config(atoms)
    standardise_excited_configs([config], full_spec())
    assert config.properties["energy"].shape == (3,)
    assert config.properties["forces"].shape == (4, 3, 3)
    assert config.properties["nacs"].shape == (4, 3, 3)
    assert config.properties["socs"].shape == (3,)
    assert config.properties["dipoles"].shape == (6, 3)


def test_raw_nacs_become_smooth(molecules):
    config = _config(molecules[0])
    raw = config.properties["nacs"].copy()
    gaps = energy_gaps(config.properties["energy"])
    standardise_excited_configs([config], full_spec(nacs_format="raw"))
    np.testing.assert_allclose(config.properties["nacs"], raw * gaps[None, :, None])
    labels = excited_labels(molecules[0].positions, 3)
    np.testing.assert_allclose(config.properties["nacs"], labels["smooth_nacs"], atol=1e-12)


def test_smooth_nacs_are_kept(molecules):
    config = _config(molecules[0])
    raw = config.properties["nacs"].copy()
    standardise_excited_configs([config], full_spec(nacs_format="smooth"))
    np.testing.assert_allclose(config.properties["nacs"], raw)


@pytest.mark.parametrize(
    "key,expected", [("smooth_nacs", "smooth"), ("REF_nacs", "raw"), ("nacs", "raw")]
)
def test_auto_nacs_format(key, expected):
    assert resolve_nacs_format("auto", key) == expected
    assert resolve_nacs_format("raw", "smooth_nacs") == "raw"


def test_alternative_layouts_are_accepted(molecules):
    atoms = molecules[0].copy()
    atoms.info = dict(molecules[0].info)
    forces = atoms.info["REF_forces"]
    nacs = atoms.info["REF_nacs"]
    atoms.info["REF_forces"] = np.transpose(forces, (1, 0, 2))  # [S, N, 3]
    atoms.info["REF_nacs"] = np.transpose(nacs, (1, 0, 2))  # [P, N, 3]
    a = _config(atoms)
    b = _config(molecules[0])
    spec = full_spec(nacs_format="smooth")
    standardise_excited_configs([a, b], spec)
    np.testing.assert_allclose(a.properties["forces"], b.properties["forces"])
    np.testing.assert_allclose(a.properties["nacs"], b.properties["nacs"])
    # flattened per-atom arrays, as atoms.arrays would store them
    atoms.info.pop("REF_forces")
    atoms.arrays["REF_forces"] = forces.reshape(4, 9)
    c = _config(atoms)
    standardise_excited_configs([c], spec)
    np.testing.assert_allclose(c.properties["forces"], forces)


def test_single_state_ground_state_layout():
    atoms = ase.Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.2]])
    atoms.info["REF_energy"] = -3.0
    atoms.arrays["REF_forces"] = np.ones((2, 3))
    config = _config(atoms)
    standardise_excited_configs([config], ExcitedStateSpec(n_states=1))
    assert config.properties["energy"].shape == (1,)
    assert config.properties["forces"].shape == (2, 1, 3)


def test_missing_labels_get_zero_weight(molecules):
    atoms = molecules[1].copy()
    atoms.info = {"REF_energy": molecules[1].info["REF_energy"]}  # energy only
    configs = [_config(molecules[0]), _config(atoms)]
    counts = standardise_excited_configs(configs, full_spec())
    assert counts == {
        "energy": 2,
        "forces": 1,
        "nacs": 1,
        "socs": 1,
        "dipoles": 1,
        "oscillator_strengths": 0,
    }
    weights = configs[1].property_weights
    for name in ("forces", "nacs", "socs", "dipoles"):
        assert weights[name] == 0.0
    assert weights["energy"] == 1.0
    assert configs[1].properties["nacs"].shape == (4, 3, 3)
    # frames without labels batch together with labelled ones
    batch = atoms_to_batch([molecules[0], atoms], full_spec())
    assert batch.nacs.shape == (8, 3, 3)
    assert torch.equal(batch.nacs_weight, torch.tensor([1.0, 0.0]))
    assert torch.equal(batch.forces_weight, torch.tensor([1.0, 0.0]))


def test_raw_nacs_without_energies_are_dropped(molecules):
    atoms = molecules[0].copy()
    atoms.info = {"REF_nacs": molecules[0].info["REF_nacs"]}
    config = _config(atoms)
    standardise_excited_configs([config], full_spec(nacs_format="raw"))
    assert config.property_weights["nacs"] == 0.0


def test_wrong_number_of_states_is_reported(molecules):
    config = _config(molecules[0])
    with pytest.raises(ValueError, match="expected 4 energies"):
        standardise_excited_configs([config], ExcitedStateSpec(n_states=4))


def test_wrong_soc_count_is_reported(molecules):
    config = _config(molecules[0])
    with pytest.raises(ValueError, match="SOC values"):
        standardise_excited_configs([config], full_spec(n_socs=5))


def test_complex_socs_are_rejected(molecules):
    atoms = molecules[0].copy()
    atoms.info = dict(molecules[0].info)
    atoms.info["REF_socs"] = np.array([1 + 1j, 2, 3])
    with pytest.raises(ValueError, match="complex"):
        standardise_excited_configs([_config(atoms)], full_spec())


def test_custom_nac_layout_needs_smooth_nacs():
    with pytest.raises(ValueError, match="smooth"):
        ExcitedStateSpec(n_states=3, n_nacs=2, nacs_format="raw")
    ExcitedStateSpec(n_states=3, n_nacs=2, nacs_format="smooth")


def test_disabled_properties_are_removed(molecules):
    config = _config(molecules[0])
    standardise_excited_configs([config], ExcitedStateSpec(n_states=3))
    assert "nacs" not in config.properties
    assert "socs" not in config.properties


def test_batch_shapes(molecules):
    batch = atoms_to_batch(molecules, full_spec())
    n_atoms = 4 * len(molecules)
    assert batch.energy.shape == (4, 3)
    assert batch.forces.shape == (n_atoms, 3, 3)
    assert batch.nacs.shape == (n_atoms, 3, 3)
    assert batch.socs.shape == (4, 3)
    assert batch.dipoles.shape == (4, 6, 3)
    for name in ("nacs", "socs", "dipoles"):
        assert batch[f"{name}_weight"].shape == (4,)


def test_average_e0s_with_states_and_missing_energies(molecules):
    z_table = AtomicNumberTable([1, 6, 8])
    configs = [_config(a) for a in make_excited_configs(12, 3, seed=5)]
    unlabelled = _config(molecules[0])
    unlabelled.properties["energy"] = None
    unlabelled.property_weights["energy"] = 0.0
    e0s = data.compute_average_E0s(configs + [unlabelled], z_table)
    assert set(e0s) == {1, 6, 8}
    assert all(np.isfinite(v) for v in e0s.values())


def test_num_state_pairs():
    assert [num_state_pairs(s) for s in (1, 2, 3, 4)] == [0, 1, 3, 6]
