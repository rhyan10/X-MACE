"""ExcitedMACE / AutoencoderExcitedMACE: symmetries, forces, core repulsion, TorchScript."""

import ase
import numpy as np
import pytest
import torch
from e3nn import o3
from e3nn.util import jit

from mace import data, modules
from mace.modules.excited import (
    AutoencoderExcitedMACE,
    ExcitedMACE,
    PermutationInvariantDecoder,
    compute_excited_scaling,
    compute_multistate_forces,
    load_foundations_excited,
)
from mace.modules.radial import ZBLBasis
from mace.tools import torch_geometric

from conftest import (  # pylint: disable=import-error
    R_MAX,
    Z_TABLE,
    atoms_to_batch,
    build_excited,
    full_spec,
    model_config,
)
from excited_data import make_excited_configs  # pylint: disable=import-error

BOTH = [False, True]  # autoencoder


def _rotated(atoms_list, rotation, shift=np.zeros(3), permute=None):
    out = []
    for atoms in atoms_list:
        new = atoms.copy()
        new.info = dict(atoms.info)
        positions = atoms.positions @ rotation.T + shift
        if permute is not None:
            positions = positions[permute]
            new = ase.Atoms([atoms.get_chemical_symbols()[i] for i in permute], positions=positions)
        else:
            new.positions = positions
        out.append(new)
    return out


@pytest.mark.parametrize("autoencoder", BOTH)
def test_output_shapes(molecules, autoencoder):
    model = build_excited(autoencoder=autoencoder, soc_num=4, dipole_num=5)
    batch = atoms_to_batch(molecules, None)
    out = model(batch.to_dict(), training=True)
    n_atoms = batch.positions.shape[0]
    assert out["energy"].shape == (4, 3)
    assert out["forces"].shape == (n_atoms, 3, 3)
    assert out["nacs"].shape == (n_atoms, 3, 3)
    assert out["socs"].shape == (4, 4)
    assert out["dipoles"].shape == (4, 5, 3)
    if autoencoder:
        assert out["invariant_vals"].shape == (4, 6)
        assert out["encoded_energy"].shape == (4, 6)
        assert out["decoded_energy"].shape == (4, 3)
    else:
        assert out["node_energy"].shape == (n_atoms, 3)
        assert out["contributions"].shape == (4, 3, 4)
        torch.testing.assert_close(out["contributions"].sum(-1), out["energy"])


def test_energy_only_single_state_model(molecules):
    model = build_excited(n_states=1, nacs=False, socs=False, dipoles=False)
    batch = atoms_to_batch(molecules, None)
    out = model(batch.to_dict())
    assert out["energy"].shape == (4, 1)
    assert out["nacs"] is None and out["socs"] is None and out["dipoles"] is None
    # without vector outputs the last layer is reduced to scalars, as in MACE
    assert not model.keep_last_layer_irreps


def test_vector_outputs_keep_l1_in_last_layer():
    model = build_excited(nacs=True, socs=False, dipoles=False)
    assert model.keep_last_layer_irreps
    last_irreps = o3.Irreps(str(model.products[-1].linear.irreps_out))
    assert last_irreps.count(o3.Irrep(1, -1)) > 0
    assert len(model.nacs_readouts) == 2  # both layers contribute to the NACs


@pytest.mark.parametrize("autoencoder", BOTH)
def test_rotation_translation_permutation_symmetry(molecules, autoencoder):
    model = build_excited(autoencoder=autoencoder)
    spec = full_spec()
    out = model(atoms_to_batch(molecules, spec).to_dict())
    rotation = o3.rand_matrix().numpy()
    perm = [2, 0, 3, 1]
    moved = _rotated(molecules, rotation, shift=np.array([3.0, -1.0, 2.0]), permute=perm)
    out_r = model(atoms_to_batch(moved, None).to_dict())
    R = torch.tensor(rotation)
    torch.testing.assert_close(out_r["energy"], out["energy"])
    torch.testing.assert_close(out_r["socs"], out["socs"])
    n = 4
    index = torch.cat([torch.tensor(perm) + n * g for g in range(len(molecules))])
    torch.testing.assert_close(out_r["forces"], out["forces"][index] @ R.T)
    torch.testing.assert_close(out_r["nacs"], out["nacs"][index] @ R.T)
    torch.testing.assert_close(out_r["dipoles"], out["dipoles"] @ R.T)


def test_inversion_flips_vectors(molecules):
    # coupling_parity="fixed": NACs and all dipoles are polar vectors
    model = build_excited(coupling_parity="fixed")
    out = model(atoms_to_batch(molecules, None).to_dict())
    out_i = model(atoms_to_batch(_rotated(molecules, -np.eye(3)), None).to_dict())
    torch.testing.assert_close(out_i["energy"], out["energy"])
    torch.testing.assert_close(out_i["nacs"], -out["nacs"])
    torch.testing.assert_close(out_i["dipoles"], -out["dipoles"])


def test_both_parities_under_inversion(molecules):
    # coupling_parity="both" (default): couplings have a polar and an axial part;
    # permanent dipoles stay polar
    model = build_excited()
    out = model(atoms_to_batch(molecules, None).to_dict())
    out_i = model(atoms_to_batch(_rotated(molecules, -np.eye(3)), None).to_dict())
    torch.testing.assert_close(out_i["energy"], out["energy"])
    n_states = out["energy"].shape[1]
    torch.testing.assert_close(out_i["dipoles"][:, :n_states], -out["dipoles"][:, :n_states])
    polar_nacs = 0.5 * (out["nacs"] - out_i["nacs"])
    axial_nacs = 0.5 * (out["nacs"] + out_i["nacs"])
    assert polar_nacs.abs().max() > 1e-6 and axial_nacs.abs().max() > 1e-6
    axial_dipoles = 0.5 * (out["dipoles"] + out_i["dipoles"])[:, n_states:]
    assert axial_dipoles.abs().max() > 1e-6


@pytest.mark.parametrize("autoencoder", BOTH)
def test_forces_match_finite_differences(molecules, autoencoder):
    model = build_excited(autoencoder=autoencoder, pair_repulsion=True)
    atoms = molecules[0]
    batch = atoms_to_batch([atoms], None)
    forces = model(batch.to_dict())["forces"].detach().numpy()
    h = 1e-5
    for atom in range(len(atoms)):
        for xyz in range(3):
            energies = []
            for sign in (1, -1):
                displaced = atoms.copy()
                displaced.positions[atom, xyz] += sign * h
                e = model(atoms_to_batch([displaced], None).to_dict())["energy"]
                energies.append(e.detach().numpy()[0])
            fd = -(energies[0] - energies[1]) / (2 * h)
            np.testing.assert_allclose(forces[atom, :, xyz], fd, atol=1e-6, rtol=1e-5)


def test_multistate_forces_helper():
    positions = torch.randn(5, 3, requires_grad=True)
    energy = torch.stack(
        [(positions**2).sum(), (positions**3).sum(), positions.sum()]
    ).unsqueeze(0)
    forces = compute_multistate_forces(energy, positions, training=False)
    torch.testing.assert_close(forces[:, 0], -2 * positions)
    torch.testing.assert_close(forces[:, 1], -3 * positions**2)
    torch.testing.assert_close(forces[:, 2], -torch.ones(5, 3))


def test_training_mode_keeps_graph_for_force_loss(molecules):
    model = build_excited()
    out = model(atoms_to_batch(molecules, full_spec()).to_dict(), training=True)
    (out["forces"] ** 2).sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)


@pytest.mark.parametrize("autoencoder", BOTH)
def test_core_repulsion_is_state_independent(molecules, autoencoder):
    torch.manual_seed(1)
    with_zbl = build_excited(autoencoder=autoencoder, pair_repulsion=True)
    without = build_excited(autoencoder=autoencoder, pair_repulsion=False)
    without.load_state_dict(
        {k: v for k, v in with_zbl.state_dict().items() if not k.startswith("pair_repulsion_fn")}
    )
    batch = atoms_to_batch(molecules, None)
    out_zbl = with_zbl(batch.to_dict())
    out = without(batch.clone().to_dict())
    difference = (out_zbl["energy"] - out["energy"]).detach()
    # the same repulsion for every state ...
    torch.testing.assert_close(difference, difference[:, :1].expand_as(difference))
    # ... equal to the ZBL energy of each molecule
    zbl = ZBLBasis(p=5)
    vectors, lengths = modules.utils.get_edge_vectors_and_lengths(
        batch.positions, batch.edge_index, batch.shifts
    )
    node = zbl(lengths, batch.node_attrs, batch.edge_index, with_zbl.atomic_numbers)
    expected = torch.zeros(len(molecules)).index_add(0, batch.batch, node)
    torch.testing.assert_close(difference[:, 0], expected.detach())
    torch.testing.assert_close(out_zbl["pair_energy"].detach(), expected.detach())
    # couplings do not see the repulsion
    torch.testing.assert_close(out_zbl["nacs"], out["nacs"])


def test_core_repulsion_dominates_at_short_range():
    model = build_excited(nacs=False, socs=False, dipoles=False, pair_repulsion=True)
    energies = []
    for distance in (0.3, 0.6, 1.0):
        atoms = ase.Atoms("CO", positions=[[0, 0, 0], [0, 0, distance]])
        energies.append(model(atoms_to_batch([atoms], None).to_dict())["energy"].detach())
    assert torch.all(energies[0] > energies[1]) and torch.all(energies[1] > energies[2])
    assert torch.all(energies[0] - energies[2] > 10.0)  # eV


def test_scale_and_shift(molecules):
    torch.manual_seed(2)
    base = build_excited(nacs=False, socs=False, dipoles=False)
    shifted = build_excited(
        nacs=False, socs=False, dipoles=False, atomic_inter_scale=2.0, atomic_inter_shift=[0.1, 0.2, 0.3]
    )
    state = base.state_dict()
    state["scale_shift.scale"] = shifted.scale_shift.scale
    state["scale_shift.shift"] = shifted.scale_shift.shift
    shifted.load_state_dict(state)
    batch = atoms_to_batch(molecules, None)
    e0 = base.atomic_energies_fn(batch.node_attrs)[:, 0]
    e0 = torch.zeros(4).index_add(0, batch.batch, e0)
    inter = base(batch.to_dict())["energy"].detach() - e0[:, None]
    expected = 2.0 * inter + 4 * torch.tensor([0.1, 0.2, 0.3])
    out = shifted(batch.clone().to_dict())["energy"].detach() - e0[:, None]
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("autoencoder", BOTH)
def test_batching_matches_single_configurations(molecules, autoencoder):
    model = build_excited(autoencoder=autoencoder)
    together = model(atoms_to_batch(molecules, None).to_dict())
    for g, atoms in enumerate(molecules):
        alone = model(atoms_to_batch([atoms], None).to_dict())
        torch.testing.assert_close(alone["energy"][0], together["energy"][g])
        torch.testing.assert_close(alone["socs"][0], together["socs"][g])
        torch.testing.assert_close(alone["dipoles"][0], together["dipoles"][g])
        torch.testing.assert_close(alone["nacs"], together["nacs"][4 * g : 4 * g + 4])
        torch.testing.assert_close(alone["forces"], together["forces"][4 * g : 4 * g + 4])


def test_isolated_atom_has_e0_energy():
    model = build_excited(nacs=False, socs=False, dipoles=False, atomic_inter_shift=0.0)
    atoms = ase.Atoms("O", positions=[[0, 0, 0]])
    out = model(atoms_to_batch([atoms], None).to_dict())
    # no neighbours: only E0 and the (zero-shift) readouts of the bare embedding
    assert out["energy"].shape == (1, 3)
    torch.testing.assert_close(out["forces"], torch.zeros(1, 3, 3))


@pytest.mark.parametrize("autoencoder", BOTH)
def test_torchscript(molecules, autoencoder):
    model = build_excited(autoencoder=autoencoder, pair_repulsion=True)
    scripted = jit.compile(model)
    batch = atoms_to_batch(molecules, full_spec())
    out = model(batch.to_dict())
    out_s = scripted(batch.clone().to_dict())
    for key in ("energy", "forces", "nacs", "socs", "dipoles"):
        torch.testing.assert_close(out_s[key], out[key])


@pytest.mark.parametrize("autoencoder", BOTH)
def test_save_and_load(tmp_path, molecules, autoencoder):
    model = build_excited(autoencoder=autoencoder, pair_repulsion=True)
    torch.save(model, tmp_path / "m.model")
    loaded = torch.load(tmp_path / "m.model", weights_only=False)
    batch = atoms_to_batch(molecules, None)
    torch.testing.assert_close(loaded(batch.to_dict())["energy"], model(batch.clone().to_dict())["energy"])


def test_float32(molecules):
    torch.set_default_dtype(torch.float32)
    model = build_excited()
    out = model(atoms_to_batch(molecules, None).to_dict())
    assert out["energy"].dtype == torch.float32
    assert out["nacs"].dtype == torch.float32


def test_autoencoder_energies_are_sorted(molecules):
    model = build_excited(autoencoder=True)
    out = model(atoms_to_batch(molecules, None).to_dict())
    energies = out["energy"].detach()
    assert torch.all(energies[:, 1:] >= energies[:, :-1])


def test_autoencoder_reconstruction_path(molecules):
    model = build_excited(autoencoder=True, atomic_inter_scale=1.5, atomic_inter_shift=[0.1, 0.2, 0.3])
    batch = atoms_to_batch(molecules, full_spec())
    out = model(batch.to_dict())
    # decoded_energy = baseline + scale * decoder(encoder((E_ref - baseline) / scale))
    num_atoms = (batch.ptr[1:] - batch.ptr[:-1]).double()
    e0 = torch.zeros(4).index_add(0, batch.batch, model.atomic_energies_fn(batch.node_attrs)[:, 0])
    shift = model.shift_average @ model.scale_shift.shift  # one shift per multiplicity
    baseline = e0[:, None] + num_atoms[:, None] * shift
    encoded = model.perm_encoder((batch.energy - baseline) / 1.5)
    torch.testing.assert_close(out["encoded_energy"], encoded)
    torch.testing.assert_close(out["decoded_energy"], baseline + 1.5 * model.perm_decoder(encoded))


def test_autoencoder_is_permutation_invariant_over_states():
    torch.manual_seed(3)
    model = build_excited(autoencoder=True, nacs=False, socs=False, dipoles=False)
    energies = torch.randn(5, 3)
    latent = model.perm_encoder(energies)
    torch.testing.assert_close(model.perm_encoder(energies[:, [2, 0, 1]]), latent)


def test_decoder_handles_one_state():
    decoder = PermutationInvariantDecoder(latent_dim=4, n_energies=1)
    assert decoder(torch.randn(3, 4)).shape == (3, 1)


def test_invalid_configurations_are_rejected():
    with pytest.raises(ValueError, match="n_energies >= 2"):
        ExcitedMACE(**model_config(), n_energies=1, compute_nacs=True)
    with pytest.raises(ValueError, match="l=1"):
        ExcitedMACE(
            **model_config(hidden_irreps=o3.Irreps("8x0e")), n_energies=2, compute_nacs=True
        )
    with pytest.raises(NotImplementedError):
        ExcitedMACE(**model_config(), n_energies=2, heads=["a", "b"])
    # custom NAC count with one state is allowed
    ExcitedMACE(**model_config(), n_energies=1, compute_nacs=True, nac_num=2)


def test_excited_scaling_statistics():
    batch_atoms = make_excited_configs(20, 3, seed=7)
    configs = [data.config_from_atoms(a, key_specification=__import__("conftest").KEYSPEC) for a in batch_atoms]
    from mace.data.excited import standardise_excited_configs

    standardise_excited_configs(configs, full_spec())
    dataset = [data.AtomicData.from_config(c, z_table=Z_TABLE, cutoff=R_MAX) for c in configs]
    loader = torch_geometric.dataloader.DataLoader(dataset, batch_size=6)
    e0s = np.array([[-1.0, -3.0, -5.0]])
    scale, shift = compute_excited_scaling(loader, e0s[0], "rms_forces_scaling", 3)
    forces = np.concatenate([c.properties["forces"] for c in configs])
    np.testing.assert_allclose(scale, np.sqrt(np.mean(forces**2)), rtol=1e-10)
    e0 = {1: -1.0, 6: -3.0, 8: -5.0}
    inter = np.array(
        [
            (c.properties["energy"] - sum(e0[z] for z in c.atomic_numbers)) / len(c.atomic_numbers)
            for c in configs
        ]
    )
    np.testing.assert_allclose(shift, inter.mean(axis=0), rtol=1e-10)
    assert compute_excited_scaling(loader, e0s[0], "no_scaling", 3) == (1.0, [0.0] * 3)


def _foundation(pair_repulsion=False):
    torch.manual_seed(4)
    return modules.ScaleShiftMACE(
        **model_config(),
        atomic_inter_scale=1.3,
        atomic_inter_shift=0.2,
        pair_repulsion=pair_repulsion,
    )


@pytest.mark.parametrize("nacs", [False, True])
def test_foundation_transfer_reproduces_ground_state(molecules, nacs):
    foundation = _foundation()
    model = build_excited(nacs=nacs, socs=False, dipoles=False)
    load_foundations_excited(model, foundation, Z_TABLE, load_readout=True, max_L=1)
    batch = atoms_to_batch(molecules, None)
    out = model(batch.to_dict())
    ref = foundation(batch.clone().to_dict())
    for state in range(3):
        torch.testing.assert_close(out["energy"][:, state], ref["energy"])
        torch.testing.assert_close(out["forces"][:, state], ref["forces"])


def test_foundation_transfer_without_readouts_keeps_backbone(molecules):
    foundation = _foundation()
    model = build_excited(nacs=True, socs=False, dipoles=False)
    load_foundations_excited(model, foundation, Z_TABLE, load_readout=False, max_L=1)
    torch.testing.assert_close(
        model.interactions[0].linear_up.weight, foundation.interactions[0].linear_up.weight
    )
    # the l=1 block of the last product stays trainable and freshly initialised
    assert model.products[-1].linear.weight.numel() > foundation.products[-1].linear.weight.numel()


def test_core_repulsion_is_not_scaled(molecules):
    """ScaleShiftMACE multiplies its ZBL energy by the scale; ExcitedMACE adds the
    ZBL energy in eV, so a transferred model differs by (scale - 1) * E_ZBL."""
    foundation = _foundation(pair_repulsion=True)
    model = build_excited(nacs=False, socs=False, dipoles=False, pair_repulsion=True)
    load_foundations_excited(model, foundation, Z_TABLE, load_readout=True, max_L=1)
    batch = atoms_to_batch(molecules, None)
    out = model(batch.to_dict())
    ref = foundation(batch.clone().to_dict())
    scale = float(foundation.scale_shift.scale)
    expected = ref["energy"] - (scale - 1.0) * out["pair_energy"]
    torch.testing.assert_close(out["energy"][:, 0], expected)
