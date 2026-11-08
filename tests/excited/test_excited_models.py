"""ExcitedMACE / AutoencoderExcitedMACE: symmetries, forces, core repulsion, TorchScript."""

import copy

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
    coulomb_energies,
    compute_excited_scaling,
    compute_multistate_forces,
    excited_head_layout,
    fit_excited_shifts,
)
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
    assert out["ground_energy"].shape == (4,)


def test_energy_only_single_state_model(molecules):
    model = build_excited(n_states=1, nacs=False, socs=False, dipoles=False)
    batch = atoms_to_batch(molecules, None)
    out = model(batch.to_dict())
    assert out["energy"].shape == (4, 1)
    assert out["nacs"] is None and out["socs"] is None and out["dipoles"] is None


def test_adapter_gives_vectors_to_scalar_backbones(molecules):
    """A backbone without l=1 features (e.g. a scalar foundation model): the adapter
    layer provides the equivariant features of the couplings."""
    model = ExcitedMACE(
        **model_config(hidden_irreps=o3.Irreps("8x0e")),
        n_energies=3,
        compute_nacs=True,
        compute_dipoles=True,
    )
    assert o3.Irreps(str(model.backbone.model.products[-1].linear.irreps_out)).lmax == 0
    out = model(atoms_to_batch(molecules, None).to_dict())
    assert out["nacs"].abs().max() > 0 and out["dipoles"].abs().max() > 0
    rotation = o3.rand_matrix().numpy()
    out_r = model(atoms_to_batch(_rotated(molecules, rotation), None).to_dict())
    torch.testing.assert_close(out_r["nacs"], out["nacs"] @ torch.tensor(rotation).T)


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
        {
            k: v
            for k, v in with_zbl.state_dict().items()
            if "pair_repulsion_fn" not in k
        }
    )
    batch = atoms_to_batch(molecules, None)
    out_zbl = with_zbl(batch.to_dict())
    out = without(batch.clone().to_dict())
    difference = (out_zbl["energy"] - out["energy"]).detach()
    # the repulsion is part of the backbone: the same for every state ...
    torch.testing.assert_close(difference, difference[:, :1].expand_as(difference))
    torch.testing.assert_close(
        difference[:, 0], (out_zbl["ground_energy"] - out["ground_energy"]).detach()
    )
    assert torch.all(difference[:, 0].abs() > 0)
    # ... and the couplings do not see it
    torch.testing.assert_close(out_zbl["nacs"], out["nacs"])


def test_core_repulsion_dominates_at_short_range():
    model = build_excited(nacs=False, socs=False, dipoles=False, pair_repulsion=True)
    energies = []
    for distance in (0.3, 0.6, 1.0):
        atoms = ase.Atoms("CO", positions=[[0, 0, 0], [0, 0, distance]])
        energies.append(model(atoms_to_batch([atoms], None).to_dict())["energy"].detach())
    assert torch.all(energies[0] > energies[1]) and torch.all(energies[1] > energies[2])
    assert torch.all(energies[0] - energies[2] > 10.0)  # eV


def test_states_start_on_the_backbone_surface(molecules):
    """Energy readouts start at zero: E_s = E_backbone + n_atoms * shift_s."""
    model = build_excited(
        nacs=False, socs=False, dipoles=False, perturb=False, atomic_inter_shift=[0.1, 0.2, 0.6]
    )
    batch = atoms_to_batch(molecules, None)
    out = model(batch.to_dict())
    num_atoms = (batch.ptr[1:] - batch.ptr[:-1]).double()
    # the backbone gets the mean shift, the states the differences
    torch.testing.assert_close(model.delta_shift[0], torch.tensor([-0.2, -0.1, 0.3]))
    expected = out["ground_energy"][:, None] + num_atoms[:, None] * model.delta_shift[0]
    torch.testing.assert_close(out["energy"], expected)
    torch.testing.assert_close(
        out["forces"], out["forces"][:, :1].expand_as(out["forces"])
    )


def test_delta_scale_and_shift(molecules):
    model = build_excited(nacs=False, socs=False, dipoles=False)
    batch = atoms_to_batch(molecules, None)
    out = model(batch.to_dict())
    delta = (out["energy"] - out["ground_energy"][:, None]).detach()
    model.delta_scale.fill_(2.0)
    model.delta_shift[0] = torch.tensor([0.1, 0.2, 0.3])
    out2 = model(batch.clone().to_dict())
    num_atoms = (batch.ptr[1:] - batch.ptr[:-1]).double()
    expected = 2.0 * delta + num_atoms[:, None] * torch.tensor([0.1, 0.2, 0.3])
    torch.testing.assert_close(
        (out2["energy"] - out2["ground_energy"][:, None]).detach(), expected
    )


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


def test_isolated_atom_has_no_forces():
    model = build_excited(nacs=False, socs=False, dipoles=False, atomic_inter_shift=0.0)
    atoms = ase.Atoms("O", positions=[[0, 0, 0]])
    out = model(atoms_to_batch([atoms], None).to_dict())
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
    model = build_excited(autoencoder=True, atomic_inter_shift=[0.1, 0.2, 0.3])
    model.delta_scale.fill_(1.5)
    batch = atoms_to_batch(molecules, full_spec())
    out = model(batch.to_dict())
    # decoded_energy = baseline + scale * decoder(encoder((E_ref - baseline) / scale))
    num_atoms = (batch.ptr[1:] - batch.ptr[:-1]).double()
    shift = model.shift_average[0] @ model.delta_shift[0]  # one shift per multiplicity
    baseline = out["ground_energy"][:, None] + num_atoms[:, None] * shift
    encoded = model.perm_encoders[0]((batch.energy - baseline) / 1.5)
    torch.testing.assert_close(out["encoded_energy"], encoded)
    torch.testing.assert_close(
        out["decoded_energy"], baseline + 1.5 * model.perm_decoders[0](encoded)
    )


def test_autoencoder_is_permutation_invariant_over_states():
    torch.manual_seed(3)
    model = build_excited(autoencoder=True, nacs=False, socs=False, dipoles=False)
    energies = torch.randn(5, 3)
    latent = model.perm_encoders[0](energies)
    torch.testing.assert_close(model.perm_encoders[0](energies[:, [2, 0, 1]]), latent)


def test_decoder_handles_one_state():
    decoder = PermutationInvariantDecoder(latent_dim=4, n_energies=1)
    assert decoder(torch.randn(3, 4)).shape == (3, 1)


def test_invalid_configurations_are_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        ExcitedMACE(**model_config(), n_energies=0)
    with pytest.raises(ValueError, match="long_range_mode"):
        ExcitedMACE(**model_config(), n_energies=2, long_range_mode="ewald")
    with pytest.raises(ValueError, match="coupling_parity"):
        ExcitedMACE(**model_config(), n_energies=2, coupling_parity="odd")
    # one state has no NAC pairs, unless a custom count is given
    with pytest.raises(ValueError, match="n_energies >= 2"):
        ExcitedMACE(**model_config(), n_energies=1, compute_nacs=True)
    assert ExcitedMACE(**model_config(), n_energies=1, compute_nacs=True, nac_num=2).nac_num == 2


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


# ------------------------------------------------------------------------------------
# foundation backbones
# ------------------------------------------------------------------------------------


def _foundation(pair_repulsion=False, heads=None, **overrides):
    torch.manual_seed(4)
    if heads is not None:
        e0 = np.array([-1.0, -3.0, -5.0])
        overrides["atomic_energies"] = np.stack([e0 + 0.1 * h for h in range(len(heads))])
    return modules.ScaleShiftMACE(
        **model_config(**overrides),
        atomic_inter_scale=[1.3] * len(heads or [0]),
        atomic_inter_shift=[0.2] * len(heads or [0]),
        pair_repulsion=pair_repulsion,
        heads=heads,
    )


def _wrap(backbone, layouts, autoencoder=False, **kwargs):
    cls = AutoencoderExcitedMACE if autoencoder else ExcitedMACE
    extra = {"num_permutational_invariant": 6} if autoencoder else {}
    return cls(
        backbone=backbone,
        head_layouts=layouts,
        MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu,
        adapter_irreps=o3.Irreps("8x0e + 8x1o"),
        **extra,
        **kwargs,
    )


@pytest.mark.parametrize("hidden", ["8x0e + 8x1o", "8x0e"])
@pytest.mark.parametrize("pair_repulsion", [False, True])
def test_foundation_backbone_reproduces_ground_state(molecules, hidden, pair_repulsion):
    foundation = _foundation(pair_repulsion, hidden_irreps=o3.Irreps(hidden))
    layout = excited_head_layout(3, compute_nacs=True, compute_dipoles=True)
    model = _wrap(copy.deepcopy(foundation), [layout])
    batch = atoms_to_batch(molecules, None)
    out = model(batch.to_dict())
    ref = foundation(batch.clone().to_dict())
    for state in range(3):
        torch.testing.assert_close(out["energy"][:, state], ref["energy"])
        torch.testing.assert_close(out["forces"][:, state], ref["forces"])
    assert out["nacs"].shape == (16, 3, 3)
    # the backbone's weights are the foundation's
    torch.testing.assert_close(
        model.backbone.model.interactions[0].linear_up.weight,
        foundation.interactions[0].linear_up.weight,
    )


def test_fit_excited_shifts_to_the_foundation(molecules):
    foundation = _foundation()
    layout = excited_head_layout(3)
    model = _wrap(copy.deepcopy(foundation), [layout])
    atoms_list = [a.copy() for a in molecules]
    batch = atoms_to_batch(atoms_list, None)
    ground = foundation(batch.to_dict())["energy"].detach()
    offsets = np.array([0.0, 0.5, 1.25])
    for atoms, e in zip(atoms_list, ground.numpy()):
        atoms.info["REF_energy"] = e + len(atoms) * offsets
        atoms.arrays["REF_forces"] = np.zeros((len(atoms), 3, 3))
    loader = [atoms_to_batch(atoms_list, full_spec(nacs_format="raw"))]
    fit_excited_shifts(model, loader)
    torch.testing.assert_close(model.delta_shift[0], torch.tensor(offsets))
    out = model(loader[0].clone().to_dict())
    torch.testing.assert_close(out["energy"], loader[0].energy)


@pytest.mark.parametrize("autoencoder", BOTH)
def test_mixed_ground_and_excited_heads(molecules, autoencoder):
    """One backbone, a ground-state head (one state) and an excited head (3 states
    with couplings): outputs are padded to the larger head; masks say what is real."""
    torch.manual_seed(5)
    backbone = _foundation(heads=["ground", "excited"])
    layouts = [
        excited_head_layout(1),
        excited_head_layout(3, compute_nacs=True, compute_socs=True, compute_dipoles=True),
    ]
    model = _wrap(backbone, layouts, autoencoder=autoencoder)
    assert model.heads == ["ground", "excited"] and model.head_n_states == [1, 3]
    masks = model.mask_tables()
    torch.testing.assert_close(masks["energy"], torch.tensor([[1.0, 0, 0], [1, 1, 1]]))
    torch.testing.assert_close(masks["nacs"], torch.tensor([[0.0, 0, 0], [1, 1, 1]]))
    batch = atoms_to_batch(molecules, None)
    batch.head = torch.tensor([0, 1, 0, 1])
    out = model(batch.to_dict())
    assert out["energy"].shape == (4, 3) and out["nacs"].shape == (16, 3, 3)
    node_heads = batch.head[batch.batch]
    # the ground-state head has no couplings
    assert torch.all(out["nacs"][node_heads == 0] == 0)
    assert torch.all(out["socs"][batch.head == 0] == 0)
    assert out["nacs"][node_heads == 1].abs().max() > 0
    # each frame gets its own head's backbone energy
    backbone_out = backbone(batch.clone().to_dict())
    torch.testing.assert_close(out["ground_energy"], backbone_out["energy"])
    # a frame alone gives the same results as in a mixed batch
    for g in range(4):
        alone = atoms_to_batch([molecules[g]], None)
        alone.head = batch.head[g : g + 1]
        single = model(alone.to_dict())
        n = model.head_n_states[int(batch.head[g])]
        torch.testing.assert_close(single["energy"][0, :n], out["energy"][g, :n])
    jit.compile(model)


def test_lora_on_backbone_keeps_excited_heads_trainable(molecules):
    from mace.modules.lora import inject_LoRAs

    foundation = _foundation()
    model = _wrap(foundation, [excited_head_layout(3, compute_nacs=True)])
    batch = atoms_to_batch(molecules, None)
    before = model(batch.to_dict())["energy"].detach()
    inject_LoRAs(model.backbone.model, rank=2, alpha=1.0)
    after = model(batch.clone().to_dict())["energy"].detach()
    torch.testing.assert_close(after, before)  # LoRA starts as the identity
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any("lora" in n.lower() for n in trainable)
    assert any(n.startswith("readouts") for n in trainable)
    assert any(n.startswith("nacs_readouts") for n in trainable)
    assert any(n.startswith("adapter_product") for n in trainable)
    assert not any(
        n.startswith("backbone.model.interactions") and "lora" not in n.lower()
        for n in trainable
    )


# ------------------------------------------------------------------------------------
# state-specific electrostatics
# ------------------------------------------------------------------------------------


def test_coulomb_energy_of_two_point_like_charges():
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 30.0]])
    charges = torch.tensor([[1.0], [-1.0]])
    energy = coulomb_energies(charges, positions, torch.zeros(2, dtype=torch.long), 1, 1.0)
    torch.testing.assert_close(energy, torch.tensor([[-14.399645 / 30.0]]))


@pytest.mark.parametrize("autoencoder", BOTH)
def test_state_charges(molecules, autoencoder):
    model = build_excited(autoencoder=autoencoder, long_range_mode="state")
    batch = atoms_to_batch(molecules, None)
    out = model(batch.to_dict())
    # charge readouts start at zero: no electrostatic correction
    torch.testing.assert_close(
        out["state_electrostatic_energy"], torch.zeros_like(out["energy"])
    )
    with torch.no_grad():
        for readout in model.charge_readouts:
            last = getattr(readout, "linear_2", None) or readout.linear
            last.weight.normal_(0.0, 0.5)
    out = model(batch.clone().to_dict())
    assert out["state_electrostatic_energy"].abs().max() > 1e-6
    # every state's charges sum to the backbone's total charge (zero here)
    total = torch.zeros(4, 3).index_add(0, batch.batch, out["state_charges"])
    torch.testing.assert_close(total, torch.zeros(4, 3))
    # invariant energies, forces consistent with the energy
    rotation = o3.rand_matrix().numpy()
    out_r = model(atoms_to_batch(_rotated(molecules, rotation), None).to_dict())
    torch.testing.assert_close(out_r["energy"], out["energy"])
    atoms = molecules[0]
    forces = model(atoms_to_batch([atoms], None).to_dict())["forces"].detach().numpy()
    h = 1e-5
    displaced = [atoms.copy(), atoms.copy()]
    displaced[0].positions[1, 2] += h
    displaced[1].positions[1, 2] -= h
    e = [model(atoms_to_batch([d], None).to_dict())["energy"].detach().numpy()[0] for d in displaced]
    np.testing.assert_allclose(forces[1, :, 2], -(e[0] - e[1]) / (2 * h), atol=1e-6, rtol=1e-5)


# ------------------------------------------------------------------------------------
# other backbone classes
# ------------------------------------------------------------------------------------


@pytest.mark.parametrize("autoencoder", BOTH)
def test_plain_mace_backbone(molecules, autoencoder):
    model = build_excited(autoencoder=autoencoder, backbone_cls=modules.MACE)
    assert model.backbone_class == "MACE"
    out = model(atoms_to_batch(molecules, None).to_dict())
    assert out["energy"].shape == (4, 3)
    jit.compile(model)
