"""The core MACE components, exercised through excited-state models.

Architecture options of the backbone (radial bases, distance transforms, interaction
blocks, gates, correlation orders, scalar-only features, reduced CG, embeddings of
charge and spin), and the library machinery X-MACE relies on (copying models, head
selection, E0 estimation, optimiser groups and freezing, preprocessed data)."""

import argparse
import copy
import io
import json

import numpy as np
import pytest
import torch
from conftest import (  # pylint: disable=import-error
    KEYSPEC,
    R_MAX,
    Z_TABLE,
    atoms_to_batch,
    build_excited,
    full_spec,
    model_config,
    perturb_energy_readouts,
)
from e3nn import o3
from e3nn.util import jit
from excited_data import make_excited_configs  # pylint: disable=import-error

from mace import data, modules
from mace.modules.excited import (
    AutoencoderExcitedMACE,
    ExcitedMACE,
    excited_head_layout,
)
from mace.tools.scripts_utils import get_params_options, remove_pt_head


def _rotated(atoms_list, rotation):
    out = []
    for atoms in atoms_list:
        new = atoms.copy()
        new.positions = atoms.positions @ rotation.T
        out.append(new)
    return out


def _check_symmetry_and_forces(model, molecules):
    """Invariant energies, equivariant forces and NACs, forces = -dE/dr."""
    out = model(atoms_to_batch(molecules, None).to_dict())
    rotation = o3.rand_matrix().numpy()
    out_r = model(atoms_to_batch(_rotated(molecules, rotation), None).to_dict())
    R = torch.tensor(rotation)
    torch.testing.assert_close(out_r["energy"], out["energy"])
    torch.testing.assert_close(out_r["forces"], out["forces"] @ R.T)
    if out["nacs"] is not None:
        torch.testing.assert_close(out_r["nacs"], out["nacs"] @ R.T)
    atoms = molecules[0]
    h = 1e-5
    displaced = [atoms.copy(), atoms.copy()]
    displaced[0].positions[1, 0] += h
    displaced[1].positions[1, 0] -= h
    e = [
        model(atoms_to_batch([d], None).to_dict())["energy"].detach().numpy()[0]
        for d in displaced
    ]
    forces = out["forces"].detach().numpy()[1, :, 0]
    np.testing.assert_allclose(forces, -(e[0] - e[1]) / (2 * h), atol=1e-6, rtol=1e-5)
    return out


ARCHITECTURES = {
    "bessel": dict(),
    "gaussian_agnesi": dict(radial_type="gaussian", distance_transform="Agnesi"),
    "chebyshev_soft": dict(radial_type="chebyshev", distance_transform="Soft"),
    "density_blocks": dict(
        interaction_cls=modules.interaction_classes[
            "RealAgnosticDensityResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticDensityInteractionBlock"
        ],
    ),
    "nonlinear_block": dict(
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualNonLinearInteractionBlock"
        ],
    ),
    "attention_block": dict(
        interaction_cls=modules.interaction_classes[
            "RealAgnosticAttResidualInteractionBlock"
        ],
    ),
    "tanh_gate_correlation3": dict(gate=torch.tanh, correlation=3),
    "correlation1": dict(correlation=1),
    "scalar_features": dict(hidden_irreps=o3.Irreps("8x0e")),
    "lmax2_features": dict(hidden_irreps=o3.Irreps("8x0e + 8x1o + 8x2e")),
    "reduced_cg": dict(use_reduced_cg=True),
    "agnostic_product": dict(use_agnostic_product=True),
    "last_readout_only": dict(use_last_readout_only=True),
    "no_cutoff": dict(apply_cutoff=False),
    "three_layers": dict(num_interactions=3),
}


@pytest.mark.parametrize("autoencoder", [False, True])
@pytest.mark.parametrize("name", list(ARCHITECTURES))
def test_backbone_architectures(molecules, name, autoencoder):
    torch.manual_seed(2)
    cls = AutoencoderExcitedMACE if autoencoder else ExcitedMACE
    extra = {"num_permutational_invariant": 4} if autoencoder else {}
    model = cls(
        **model_config(**ARCHITECTURES[name]),
        n_energies=3,
        compute_nacs=True,
        compute_dipoles=True,
        **extra,
    )
    perturb_energy_readouts(model)
    _check_symmetry_and_forces(model, molecules)
    # copies (fine-tuning, EMA, SWA) and serialisation
    clone = copy.deepcopy(model)
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)
    batch = atoms_to_batch(molecules, None)
    reference = model(batch.to_dict())["energy"]
    for other in (clone, loaded):
        torch.testing.assert_close(other(batch.clone().to_dict())["energy"], reference)


def test_torchscript_of_backbone_variants(molecules):
    for name in ("gaussian_agnesi", "chebyshev_soft", "density_blocks", "scalar_features"):
        torch.manual_seed(1)
        model = ExcitedMACE(
            **model_config(**ARCHITECTURES[name]),
            n_energies=3,
            compute_nacs=True,
            compute_socs=True,
            compute_dipoles=True,
        )
        perturb_energy_readouts(model)
        scripted = jit.compile(model)
        batch = atoms_to_batch(molecules, full_spec())
        torch.testing.assert_close(
            scripted(batch.clone().to_dict())["energy"], model(batch.to_dict())["energy"]
        )


def test_charge_and_spin_embeddings(molecules):
    """A backbone with embeddings of the molecular charge and spin: the energies of
    all states depend on them, the model stays invariant."""
    torch.manual_seed(3)
    specs = {
        "total_charge": {
            "type": "categorical",
            "per": "graph",
            "in_dim": 1,
            "emb_dim": 8,
            "num_classes": 5,
            "offset": 2,
        },
        "total_spin": {
            "type": "categorical",
            "per": "graph",
            "in_dim": 1,
            "emb_dim": 8,
            "num_classes": 5,
        },
    }
    backbone = modules.ScaleShiftMACE(
        **model_config(),
        atomic_inter_scale=[1.0],
        atomic_inter_shift=[0.0],
        embedding_specs=specs,
    )
    model = ExcitedMACE(
        backbone=backbone,
        head_layouts=[excited_head_layout(3, compute_nacs=True)],
        MLP_irreps=o3.Irreps("8x0e"),
    )
    perturb_energy_readouts(model)
    batch = atoms_to_batch(molecules, None)
    neutral = model(batch.to_dict())["energy"].detach()
    charged_batch = atoms_to_batch(molecules, None)
    charged_batch.total_charge = torch.ones_like(charged_batch.total_charge)
    charged = model(charged_batch.to_dict())["energy"].detach()
    assert (charged - neutral).abs().min() > 1e-6
    _check_symmetry_and_forces(model, molecules)


# ------------------------------------------------------------------------------------
# library machinery
# ------------------------------------------------------------------------------------


def test_reduced_cg_models_can_be_copied():
    """EmptyParam (reduced Clebsch-Gordan contractions) survives copy.deepcopy, which
    fine-tuning, EMA and SWA use."""
    from mace.modules.symmetric_contraction import (  # pylint: disable=import-outside-toplevel
        EmptyParam,
    )

    model = build_excited(use_reduced_cg=True)
    clone = copy.deepcopy(model)
    empty = [p for p in clone.parameters() if isinstance(p, EmptyParam)]
    assert all(not p.requires_grad for p in empty)


@pytest.mark.parametrize("autoencoder", [False, True])
def test_select_heads_reproduces_each_head(autoencoder):
    from test_excited_heads import (  # pylint: disable=import-outside-toplevel,import-error
        _mixed_batch,
    )

    torch.manual_seed(5)
    config = model_config()
    config["atomic_energies"] = np.stack(
        [config["atomic_energies"], config["atomic_energies"] + 0.1]
    )
    backbone = modules.ScaleShiftMACE(
        **config,
        atomic_inter_scale=[1.0, 1.0],
        atomic_inter_shift=[0.0, 0.0],
        heads=["ground", "excited"],
    )
    cls = AutoencoderExcitedMACE if autoencoder else ExcitedMACE
    extra = {"num_permutational_invariant": 4} if autoencoder else {}
    model = cls(
        backbone=backbone,
        head_layouts=[
            excited_head_layout(1),
            excited_head_layout(
                3, compute_nacs=True, compute_socs=True, compute_dipoles=True
            ),
        ],
        MLP_irreps=o3.Irreps("8x0e"),
        **extra,
    )
    perturb_energy_readouts(model)
    for index, name in enumerate(model.heads):
        single = model.select_heads([name])
        assert single.heads == [name]
        batch = _mixed_batch()
        batch.head = torch.zeros(4, dtype=torch.long)
        out = single(batch.to_dict())
        full = _mixed_batch()
        full.head = torch.full((4,), index)
        ref = model(full.to_dict())
        n = single.n_energies
        torch.testing.assert_close(out["energy"], ref["energy"][:, :n])
        torch.testing.assert_close(out["forces"], ref["forces"][:, :n])
        if name == "ground":
            assert out["nacs"] is None and out["dipoles"] is None
        else:
            torch.testing.assert_close(out["nacs"], ref["nacs"])
            torch.testing.assert_close(out["socs"], ref["socs"])
            torch.testing.assert_close(out["dipoles"], ref["dipoles"])
        jit.compile(single)
    assert remove_pt_head(model, "excited").heads == ["excited"]
    assert remove_pt_head(model).heads == ["ground"]  # first head that is not pt_head
    with pytest.raises(ValueError, match="not found"):
        remove_pt_head(model, "nope")


def test_estimated_e0s_with_several_states():
    """E0 estimation against a foundation model, from multi-state energies (fitted to
    the mean over the states), skipping unlabelled frames."""
    torch.manual_seed(1)
    foundation = modules.ScaleShiftMACE(
        **model_config(), atomic_inter_scale=[1.0], atomic_inter_shift=[0.0]
    )
    frames = make_excited_configs(12, 3, seed=4)
    configs = [data.config_from_atoms(a, key_specification=KEYSPEC) for a in frames]
    data.excited.standardise_excited_configs(configs, full_spec())
    # references: the foundation's energy plus a per-element offset (and a
    # state-dependent part that averages to zero over the states)
    true_shift = {1: 0.3, 6: -0.7, 8: 1.1}
    for atoms, config in zip(frames, configs):
        batch = atoms_to_batch([atoms], None)
        e_bb = float(foundation(batch.to_dict(), compute_force=False)["energy"][0])
        offset = sum(true_shift[int(z)] for z in config.atomic_numbers)
        config.properties["energy"] = np.array([e_bb + offset + d for d in (-0.2, 0.0, 0.2)])
    # a frame without energies must not count
    configs[0].property_weights["energy"] = 0.0
    configs[0].properties["energy"] = np.array([1e6, 1e6, 1e6])
    e0s = {1: -1.0, 6: -3.0, 8: -5.0}
    estimated = data.estimate_e0s_from_foundation(foundation, e0s, configs, Z_TABLE)
    # every frame is the same molecule, so only the molecule's total offset is
    # determined (the minimum-norm solution spreads it over the elements)
    molecule = [int(z) for z in configs[1].atomic_numbers]
    predicted = sum(estimated[z] - e0s[z] for z in molecule)
    expected = sum(true_shift[z] for z in molecule)
    np.testing.assert_allclose(predicted, expected, atol=1e-6)


def test_optimizer_groups_freeze_and_lr_factors_act_on_the_backbone():
    model = build_excited(autoencoder=True)
    args = argparse.Namespace(
        lr=0.01,
        weight_decay=1e-4,
        amsgrad=False,
        beta=0.9,
        freeze=6,  # embedding, interactions and products of the backbone
        lr_params_factors=json.dumps(
            {
                "embedding_lr_factor": 0.1,
                "interactions_lr_factor": 0.1,
                "products_lr_factor": 0.1,
                "readouts_lr_factor": 0.5,
            }
        ),
    )
    options = get_params_options(args, model)
    backbone = model.backbone.model
    for part in (backbone.node_embedding, backbone.interactions, backbone.products):
        assert all(not p.requires_grad for p in part.parameters())
    assert all(p.requires_grad for p in backbone.readouts.parameters())
    for name in ("adapter_product", "readouts", "nacs_readouts", "perm_decoders"):
        assert all(p.requires_grad for p in getattr(model, name).parameters()), name
    groups = {g["name"]: g for g in options["params"]}
    assert groups["readouts"]["lr"] == pytest.approx(0.005)  # backbone readouts
    assert "lr" not in groups["excited_readouts"] or groups["excited_readouts"]["lr"] == 0.01


def test_backbone_parts_are_reachable_on_the_excited_model():
    """MACE's fine-tuning and E0 code read atomic_energies_fn, interactions and
    products on a foundation; an excited-state model forwards them to its backbone
    without registering them twice."""
    model = build_excited()
    assert model.atomic_energies_fn is model.backbone.model.atomic_energies_fn
    assert model.interactions is model.backbone.model.interactions
    names = [name for name, _ in model.named_parameters()]
    assert not any(name.startswith("interactions.") for name in names)
    assert len(names) == len(set(names))


def test_hdf5_frames_get_the_head_they_are_loaded_for(tmp_path):
    """Data prepared without heads store the head "Default"; loaded for a head of a
    multi-head run, every frame must belong to that head (it used to land on the
    last head)."""
    import h5py  # pylint: disable=import-outside-toplevel

    from mace.data.excited import (  # pylint: disable=import-outside-toplevel
        PaddedLayout,
        attach_excited_padding,
        ground_state_spec,
    )
    from mace.data.utils import (  # pylint: disable=import-outside-toplevel
        save_configurations_as_HDF5,
    )

    frames = make_excited_configs(
        4, 1, seed=1, with_nacs=False, with_socs=False, with_dipoles=False
    )
    configs = [data.config_from_atoms(a, key_specification=KEYSPEC) for a in frames]
    data.excited.standardise_excited_configs(configs, ground_state_spec())
    with h5py.File(tmp_path / "ground.h5", "w") as f:
        save_configurations_as_HDF5(configs, 0, f)
    heads = ["excited", "ground", "other"]
    dataset = data.HDF5Dataset(
        tmp_path / "ground.h5", r_max=R_MAX, z_table=Z_TABLE, heads=heads, head="ground"
    )
    assert all(int(dataset[i].head) == 1 for i in range(len(dataset)))
    padded = PaddedLayout.from_specs([ground_state_spec(), full_spec(3)])
    assert attach_excited_padding(dataset, ground_state_spec(), padded, weight=2.5) == 1
    frame = dataset[0]
    assert frame.energy.shape == (1, 3) and frame.forces.shape == (4, 3, 3)
    assert float(frame.weight) == pytest.approx(2.5)
    assert float(frame.nacs_weight) == 0.0
