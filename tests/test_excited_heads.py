"""Heads of excited-state models: per-head specs and padding, masked losses and
metrics, head selection in the calculators, backbone head maps and statistics."""

import argparse
import copy

import ase
import numpy as np
import pytest
import torch
from conftest import (  # pylint: disable=import-error
    KEYSPEC,
    R_MAX,
    Z_TABLE,
    atoms_to_batch,
    full_spec,
    model_config,
)
from e3nn import o3
from excited_data import make_excited_configs  # pylint: disable=import-error

from mace import data, modules
from mace.calculators import ExcitedMACECalculator
from mace.calculators.excited import (
    ExcitedModelLayout,
    HeadView,
    evaluate_excited_configs,
)
from mace.data.excited import (
    ExcitedStateSpec,
    PaddedLayout,
    excited_specs_for_heads,
    ground_state_spec,
    pad_excited_configs,
    spec_layout,
    standardise_excited_configs,
)
from mace.modules import WeightedExcitedStatesLoss
from mace.modules.excited import ExcitedMACE, excited_head_layout
from mace.tools import build_default_arg_parser, torch_geometric
from mace.tools.model_script_utils import (
    _foundation_as_backbone,
    excited_head_statistics,
)
from mace.tools.train import MACELoss

# ------------------------------------------------------------------------------------
# specs and padding
# ------------------------------------------------------------------------------------


def _args(*argv):
    return build_default_arg_parser().parse_args(
        ["--name=t", "--model=ExcitedMACE", "--train_file=x.xyz", *argv]
    )


def test_specs_for_heads():
    args = _args("--n_energies=3", "--compute_nacs", "--compute_dipoles")
    heads = {
        "pt_head": {},
        "excited": {},
        "ground": {"ground_state": True},
        "one": {"n_states": 1},
        "triplets": {"states": [0, 0, 3], "compute_dipoles": False},
        "big": {"n_energies": 4, "compute_socs": True},
    }
    specs = excited_specs_for_heads(args, heads)
    for name in ("pt_head", "ground", "one"):
        assert specs[name] == ground_state_spec()
        assert specs[name].describe() == "ground state (energy and forces)"
    assert specs["excited"].n_states == 3 and specs["excited"].n_nacs == 3
    assert specs["excited"].n_dipoles == 6
    assert specs["triplets"].states == [0, 0, 3]
    assert specs["triplets"].n_nacs == 3 and specs["triplets"].n_dipoles == 0
    assert specs["big"].n_states == 4 and specs["big"].n_nacs == 6
    assert specs["big"].n_socs == 6
    padded = PaddedLayout.from_specs(specs.values())
    assert (padded.n_states, padded.n_nacs, padded.n_socs, padded.n_dipoles) == (
        4,
        6,
        6,
        10,
    )
    layout = spec_layout(specs["excited"])
    assert layout["n_states"] == 3 and layout["n_permanent_dipoles"] == 3


def test_padding_ground_frames_to_an_excited_layout():
    atoms_list = make_excited_configs(
        2, 1, seed=1, with_nacs=False, with_socs=False, with_dipoles=False
    )
    configs = [data.config_from_atoms(a, key_specification=KEYSPEC) for a in atoms_list]
    standardise_excited_configs(configs, ground_state_spec())
    energy0 = float(np.reshape(configs[0].properties["energy"], -1)[0])
    padded = PaddedLayout.from_specs([ground_state_spec(), full_spec(3)])
    pad_excited_configs(configs, ground_state_spec(), padded)
    props, weights = configs[0].properties, configs[0].property_weights
    assert props["energy"].shape == (3,) and props["energy"][0] == energy0
    np.testing.assert_array_equal(props["energy"][1:], 0.0)
    assert props["forces"].shape == (4, 3, 3)
    assert props["nacs"].shape == (4, 3, 3) and weights["nacs"] == 0.0
    assert props["socs"].shape == (3,) and weights["socs"] == 0.0
    assert props["dipoles"].shape == (6, 3) and weights["dipoles"] == 0.0


# ------------------------------------------------------------------------------------
# a model with a ground-state and an excited head
# ------------------------------------------------------------------------------------


def _mixed_model(seed=5):
    torch.manual_seed(seed)
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
    model = ExcitedMACE(
        backbone=backbone,
        head_layouts=[
            excited_head_layout(1),
            excited_head_layout(
                3, compute_nacs=True, compute_socs=True, compute_dipoles=True
            ),
        ],
        MLP_irreps=config["MLP_irreps"],
        gate=torch.nn.functional.silu,
    )
    with torch.no_grad():
        for readout in model.readouts:
            last = getattr(readout, "linear_2", None) or readout.linear
            last.weight.normal_(0.0, 0.3)
    return model


def _mixed_batch(n=4):
    batch = atoms_to_batch(make_excited_configs(n, 3, seed=2), full_spec())
    batch.head = torch.tensor([0, 1] * (n // 2))
    return batch


def _loss(model):
    return WeightedExcitedStatesLoss(
        1.0, 10.0, 10.0, 10.0, 1.0, n_permanent_dipoles=3, masks=model.mask_tables()
    )


def _perturb_padding(out, batch):
    """Change every padded (meaningless) entry of a ground-state frame's outputs."""
    out = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}
    ground = batch.head == 0
    ground_atoms = ground[batch.batch]
    out["energy"][ground, 1:] += 5.0
    out["forces"][ground_atoms, 1:] += 1.0
    out["nacs"][ground_atoms] += 1.0
    out["socs"][ground] += 1.0
    out["dipoles"][ground] += 1.0
    return out


def test_padded_entries_do_not_count_in_the_loss():
    model = _mixed_model()
    batch = _mixed_batch()
    out = model(batch.to_dict())
    loss_fn = _loss(model)
    loss = loss_fn(batch, out)
    torch.testing.assert_close(loss_fn(batch, _perturb_padding(out, batch)), loss)
    # a real entry does count
    changed = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in out.items()
    }
    changed["energy"][0, 0] += 1.0
    assert abs(float(loss_fn(batch, changed)) - float(loss)) > 1e-3


def test_padded_entries_do_not_count_in_the_metrics():
    model = _mixed_model()
    batch = _mixed_batch()
    out = model(batch.to_dict())
    results = []
    for output in (out, _perturb_padding(out, batch)):
        metric = MACELoss(loss_fn=_loss(model))
        metric.update(batch, output)
        _, aux = metric.compute()
        results.append(aux)
    for key in ("mae_e", "rmse_f", "mae_nacs", "rmse_socs", "mae_dipoles"):
        np.testing.assert_allclose(results[1][key], results[0][key], rtol=1e-12)


def test_head_view_and_calculator_head_selection(tmp_path):
    model = _mixed_model()
    path = tmp_path / "mixed.model"
    torch.save(model, path)
    default = HeadView.from_model(model)
    assert default.name == "ground" and default.n_energies == 1  # first non-pt head
    excited = HeadView.from_model(model, "excited")
    assert excited.index == 1 and excited.nac_num == 3 and excited.dipole_num == 6
    assert HeadView.from_model(model, 1) == excited
    with pytest.raises(ValueError, match="not in the model's heads"):
        HeadView.from_model(model, "nope")
    selected = excited.select(
        {
            "energy": np.zeros((1, 3)),
            "nacs": np.zeros((4, 3, 3)),
            "socs": np.zeros((1, 3)),
        }
    )
    assert selected["energy"].shape == (1, 3)
    assert set(HeadView.from_model(model, "ground").select(selected)) == {"energy"}
    layout = ExcitedModelLayout.from_model(model, "excited")
    assert layout.nac_pairs == [(0, 1), (0, 2), (1, 2)] and layout.n_permanent == 3

    atoms = make_excited_configs(1, 3, seed=4)[0]
    calc = ExcitedMACECalculator(path, head="excited")
    results = calc.predict(atoms)
    batch = atoms_to_batch([atoms], None)
    batch.head = torch.tensor([1])
    ref = model(batch.to_dict())
    np.testing.assert_allclose(results["energies"], ref["energy"][0].detach().numpy())
    assert results["nacs"].shape == (4, 3, 3) and results["dipoles"].shape == (6, 3)
    ground = ExcitedMACECalculator(path, head="ground").predict(atoms)
    assert ground["energies"].shape == (1,) and "nacs" not in ground
    batch.head = torch.tensor([0])
    np.testing.assert_allclose(
        ground["energies"], model(batch.to_dict())["energy"][0, :1].detach().numpy()
    )
    # committee of two copies, by head
    committee = ExcitedMACECalculator(
        None, models=[model, copy.deepcopy(model)], head=1
    )
    np.testing.assert_allclose(
        committee.predict(atoms)["energies"], results["energies"]
    )
    # evaluation of many frames by head
    frames = make_excited_configs(3, 3, seed=6)
    evaluate_excited_configs(model, frames, head="excited")
    assert frames[0].info["MACE_energies"].shape == (3,)
    assert frames[0].info["MACE_nacs"].shape == (4, 3, 3)
    frames = make_excited_configs(3, 3, seed=6)
    evaluate_excited_configs(model, frames, head="ground")
    assert frames[0].info["MACE_energies"].shape == (1,)
    assert "MACE_nacs" not in frames[0].info


# ------------------------------------------------------------------------------------
# backbone head maps (a foundation model used as it is)
# ------------------------------------------------------------------------------------


def test_several_heads_on_one_backbone_head():
    torch.manual_seed(3)
    foundation = modules.ScaleShiftMACE(
        **model_config(), atomic_inter_scale=[1.2], atomic_inter_shift=[0.1]
    )
    model = ExcitedMACE(
        backbone=copy.deepcopy(foundation),
        heads=["a", "b"],
        head_layouts=[excited_head_layout(2), excited_head_layout(1)],
        MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu,
    )
    assert model.heads == ["a", "b"]
    assert model.backbone.head_map.tolist() == [0, 0]
    batch = atoms_to_batch(make_excited_configs(2, 3, seed=1), None)
    batch.head = torch.tensor([0, 1])
    out = model(batch.to_dict())
    ref = foundation(atoms_to_batch(make_excited_configs(2, 3, seed=1), None).to_dict())
    torch.testing.assert_close(out["ground_energy"], ref["energy"])
    torch.testing.assert_close(out["energy"][0, :2], ref["energy"][0].expand(2))
    from e3nn.util import jit  # pylint: disable=import-outside-toplevel

    scripted = jit.compile(model)
    torch.testing.assert_close(scripted(batch.to_dict())["energy"], out["energy"])
    with pytest.raises(ValueError, match="head_map"):
        ExcitedMACE(
            backbone=copy.deepcopy(foundation),
            heads=["a"],
            backbone_heads=[3],
            head_layouts=[excited_head_layout(2)],
            MLP_irreps=o3.Irreps("8x0e"),
        )


def test_foundation_as_backbone_checks_elements():
    foundation = modules.ScaleShiftMACE(
        **model_config(), atomic_inter_scale=[1.0], atomic_inter_shift=[0.0]
    )
    args = argparse.Namespace(foundation_head=None)
    backbone, head_map, _ = _foundation_as_backbone(
        args, foundation, ["x", "y"], Z_TABLE
    )
    assert head_map == [0, 0] and backbone is not foundation
    assert args.r_max == pytest.approx(R_MAX)
    from mace.tools import AtomicNumberTable  # pylint: disable=import-outside-toplevel

    with pytest.raises(ValueError, match="foundation_model_elements"):
        _foundation_as_backbone(args, foundation, ["x"], AtomicNumberTable([1, 8]))
    with pytest.raises(ValueError, match="foundation_head"):
        _foundation_as_backbone(
            argparse.Namespace(foundation_head="missing"), foundation, ["x"], Z_TABLE
        )


# ------------------------------------------------------------------------------------
# statistics per head
# ------------------------------------------------------------------------------------


def _loader(n_ground=6, n_excited=8):
    ground = make_excited_configs(
        n_ground, 1, seed=3, with_nacs=False, with_socs=False, with_dipoles=False
    )
    excited = make_excited_configs(n_excited, 3, seed=4)
    configs = []
    padded = PaddedLayout.from_specs([ground_state_spec(), full_spec(3)])
    for frames, spec, head in (
        (ground, ground_state_spec(), "g"),
        (excited, full_spec(3), "e"),
    ):
        cs = [
            data.config_from_atoms(a, key_specification=KEYSPEC, head_name=head)
            for a in frames
        ]
        standardise_excited_configs(cs, spec)
        pad_excited_configs(cs, spec, padded)
        configs += cs
    dataset = [
        data.AtomicData.from_config(c, z_table=Z_TABLE, cutoff=R_MAX, heads=["g", "e"])
        for c in configs
    ]
    return (
        torch_geometric.dataloader.DataLoader(dataset, batch_size=5, shuffle=True),
        configs,
    )


@pytest.mark.parametrize("scaling", ["rms_forces_scaling", "std_scaling", "no_scaling"])
def test_head_statistics(scaling):
    loader, configs = _loader()
    e0 = np.array([-1.0, -3.0, -5.0])
    specs = {"g": ground_state_spec(), "e": full_spec(3)}
    stats = excited_head_statistics(loader, e0, ["g", "e"], specs, scaling)
    e0_of = dict(zip(Z_TABLE.zs, e0))
    for h, (name, n_states) in enumerate((("g", 1), ("e", 3))):
        own = [c for c in configs if c.head == name]
        inter = np.array(
            [
                (
                    np.reshape(c.properties["energy"], -1)[:n_states]
                    - sum(e0_of[z] for z in c.atomic_numbers)
                )
                / len(c.atomic_numbers)
                for c in own
            ]
        )
        forces = np.concatenate(
            [
                np.reshape(c.properties["forces"], (len(c.atomic_numbers), -1, 3))[
                    :, :n_states
                ]
                for c in own
            ]
        )
        scale, shift = stats[h]
        if scaling == "no_scaling":
            assert (scale, shift) == (1.0, [0.0] * n_states)
            continue
        np.testing.assert_allclose(shift, inter.mean(axis=0), rtol=1e-10)
        expected = (
            np.sqrt(np.mean(forces**2))
            if scaling == "rms_forces_scaling"
            else np.std(inter, ddof=1)
        )
        np.testing.assert_allclose(scale, expected, rtol=1e-10)


def test_state_charges_reject_periodic_cells():
    from conftest import build_excited  # pylint: disable=import-outside-toplevel

    model = build_excited(long_range_mode="state")
    atoms = make_excited_configs(1, 3, seed=1)[0]
    atoms.cell = np.eye(3) * 10.0
    atoms.pbc = True
    with pytest.raises(RuntimeError, match="molecules only"):
        model(atoms_to_batch([atoms], None).to_dict())


def test_ground_state_spec_layout_is_ground():
    spec = ExcitedStateSpec(n_states=1)
    assert spec_layout(spec) == {
        "n_states": 1,
        "states": [1],
        "nac_num": 0,
        "soc_num": 0,
        "soc_vector_num": 0,
        "dipole_num": 0,
        "n_permanent_dipoles": 0,
        "osc_num": 0,
    }
    assert isinstance(ase.Atoms, type)
