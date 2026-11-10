"""Oscillator strengths: learned directly as invariant, non-negative outputs, from the
ground state to the other states of its multiplicity."""

import io
import json
from pathlib import Path

import ase.io
import numpy as np
import pytest
import torch
from conftest import (  # pylint: disable=import-error
    KEYSPEC,
    atoms_to_batch,
    build_excited,
)
from e3nn.util import jit
from excited_data import make_excited_configs  # pylint: disable=import-error
from scipy.spatial.transform import Rotation

from mace import data
from mace.calculators import ExcitedMACECalculator
from mace.calculators.excited import ExcitedModelLayout, evaluate_excited_configs
from mace.data.excited import (
    ExcitedStateSpec,
    PaddedLayout,
    excited_spec_from_args,
    pad_excited_configs,
    standardise_excited_configs,
)
from mace.modules.excited import excited_head_layout, mask_tables_from_layouts
from mace.modules.excited_loss import WeightedExcitedStatesLoss
from mace.modules.spin_states import default_oscillator_strength_num
from mace.tools.arg_parser import build_default_arg_parser
from tests.helpers import run_mace_train

OSC_KEYSPEC = data.KeySpecification(
    info_keys={**KEYSPEC.info_keys, "oscillator_strengths": "REF_oscillator_strengths"},
    arrays_keys=dict(KEYSPEC.arrays_keys),
)


def _molecules(n=4, n_states=4, seed=3):
    return make_excited_configs(n, n_states, seed=seed, with_oscillator_strengths=True)


def _batch(molecules, spec=None):
    return atoms_to_batch(molecules, spec, keyspec=OSC_KEYSPEC)


def _osc_model(n_states=4, **kwargs):
    torch.manual_seed(2)
    return build_excited(n_states=n_states, compute_oscillator_strengths=True, **kwargs)


# ------------------------------------------------------------------------------------
# layout and labels
# ------------------------------------------------------------------------------------


def test_default_number_of_oscillator_strengths():
    assert default_oscillator_strength_num([4]) == 3
    assert default_oscillator_strength_num([3, 0, 2]) == 2  # S0 -> S1, S2
    assert default_oscillator_strength_num([0, 3, 0, 1]) == 2  # doublet ground state
    layout = excited_head_layout(5, states=[3, 0, 2], compute_oscillator_strengths=True)
    assert layout["osc_num"] == 2
    assert excited_head_layout(3)["osc_num"] == 0
    with pytest.raises(ValueError, match="osc_num"):
        excited_head_layout(3, compute_oscillator_strengths=True, osc_num=3)
    masks = mask_tables_from_layouts(
        [
            layout,
            excited_head_layout(1),
            {k: v for k, v in layout.items() if k != "osc_num"},
        ]
    )
    np.testing.assert_array_equal(
        masks["oscillator_strengths"], [[1, 1], [0, 0], [0, 0]]
    )


def test_spec_from_command_line():
    parser = build_default_arg_parser()
    base = ["--name=x", "--train_file=t.xyz", "--model=ExcitedMACE"]
    args = parser.parse_args(
        base + ["--states=3 0 2", "--compute_oscillator_strengths"]
    )
    spec = excited_spec_from_args(args)
    assert (
        spec.n_osc == 2 and "oscillator strengths=2 (S0 -> S1..S2)" in spec.describe()
    )
    args = parser.parse_args(
        base + ["--n_energies=4", "--compute_oscillator_strengths", "--osc_num=1"]
    )
    assert excited_spec_from_args(args).n_osc == 1
    args = parser.parse_args(
        base + ["--n_energies=1", "--compute_oscillator_strengths"]
    )
    with pytest.raises(ValueError, match="oscillator"):
        excited_spec_from_args(args)


def test_standardise_labels():
    spec = ExcitedStateSpec(n_states=4, n_osc=3)
    molecules = _molecules(3)
    molecules[1].info["REF_oscillator_strengths"] = np.concatenate(
        [[0.0], molecules[1].info["REF_oscillator_strengths"]]
    )  # with the ground state's own zero entry
    del molecules[2].info["REF_oscillator_strengths"]
    configs = [
        data.config_from_atoms(a, key_specification=OSC_KEYSPEC) for a in molecules
    ]
    counts = standardise_excited_configs(configs, spec)
    assert counts["oscillator_strengths"] == 2
    assert configs[1].properties["oscillator_strengths"].shape == (3,)
    assert configs[2].property_weights["oscillator_strengths"] == 0.0
    pad_excited_configs(
        configs, spec, PaddedLayout.from_specs([spec, ExcitedStateSpec(5, n_osc=4)])
    )
    assert configs[0].properties["oscillator_strengths"].shape == (4,)
    bad = [data.config_from_atoms(_molecules(1)[0], key_specification=OSC_KEYSPEC)]
    with pytest.raises(ValueError, match="oscillator strengths"):
        standardise_excited_configs(bad, ExcitedStateSpec(n_states=4, n_osc=2))


# ------------------------------------------------------------------------------------
# model and loss
# ------------------------------------------------------------------------------------


def test_model_output_is_invariant_and_non_negative():
    model = _osc_model()
    molecules = _molecules(3)
    out = model(_batch(molecules).to_dict())
    strengths = out["oscillator_strengths"].detach()
    assert strengths.shape == (3, 3) and torch.all(strengths >= 0)
    rotation = Rotation.from_euler("xyz", [0.3, -1.1, 2.0]).as_matrix()
    for atoms in molecules:
        atoms.positions = atoms.positions @ rotation.T
        atoms.positions = -atoms.positions  # and an inversion
    rotated = model(_batch(molecules).to_dict())["oscillator_strengths"].detach()
    torch.testing.assert_close(rotated, strengths)
    # a model without them gives none
    assert (
        build_excited(n_states=4)(_batch(molecules).to_dict())["oscillator_strengths"]
        is None
    )


def test_loss_and_masks():
    model = _osc_model()
    spec = ExcitedStateSpec(n_states=4, n_nacs=6, n_socs=6, n_dipoles=10, n_osc=3)
    batch = _batch(_molecules(4), spec)
    out = model(batch.to_dict(), training=True)
    loss_fn = WeightedExcitedStatesLoss(
        energy_weight=0.0,
        forces_weight=0.0,
        nacs_weight=0.0,
        socs_weight=0.0,
        dipoles_weight=0.0,
        oscillator_strengths_weight=2.0,
    )
    expected = 2.0 * torch.mean(
        (batch["oscillator_strengths"] - out["oscillator_strengths"]) ** 2
    )
    torch.testing.assert_close(loss_fn(batch, out), expected)
    loss_fn(batch, out).backward()
    grads = [p.grad for p in model.osc_readouts.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_scripting_heads_and_old_models():
    model = _osc_model()
    molecules = _molecules(2)
    reference = model(_batch(molecules).to_dict())["oscillator_strengths"]
    scripted = jit.compile(model)
    torch.testing.assert_close(
        scripted(_batch(molecules).to_dict())["oscillator_strengths"], reference
    )
    selected = model.select_heads(["Default"])
    torch.testing.assert_close(
        selected(_batch(molecules).to_dict())["oscillator_strengths"], reference
    )
    # a model saved before oscillator strengths existed
    old = build_excited(n_states=4)
    del old._modules["osc_readouts"]
    del old._buffers["osc_index"]
    for name in ("osc_num", "compute_oscillator_strengths"):
        del old.__dict__[name]
    old.head_layouts = [
        {k: v for k, v in layout.items() if k != "osc_num"}
        for layout in old.head_layouts
    ]
    buffer = io.BytesIO()
    torch.save(old, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)
    assert loaded.osc_num == 0 and loaded.head_layouts[0]["osc_num"] == 0
    assert loaded(_batch(molecules).to_dict())["oscillator_strengths"] is None
    jit.compile(loaded)


# ------------------------------------------------------------------------------------
# calculators
# ------------------------------------------------------------------------------------


def test_calculator_and_evaluation(tmp_path):
    model = _osc_model()
    path = tmp_path / "osc.model"
    torch.save(model, path)
    atoms = _molecules(1)[0]
    reference = (
        model(_batch([atoms]).to_dict())["oscillator_strengths"][0].detach().numpy()
    )
    calc = ExcitedMACECalculator(path)
    results = calc.predict(atoms)
    np.testing.assert_allclose(results["oscillator_strengths"], reference)
    assert calc.layout.osc_pairs == [(0, 1), (0, 2), (0, 3)]
    committee = ExcitedMACECalculator([path, path]).predict(atoms)
    np.testing.assert_allclose(committee["oscillator_strengths"], reference)
    np.testing.assert_allclose(committee["oscillator_strengths_var"], 0.0, atol=1e-20)
    evaluated = evaluate_excited_configs(model, [atoms.copy()])
    np.testing.assert_allclose(
        evaluated[0].info["MACE_oscillator_strengths"], reference
    )
    without = tmp_path / "plain.model"
    torch.save(build_excited(n_states=4), without)
    assert "oscillator_strengths" not in ExcitedMACECalculator(without).predict(atoms)
    assert (
        ExcitedModelLayout.from_model(torch.load(without, weights_only=False)).osc_pairs
        == []
    )


# ------------------------------------------------------------------------------------
# training through the command line
# ------------------------------------------------------------------------------------


def _params(tmp_path, **overrides):
    params = {
        "name": "osc",
        "train_file": str(tmp_path / "train.xyz"),
        "valid_fraction": 0.2,
        "test_file": str(tmp_path / "test.xyz"),
        "model": "ExcitedMACE",
        "n_energies": 3,
        "E0s": "average",
        "r_max": 4.0,
        "hidden_irreps": "8x0e + 8x1o",
        "MLP_irreps": "8x0e",
        "num_interactions": 2,
        "correlation": 2,
        "batch_size": 4,
        "max_num_epochs": 8,
        "lr": 0.01,
        "device": "cpu",
        "default_dtype": "float64",
        "seed": 1,
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "oscillator_strengths_weight": 100.0,
        "error_table": "EnergyNacsDipoleMAE",
        "work_dir": str(tmp_path),
    }
    params.update(overrides)
    return params


def _write(tmp_path, n_states=3):
    kwargs = dict(with_socs=False, with_oscillator_strengths=True)
    ase.io.write(
        tmp_path / "train.xyz", make_excited_configs(24, n_states, seed=1, **kwargs)
    )
    ase.io.write(
        tmp_path / "test.xyz", make_excited_configs(6, n_states, seed=2, **kwargs)
    )


def _log(tmp_path, name="osc"):
    return "\n".join(p.read_text() for p in (tmp_path / "logs").glob(f"{name}*.log"))


def _osc_errors(tmp_path, name="osc"):
    values = []
    for path in (tmp_path / "results").glob(f"{name}*_train.txt"):
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            if entry.get("mode") == "eval" and entry.get("mae_osc") is not None:
                values.append(entry["mae_osc"])
    return values


@pytest.mark.slow
@pytest.mark.parametrize("model", ["ExcitedMACE", "AutoencoderExcitedMACE"])
def test_training_learns_oscillator_strengths(tmp_path, model):
    _write(tmp_path)
    extra = ["--compute_oscillator_strengths", "--compute_nacs"]
    if model == "AutoencoderExcitedMACE":
        extra.append("--num_permutational_invariant=4")
    run_mace_train(_params(tmp_path, model=model), extra_argv=extra)
    log = _log(tmp_path)
    assert "oscillator strengths=2 (S0 -> S1..S2)" in log
    assert "MAE_f_osc" in log and "MAE f_osc (x1e-3)" in log
    assert "oscillator strength scale (rms) per transition" in log
    trained = torch.load(tmp_path / "osc.model", weights_only=False)
    train = ase.io.read(tmp_path / "train.xyz", ":")
    rms = np.sqrt(
        np.mean([a.info["REF_oscillator_strengths"] ** 2 for a in train], axis=0)
    )
    # the scale is the rms of the training frames (80 % of train.xyz)
    np.testing.assert_allclose(trained.osc_scale[0].numpy(), rms, rtol=0.3)
    errors = _osc_errors(tmp_path)
    assert errors and errors[-1] < errors[0]
    atoms = ase.io.read(tmp_path / "test.xyz", 0)
    calc = ExcitedMACECalculator(tmp_path / "osc.model")
    strengths = calc.predict(atoms)["oscillator_strengths"]
    assert strengths.shape == (2,) and np.all(strengths >= 0)
    assert (tmp_path / "osc_compiled.model").exists()


@pytest.mark.slow
def test_heads_with_and_without_oscillator_strengths_from_hdf5(tmp_path):
    """One head with oscillator strengths (HDF5), one without (xyz): padded and masked."""
    _write(tmp_path)
    ase.io.write(
        tmp_path / "other.xyz", make_excited_configs(16, 2, seed=4, with_socs=False)
    )
    script = Path(__file__).resolve().parents[1] / "mace" / "cli" / "preprocess_data.py"
    run_mace_train(
        {
            "train_file": str(tmp_path / "train.xyz"),
            "valid_fraction": 0.2,
            "h5_prefix": str(tmp_path / "h5") + "/",
            "r_max": 4.0,
            "E0s": "average",
            "n_energies": 3,
            "seed": 1,
            "num_process": 1,
            "compute_oscillator_strengths": None,
        },
        script=script,
    )
    heads = {
        "bright": {
            "train_file": str(tmp_path / "h5" / "train" / "train_0.h5"),
            "valid_file": str(tmp_path / "h5" / "val" / "val_0.h5"),
            "E0s": "{1: -1.0, 6: -3.0, 8: -5.0}",
            "compute_oscillator_strengths": True,
        },
        "other": {
            "train_file": str(tmp_path / "other.xyz"),
            "valid_fraction": 0.2,
            "E0s": "{1: -1.0, 6: -3.0, 8: -5.0}",
            "n_energies": 2,
        },
    }
    params = _params(tmp_path, max_num_epochs=3)
    for key in ("train_file", "valid_fraction", "E0s", "test_file", "n_energies"):
        params.pop(key)
    params["heads"] = repr(heads)
    params["atomic_numbers"] = "[1, 6, 8]"
    run_mace_train(params, extra_argv=["--n_energies=3"])
    trained = torch.load(tmp_path / "osc.model", weights_only=False)
    assert [layout["osc_num"] for layout in trained.head_layouts] == [2, 0]
    assert trained.mask_tables()["oscillator_strengths"].tolist() == [[1, 1], [0, 0]]
    atoms = ase.io.read(tmp_path / "test.xyz", 0)
    bright = ExcitedMACECalculator(tmp_path / "osc.model", head="bright").predict(atoms)
    assert bright["oscillator_strengths"].shape == (2,)
    other = ExcitedMACECalculator(tmp_path / "osc.model", head="other").predict(atoms)
    assert "oscillator_strengths" not in other
    assert _osc_errors(tmp_path)


@pytest.mark.slow
def test_adding_oscillator_strengths_to_a_trained_model(tmp_path):
    """A trained model without oscillator strengths as foundation of the same model
    with them: every weight, shift and scale is kept, only the oscillator-strength
    readouts are new (the energies start where the trained model left them)."""
    _write(tmp_path)
    run_mace_train(_params(tmp_path, name="first", max_num_epochs=4))
    params = _params(tmp_path, foundation_model=str(tmp_path / "first.model"))
    for key in (
        "r_max",
        "hidden_irreps",
        "MLP_irreps",
        "num_interactions",
        "correlation",
    ):
        params.pop(key)
    run_mace_train(params, extra_argv=["--compute_oscillator_strengths"])
    log = _log(tmp_path)
    assert "Adding oscillator strengths to the excited-state foundation model" in log
    trained = torch.load(tmp_path / "osc.model", weights_only=False)
    assert trained.osc_num == 2

    def initial_energy_error(name):
        for line in _log(tmp_path, name).splitlines():
            if "Initial:" in line and "MAE_E=" in line:
                return float(line.split("MAE_E=")[1].split()[0])
        raise AssertionError(f"no initial validation in the log of {name}")

    def last_energy_error(name):
        lines = [
            l
            for l in _log(tmp_path, name).splitlines()
            if "Epoch" in l and "MAE_E=" in l
        ]
        return float(lines[-1].split("MAE_E=")[1].split()[0])

    # starts from the trained energies, not from the backbone's surface
    assert initial_energy_error("osc") < 1.5 * last_energy_error("first") + 1.0


@pytest.mark.slow
def test_oscillator_strengths_only_data(tmp_path):
    """Frames with oscillator strengths only (no energies or forces), stored under
    another key: trains without computing forces."""
    frames = []
    for atoms in make_excited_configs(20, 4, seed=1, with_oscillator_strengths=True):
        f_only = ase.Atoms(atoms.get_chemical_symbols(), positions=atoms.positions)
        f_only.info["REF_f"] = np.atleast_2d(atoms.info["REF_oscillator_strengths"])
        frames.append(f_only)
    ase.io.write(tmp_path / "train.xyz", frames)
    params = _params(
        tmp_path,
        n_energies=4,
        oscillator_strengths_key="REF_f",
        energy_key="REF_none",
        forces_key="REF_none",
        E0s="{1: 0.0, 6: 0.0, 8: 0.0}",
        energy_weight=0.0,
        forces_weight=0.0,
        max_num_epochs=4,
    )
    params.pop("test_file")
    run_mace_train(params, extra_argv=["--compute_oscillator_strengths"])
    log = _log(tmp_path)
    assert "forces are not computed in training" in log
    assert "oscillator strengths=3 (S0 -> S1..S3)" in log
    errors = _osc_errors(tmp_path)
    assert errors and np.all(np.isfinite(errors))
    strengths = ExcitedMACECalculator(tmp_path / "osc.model").predict(frames[0])
    assert strengths["oscillator_strengths"].shape == (3,)


@pytest.mark.slow
def test_train_only_oscillator_strengths(tmp_path):
    """--train_only_oscillator_strengths: only the new readouts change, from data with
    oscillator strengths only; every other weight is that of the trained model."""
    _write(tmp_path)
    run_mace_train(_params(tmp_path, name="first", max_num_epochs=2))
    frames = []
    for atoms in ase.io.read(tmp_path / "train.xyz", ":"):
        f_only = ase.Atoms(atoms.get_chemical_symbols(), positions=atoms.positions)
        f_only.info["REF_oscillator_strengths"] = atoms.info["REF_oscillator_strengths"]
        frames.append(f_only)
    ase.io.write(tmp_path / "f_only.xyz", frames)
    params = _params(
        tmp_path,
        train_file=str(tmp_path / "f_only.xyz"),
        foundation_model=str(tmp_path / "first.model"),
        E0s="{1: 0.0, 6: 0.0, 8: 0.0}",
        energy_weight=0.0,
        forces_weight=0.0,
        max_num_epochs=3,
    )
    for key in (
        "r_max",
        "hidden_irreps",
        "MLP_irreps",
        "num_interactions",
        "correlation",
    ):
        params.pop(key)
    run_mace_train(
        params,
        extra_argv=[
            "--compute_oscillator_strengths",
            "--train_only_oscillator_strengths",
        ],
    )
    assert "Training only the oscillator-strength readouts" in _log(tmp_path)
    first = torch.load(tmp_path / "first.model", weights_only=False).state_dict()
    trained = torch.load(tmp_path / "osc.model", weights_only=False).state_dict()
    changed = [
        key
        for key, value in first.items()
        if key in trained
        and not key.startswith("osc_")
        and not torch.equal(value, trained[key])
    ]
    assert changed == []
    osc_first = torch.load(tmp_path / "osc.model", weights_only=False)
    assert osc_first.osc_num == 2 and _osc_errors(tmp_path)


@pytest.mark.slow
def test_distributed_oscillator_strengths_only(tmp_path):
    """Data-parallel training of oscillator strengths only: the energy readouts get no
    gradients, which DDP must allow (find_unused_parameters)."""
    from test_excited_train import _torchrun  # pylint: disable=import-error

    frames = []
    for atoms in make_excited_configs(24, 4, seed=1, with_oscillator_strengths=True):
        f_only = ase.Atoms(atoms.get_chemical_symbols(), positions=atoms.positions)
        f_only.info["REF_oscillator_strengths"] = atoms.info["REF_oscillator_strengths"]
        frames.append(f_only)
    ase.io.write(tmp_path / "train.xyz", frames)
    params = _params(
        tmp_path,
        n_energies=4,
        E0s="{1: 0.0, 6: 0.0, 8: 0.0}",
        energy_weight=0.0,
        forces_weight=0.0,
        max_num_epochs=3,
    )
    params.pop("test_file")
    _torchrun(params, ["--compute_oscillator_strengths"])
    log = _log(tmp_path)
    assert "Processes: 2" in log and "forces are not computed in training" in log
    assert _osc_errors(tmp_path)
