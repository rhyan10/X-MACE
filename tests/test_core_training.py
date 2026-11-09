"""The core MACE training, fine-tuning and command-line features, with excited-state
models: every option of mace_run_train that is not about the model itself (EMA,
Stage Two, restarts, checkpoints, optimisers and schedulers, freezing, learning-rate
factors, config files, several files per head, validation and test files, config
type and head weights, E0 options, data-loader workers, scaling, error tables,
plotting, foundation options, multi-head fine-tuning with replay, LoRA), the other
command-line tools (mace_select_head, mace_convert_device, mace_plot_train,
mace_eval_configs) and the excited-state foundation pipeline (foundation/)."""

import os
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
import torch
from test_excited_train import (  # pylint: disable=import-error
    GOLDEN_FOUNDATION,
    _ground_state_data,
    _log,
    _params,
    _valid_losses,
    _write_data,
)

from mace.calculators import ExcitedMACECalculator
from tests.helpers import run_mace_train

pytestmark = pytest.mark.slow
REPO = Path(__file__).resolve().parents[1]
ARCH = ("hidden_irreps", "MLP_irreps", "num_interactions", "correlation", "r_max")


def _foundation(params):
    params = {k: v for k, v in params.items() if k not in ARCH}
    params["foundation_model"] = str(GOLDEN_FOUNDATION)
    return params


def _heads(tmp_path, **excited):
    return {
        "excited": {"train_file": str(tmp_path / "train.xyz"), "valid_fraction": 0.2, **excited},
        "ground": {
            "train_file": str(tmp_path / "ground.xyz"),
            "valid_fraction": 0.2,
            "ground_state": True,
        },
    }


def _run(tmp_path, params, extra=(), name="xmace"):
    run_mace_train(params, extra_argv=list(extra))
    return _log(tmp_path, name)


@pytest.fixture(name="data_dir")
def fixture_data_dir(tmp_path):
    _write_data(tmp_path, with_socs=False)
    _ground_state_data(tmp_path / "ground.xyz", 16, seed=5)
    return tmp_path


EXCITED = ["--compute_nacs", "--compute_dipoles"]


# ------------------------------------------------------------------------------------
# training options
# ------------------------------------------------------------------------------------


def test_ema_and_stage_two(data_dir):
    params = _params(data_dir, max_num_epochs=4, ema=None, ema_decay=0.9, swa=None, start_swa=2)
    log = _run(data_dir, params, EXCITED)
    assert "Stage Two" in log or "stage two" in log.lower()
    assert any(data_dir.glob("*stagetwo*.model"))


def test_restart_latest_continues(data_dir):
    params = _params(data_dir, max_num_epochs=2, restart_latest=None)
    run_mace_train(params, extra_argv=EXCITED)
    params["max_num_epochs"] = 4
    log = _run(data_dir, params, EXCITED)
    assert "Loading checkpoint" in log and "Epoch 3" in log


def test_checkpoints_are_kept(data_dir):
    params = _params(data_dir, max_num_epochs=3, save_all_checkpoints=None, keep_checkpoints=None)
    _run(data_dir, params, EXCITED)
    assert len(list((data_dir / "checkpoints").glob("*.pt"))) >= 2


@pytest.mark.parametrize(
    "options",
    [
        {"lbfgs": None, "max_num_epochs": 2},
        {"optimizer": "adamw", "scheduler": "ExponentialLR", "weight_decay": 1e-4, "clip_grad": 1.0},
        {"optimizer": "schedulefree"},
    ],
    ids=["lbfgs", "adamw_exponential_clip", "schedulefree"],
)
def test_optimisers_and_schedulers(data_dir, options):
    if options.get("optimizer") == "schedulefree":
        pytest.importorskip("schedulefree")
    params = _params(data_dir, **{"max_num_epochs": 3, **options})
    _run(data_dir, params, EXCITED)
    assert np.all(np.isfinite(_valid_losses(data_dir)))


def test_freeze_and_lr_factors_on_a_foundation(data_dir):
    params = _foundation(
        _params(
            data_dir,
            max_num_epochs=2,
            freeze=6,
            lr_params_factors='{"embedding_lr_factor": 0.1, "interactions_lr_factor": 0.1, '
            '"products_lr_factor": 0.1, "readouts_lr_factor": 0.5}',
        )
    )
    log = _run(data_dir, params, EXCITED)
    assert "Freezing product weights" in log
    total, trainable = map(
        int, log.split("Parameters: ")[1].split(" trainable")[0].replace(" in total,", "").split()
    )
    assert trainable < total


def test_config_file(data_dir):
    config = data_dir / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "model: ExcitedMACE",
                "n_energies: 3",
                "compute_nacs: true",
                "compute_dipoles: true",
                f"train_file: {data_dir / 'train.xyz'}",
                "valid_fraction: 0.2",
                "E0s: average",
                "r_max: 4.0",
                "hidden_irreps: 8x0e + 8x1o",
                "MLP_irreps: 8x0e",
                "batch_size: 4",
                "max_num_epochs: 2",
                "device: cpu",
                "default_dtype: float64",
                f"work_dir: {data_dir}",
            ]
        )
    )
    run_mace_train({"name": "xmace", "config": str(config)})
    assert (data_dir / "xmace.model").exists()


def test_files_weights_and_test_sets(data_dir):
    """Several files for one head, a validation and a test file, config type weights
    and a head weight."""
    frames = ase.io.read(data_dir / "train.xyz", ":")
    for i, atoms in enumerate(frames):
        atoms.info["config_type"] = "A" if i % 2 else "B"
    ase.io.write(data_dir / "part1.xyz", frames[:12])
    ase.io.write(data_dir / "part2.xyz", frames[12:])
    heads = {
        "Default": {
            "train_file": [str(data_dir / "part1.xyz"), str(data_dir / "part2.xyz")],
            "valid_file": str(data_dir / "test.xyz"),
            "test_file": str(data_dir / "test.xyz"),
            "config_type_weights": '{"A": 1.0, "B": 0.5}',
            "weight": 2.0,
        }
    }
    params = _params(data_dir, max_num_epochs=2)
    params.pop("train_file")
    params.pop("valid_fraction")
    params["heads"] = repr(heads)
    log = _run(data_dir, params, EXCITED)
    assert "loss weight 2.0" in log
    assert "Error-table on TEST" in log


@pytest.mark.parametrize("e0s", ["{1: -1.0, 6: -3.0, 8: -5.0}", "foundation", "estimated"])
def test_e0_options(data_dir, e0s):
    params = _params(data_dir, max_num_epochs=1, E0s=e0s)
    if e0s in ("foundation", "estimated"):
        params = _foundation(params)
    log = _run(data_dir, params, EXCITED)
    assert "Atomic Energies used" in log


@pytest.mark.parametrize(
    "options",
    [
        {"scaling": "std_scaling", "error_table": "PerAtomRMSE", "num_workers": 2},
        {"scaling": "no_scaling", "eval_interval": 2, "max_num_epochs": 4, "save_cpu": None},
        {"compute_stress": True},
    ],
    ids=["std_scaling_workers", "no_scaling_eval_interval", "stress_is_ignored"],
)
def test_other_training_options(data_dir, options):
    params = _params(data_dir, **{"max_num_epochs": 2, **options})
    log = _run(data_dir, params, EXCITED)
    assert "NACs" in log  # the excited-state error table
    if "compute_stress" in options:
        assert "does not predict stress" in log


def test_foundation_elements_and_readout(data_dir):
    params = _foundation(
        _params(data_dir, max_num_epochs=1, foundation_model_elements=True, foundation_model_readout=False)
    )
    _run(data_dir, params, EXCITED)
    model = torch.load(data_dir / "xmace.model", weights_only=False)
    foundation = torch.load(GOLDEN_FOUNDATION, weights_only=False)
    assert model.atomic_numbers.tolist() == foundation.atomic_numbers.tolist()


@pytest.mark.parametrize("replay", ["fps", "pseudolabels", "filter"])
def test_multihead_replay_variants(data_dir, replay):
    _ground_state_data(data_dir / "replay.xyz", 16, seed=9)
    params = _foundation(
        _params(
            data_dir,
            max_num_epochs=1,
            multiheads_finetuning=True,
            pt_train_file=str(data_dir / "replay.xyz"),
            E0s="foundation",
            force_mh_ft_lr=True,
            num_samples_pt=8,
        )
    )
    params.pop("train_file")
    params.pop("valid_fraction")
    params["heads"] = repr(_heads(data_dir))
    params.update(
        {
            "fps": {"subselect_pt": "fps"},
            "pseudolabels": {"pseudolabel_replay": True},
            "filter": {"filter_type_pt": "combinations", "weight_pt_head": 2.0},
        }[replay]
    )
    log = _run(data_dir, params, EXCITED)
    assert "pt_head" in log
    if replay == "pseudolabels":
        assert "Successfully applied pseudolabels" in log


def test_lora_with_stage_two_and_ema(data_dir):
    params = _foundation(
        _params(data_dir, max_num_epochs=3, lora=True, lora_rank=2, swa=None, start_swa=2, ema=None)
    )
    log = _run(data_dir, params, EXCITED)
    assert "Injecting LoRA" in log and "Merging LoRA" in log


# ------------------------------------------------------------------------------------
# a trained excited-state model as the foundation
# ------------------------------------------------------------------------------------


def test_continue_and_transfer_from_an_excited_model(data_dir):
    first = data_dir / "first"
    first.mkdir()
    params = _params(data_dir, max_num_epochs=2, work_dir=str(first), name="first")
    run_mace_train(params, extra_argv=EXCITED)
    model_path = first / "first.model"
    trained = torch.load(model_path, weights_only=False)

    # continue: same head and layout, all weights kept
    cont = data_dir / "cont"
    cont.mkdir()
    params = _params(
        data_dir,
        max_num_epochs=1,
        work_dir=str(cont),
        name="cont",
        foundation_model=str(model_path),
        lr=1e-12,
    )
    params = {k: v for k, v in params.items() if k not in ARCH}
    run_mace_train(params, extra_argv=EXCITED)
    log = _log(cont, "cont")
    assert "Continuing from the excited-state foundation model" in log
    continued = torch.load(cont / "cont.model", weights_only=False)
    atoms = ase.io.read(data_dir / "test.xyz", 0)
    np.testing.assert_allclose(
        ExcitedMACECalculator(None, models=[continued]).predict(atoms)["energies"],
        ExcitedMACECalculator(None, models=[trained]).predict(atoms)["energies"],
        atol=1e-6,
    )

    # transfer: different couplings (no dipoles), new readouts, with LoRA
    transfer = data_dir / "transfer"
    transfer.mkdir()
    params = {
        k: v
        for k, v in _params(
            data_dir,
            max_num_epochs=1,
            work_dir=str(transfer),
            name="transfer",
            foundation_model=str(model_path),
            lora=True,
        ).items()
        if k not in ARCH
    }
    run_mace_train(params, extra_argv=["--compute_nacs"])
    assert "Transfer from the excited-state foundation model" in _log(transfer, "transfer")


# ------------------------------------------------------------------------------------
# other command-line tools
# ------------------------------------------------------------------------------------


def _cli(module, *args, cwd=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", module, *map(str, args)],
        check=True,
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
    )


def test_select_head_convert_device_plot_and_eval(data_dir):
    params = _params(data_dir, max_num_epochs=2)
    params.pop("train_file")
    params.pop("valid_fraction")
    params["heads"] = repr(_heads(data_dir))
    run_mace_train(params, extra_argv=EXCITED)
    model_path = data_dir / "xmace.model"

    listed = _cli("mace.cli.select_head", "--list_heads", model_path)
    assert "excited" in listed.stdout and "ground" in listed.stdout
    _cli("mace.cli.select_head", "-n", "excited", "-o", data_dir / "excited.model", model_path)
    single = torch.load(data_dir / "excited.model", weights_only=False)
    assert single.heads == ["excited"] and single.head_n_states == [3]
    atoms = ase.io.read(data_dir / "test.xyz", 0)
    np.testing.assert_allclose(
        ExcitedMACECalculator(data_dir / "excited.model").predict(atoms)["energies"],
        ExcitedMACECalculator(model_path, head="excited").predict(atoms)["energies"],
        atol=1e-8,
    )

    _cli("mace.cli.convert_device", "--target_device", "cpu", "--output_file",
         data_dir / "cpu.model", data_dir / "excited.model")
    assert (data_dir / "cpu.model").exists()

    _cli("mace.cli.plot_train", "--path", data_dir / "results", "--heads", "excited,ground",
         cwd=data_dir)
    assert len(list(data_dir.glob("*.png"))) >= 2

    out = data_dir / "pred.xyz"
    _cli("mace.cli.eval_configs", "--configs", data_dir / "test.xyz", "--model", model_path,
         "--output", out, "--head", "ground")
    # one state: extended XYZ stores a 1-element array as a number
    assert np.size(ase.io.read(out, 0).info["MACE_energies"]) == 1
    with pytest.raises(subprocess.CalledProcessError) as error:
        _cli("mace.cli.eval_configs", "--configs", data_dir / "test.xyz", "--model",
             model_path, "--output", out, "--compute_stress")
    assert "not supported for excited-state models" in error.value.stderr


# ------------------------------------------------------------------------------------
# excited-state foundation model pipeline
# ------------------------------------------------------------------------------------


def test_foundation_pipeline_dry_run(tmp_path):
    """foundation/dry_run.sh: mock data sets -> sharded HDF5 -> multi-head training on
    two data-parallel processes -> restart -> continued and transferred fine-tuning."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["NPROC"] = "2"
    done = subprocess.run(
        ["bash", str(REPO / "foundation" / "dry_run.sh"), str(tmp_path / "fm")],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    work = tmp_path / "fm"
    model = torch.load(work / "run" / "xmace_fm_mock.model", weights_only=False)
    assert sorted(model.heads) == ["public_energies", "radicals", "sharc_st", "tddft_singlets"]
    log = "\n".join(p.read_text() for p in (work / "run" / "logs").glob("*.log"))
    # every data set trains on its own head, with its own statistics
    for name in model.heads:
        assert f"Head {name}:" in log
    assert "Loading checkpoint" in log  # the restart
    cont = "\n".join(p.read_text() for p in (work / "finetune_continue" / "logs").glob("*.log"))
    assert "Continuing from the excited-state foundation model" in cont
    tran = "\n".join(p.read_text() for p in (work / "finetune_transfer" / "logs").glob("*.log"))
    assert "Transfer from the excited-state foundation model" in tran
