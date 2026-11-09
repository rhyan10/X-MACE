"""End-to-end training of excited-state models through the CLI (X-MACE workflows)."""

import json
from pathlib import Path

import ase.io
import numpy as np
import pytest
import torch
from excited_data import (  # pylint: disable=import-error
    make_excited_configs,
    make_multiplicity_configs,
)

from mace.calculators import ExcitedMACECalculator
from tests.helpers import run_mace_train

pytestmark = pytest.mark.slow

GOLDEN_FOUNDATION = (
    Path(__file__).resolve().parents[1] / "golden" / "models" / "tiny_scaleshift.model"
)
EVAL_SCRIPT = Path(__file__).resolve().parents[2] / "mace" / "cli" / "eval_configs.py"


def _params(tmp_path, **overrides):
    params = {
        "name": "xmace",
        "train_file": str(tmp_path / "train.xyz"),
        "valid_fraction": 0.2,
        "model": "ExcitedMACE",
        "n_energies": 3,
        "E0s": "average",
        "r_max": 4.0,
        "hidden_irreps": "8x0e + 8x1o",
        "MLP_irreps": "8x0e",
        "num_interactions": 2,
        "correlation": 2,
        "batch_size": 4,
        "max_num_epochs": 6,
        "lr": 0.01,
        "device": "cpu",
        "default_dtype": "float64",
        "seed": 1,
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "nacs_weight": 10.0,
        "socs_weight": 10.0,
        "dipoles_weight": 1.0,
        "work_dir": str(tmp_path),
    }
    params.update(overrides)
    return params


def _write_data(tmp_path, n_train=24, **kwargs):
    ase.io.write(
        tmp_path / "train.xyz", make_excited_configs(n_train, 3, seed=1, **kwargs)
    )
    ase.io.write(tmp_path / "test.xyz", make_excited_configs(6, 3, seed=2, **kwargs))


def _valid_losses(tmp_path, name="xmace"):
    files = sorted((tmp_path / "results").glob(f"{name}*_train.txt"))
    assert files, list((tmp_path / "results").iterdir())
    losses = []
    for line in files[0].read_text().splitlines():
        entry = json.loads(line)
        if entry.get("mode") == "eval":
            losses.append(entry["loss"])
    return losses


def _log(tmp_path, name="xmace"):
    return "\n".join(p.read_text() for p in (tmp_path / "logs").glob(f"{name}*.log"))


def test_excited_mace_all_properties(tmp_path):
    _write_data(tmp_path, random_phases=True)
    params = _params(
        tmp_path,
        test_file=str(tmp_path / "test.xyz"),
        error_table="EnergyNacsDipoleRMSE",
    )
    run_mace_train(
        params,
        extra_argv=[
            "--compute_nacs",
            "--compute_socs",
            "--compute_dipoles",
            "--pair_repulsion",
        ],
    )
    losses = _valid_losses(tmp_path)
    assert losses[-1] < 0.5 * losses[0]
    log = _log(tmp_path)
    assert "NACs=3 (raw), SOCs=3, dipoles=6, coupling parity=both, pair repulsion=on" in log
    assert "RMSE NACs" in log and "Error-table on TEST" in log
    model = torch.load(tmp_path / "xmace.model", weights_only=False)
    assert model.__class__.__name__ == "ExcitedMACE"
    assert hasattr(model.backbone.model, "pair_repulsion_fn")
    assert (tmp_path / "xmace_compiled.model").exists()
    compiled = torch.jit.load(tmp_path / "xmace_compiled.model")
    atoms = ase.io.read(tmp_path / "test.xyz", 0)
    calc = ExcitedMACECalculator(tmp_path / "xmace.model")
    ref = calc.predict(atoms)
    from mace import data
    from mace.tools import AtomicNumberTable, torch_geometric

    config = data.config_from_atoms(atoms, key_specification=data.KeySpecification())
    batch = next(
        iter(
            torch_geometric.dataloader.DataLoader(
                [
                    data.AtomicData.from_config(
                        config, z_table=AtomicNumberTable([1, 6, 8]), cutoff=4.0
                    )
                ],
                batch_size=1,
            )
        )
    )
    out = compiled(batch.to_dict())
    np.testing.assert_allclose(
        out["energy"][0].detach().numpy(), ref["energies"], rtol=1e-10
    )


def test_xmace_readme_command_line(tmp_path):
    """The X-MACE README flags: bare --compute_nacs/--compute_socs, nac_num/soc_num."""
    _write_data(tmp_path, with_dipoles=False)
    params = _params(
        tmp_path,
        nac_num=3,
        soc_num=3,
        error_table="EnergyNacsDipoleMAE",
        nacs_key="REF_nacs",
        ema=None,
        ema_decay=0.99,
        max_num_epochs=3,
    )
    run_mace_train(params, extra_argv=["--compute_nacs", "--compute_socs"])
    assert "MAE NACs" in _log(tmp_path)


def test_autoencoder_with_stage_two(tmp_path):
    _write_data(tmp_path)
    params = _params(
        tmp_path,
        model="AutoencoderExcitedMACE",
        num_permutational_invariant=6,
        swa=None,
        start_swa=3,
        max_num_epochs=5,
    )
    run_mace_train(params, extra_argv=["--compute_nacs", "--pair_repulsion"])
    model = torch.load(tmp_path / "xmace.model", weights_only=False)
    assert model.__class__.__name__ == "AutoencoderExcitedMACE"
    assert (tmp_path / "xmace_stagetwo.model").exists()
    log = _log(tmp_path)
    assert "Stage Two" in log
    losses = _valid_losses(tmp_path)
    assert losses[2] < losses[0]


def test_smooth_nacs_key_and_missing_labels(tmp_path):
    configs = make_excited_configs(
        24, 3, seed=1, smooth_key="smooth_nacs", with_socs=False
    )
    for atoms in configs[::3]:
        atoms.info.pop("smooth_nacs")  # a third of the frames without NACs
    ase.io.write(tmp_path / "train.xyz", configs)
    params = _params(tmp_path, nacs_key="smooth_nacs", max_num_epochs=2)
    run_mace_train(params, extra_argv=["--compute_nacs"])
    log = _log(tmp_path)
    assert "NACs=3 (smooth)" in log
    assert "nacs=" in log  # label counts are reported


def test_foundation_model_transfer(tmp_path):
    _write_data(tmp_path, with_socs=False, with_dipoles=False)
    params = _params(
        tmp_path,
        foundation_model=str(GOLDEN_FOUNDATION),
        max_num_epochs=2,
        hidden_irreps=None,
        MLP_irreps=None,
    )
    params = {k: v for k, v in params.items() if v is not None}
    run_mace_train(params, extra_argv=["--compute_nacs"])
    log = _log(tmp_path)
    assert "Backbone architecture taken from the foundation model" in log
    assert "every state starts from the foundation energy" in log
    foundation = torch.load(GOLDEN_FOUNDATION, weights_only=False)
    model = torch.load(tmp_path / "xmace.model", weights_only=False)
    assert float(model.r_max) == float(foundation.r_max)


def test_single_state_float32(tmp_path):
    configs = make_excited_configs(
        20, 1, seed=1, with_nacs=False, with_socs=False, with_dipoles=False
    )
    ase.io.write(tmp_path / "train.xyz", configs)
    params = _params(tmp_path, n_energies=1, default_dtype="float32", max_num_epochs=2)
    run_mace_train(params)
    model = torch.load(tmp_path / "xmace.model", weights_only=False)
    assert model.n_energies == 1


def test_eval_configs_cli(tmp_path):
    _write_data(tmp_path)
    params = _params(tmp_path, max_num_epochs=1)
    run_mace_train(params, extra_argv=["--compute_nacs", "--compute_dipoles"])
    run_mace_train(
        {
            "configs": str(tmp_path / "test.xyz"),
            "model": str(tmp_path / "xmace.model"),
            "output": str(tmp_path / "pred.xyz"),
            "device": "cpu",
        },
        script=EVAL_SCRIPT,
    )
    atoms = ase.io.read(tmp_path / "pred.xyz", 0)
    assert atoms.info["MACE_energies"].shape == (3,)
    assert np.shape(atoms.info["MACE_forces"]) == (4, 3, 3)
    assert np.shape(atoms.info["MACE_nacs"]) == (4, 3, 3)
    assert np.shape(atoms.info["MACE_dipoles"]) == (6, 3)


def test_missing_n_energies_is_rejected_cleanly(tmp_path):
    _write_data(tmp_path)
    params = _params(tmp_path, max_num_epochs=1)
    params.pop("n_energies")
    result = run_mace_train(params, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "expected 1 energies" in result.stderr + result.stdout


def test_hdf5_preprocessing_and_training(tmp_path):
    _write_data(tmp_path)
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
            "compute_statistics": None,
            "compute_nacs": None,
            "compute_socs": None,
        },
        script=Path(__file__).resolve().parents[2]
        / "mace"
        / "cli"
        / "preprocess_data.py",
    )
    statistics = json.loads((tmp_path / "h5" / "statistics.json").read_text())
    assert len(statistics["mean"]) == 3  # one shift per state
    params = _params(
        tmp_path,
        train_file=str(tmp_path / "h5" / "train" / "train_0.h5"),
        valid_file=str(tmp_path / "h5" / "val" / "val_0.h5"),
        statistics_file=str(tmp_path / "h5" / "statistics.json"),
        max_num_epochs=2,
    )
    params.pop("valid_fraction")
    params.pop("E0s")
    run_mace_train(params, extra_argv=["--compute_nacs", "--compute_socs"])
    assert "MAE NACs" in _log(tmp_path)


def test_multiplicities_with_spin_orbit_couplings(tmp_path):
    # S0, T0, T1: NACs only T0-T1, SOCs as Wigner-Eckart coupling vectors
    states = [1, 0, 2]
    ase.io.write(tmp_path / "train.xyz", make_multiplicity_configs(24, states, seed=1))
    ase.io.write(tmp_path / "test.xyz", make_multiplicity_configs(6, states, seed=2))
    params = _params(
        tmp_path,
        test_file=str(tmp_path / "test.xyz"),
        error_table="EnergyNacsDipoleRMSE",
        max_num_epochs=8,
    )
    params.pop("n_energies")
    run_mace_train(
        params,
        extra_argv=[
            "--states=1 0 2",
            "--compute_nacs",
            "--compute_socs",
            "--compute_dipoles",
        ],
    )
    losses = _valid_losses(tmp_path)
    assert losses[-1] < 0.5 * losses[0]
    log = _log(tmp_path)
    assert "3 states (1 singlet, 2 triplets), NACs=1 (raw)" in log
    assert "SOCs=3 coupling vectors (spin_orbit), dipoles=4" in log
    assert "RMSE SOCs" in log
    model = torch.load(tmp_path / "xmace.model", weights_only=False)
    assert model.multiplicity_states == states and model.soc_mode == "spin_orbit"

    # evaluation writes the SOC matrix in a layout training reads back
    run_mace_train(
        {
            "configs": str(tmp_path / "test.xyz"),
            "model": str(tmp_path / "xmace.model"),
            "output": str(tmp_path / "pred.xyz"),
            "device": "cpu",
        },
        script=EVAL_SCRIPT,
    )
    atoms = ase.io.read(tmp_path / "pred.xyz", 0)
    assert np.shape(atoms.info["MACE_soc_vectors"]) == (3, 3)
    assert np.size(atoms.info["MACE_socs"]) == 7 * 6  # 7 M_S states, upper triangle
    assert np.shape(atoms.info["MACE_nacs"]) == (4, 1, 3)

    # the SHARC layer reads the multiplicities from the model
    from mace.calculators.sharc_calculator import (  # pylint: disable=import-outside-toplevel
        SharcCalculator,
    )

    sharc = SharcCalculator(
        atom_types=atoms.get_chemical_symbols(),
        model_paths=str(tmp_path / "xmace.model"),
        states=states,
        properties=["energy", "forces", "nacs", "socs", "dipoles"],
    )
    qm = sharc.calculate(atoms.positions / 0.529177210903)
    assert qm["h"].shape == (7, 7) and np.any(np.abs(qm["h"].imag) > 0)


def test_multiplicities_hdf5(tmp_path):
    states = [0, 2, 0, 1]  # doublets and a quartet
    ase.io.write(tmp_path / "train.xyz", make_multiplicity_configs(20, states, seed=1))
    run_mace_train(
        {
            "train_file": str(tmp_path / "train.xyz"),
            "valid_fraction": 0.2,
            "h5_prefix": str(tmp_path / "h5") + "/",
            "r_max": 4.0,
            "E0s": "average",
            "states": "0 2 0 1",
            "seed": 1,
            "num_process": 1,
            "compute_statistics": None,
            "compute_nacs": None,
            "compute_socs": None,
        },
        script=Path(__file__).resolve().parents[2]
        / "mace"
        / "cli"
        / "preprocess_data.py",
    )
    params = _params(
        tmp_path,
        train_file=str(tmp_path / "h5" / "train" / "train_0.h5"),
        valid_file=str(tmp_path / "h5" / "val" / "val_0.h5"),
        statistics_file=str(tmp_path / "h5" / "statistics.json"),
        max_num_epochs=2,
    )
    for key in ("valid_fraction", "E0s", "n_energies"):
        params.pop(key)
    run_mace_train(
        params, extra_argv=["--states=0 2 0 1", "--compute_nacs", "--compute_socs"]
    )
    log = _log(tmp_path)
    assert "SOCs=3 coupling vectors" in log
    assert "MAE SOCs" in log


# ------------------------------------------------------------------------------------
# heads, foundation backbones, multihead fine-tuning and LoRA
# ------------------------------------------------------------------------------------

GOLDEN = GOLDEN_FOUNDATION.parent


def _ground_state_data(path, n_configs, seed):
    """Plain ground-state frames (one energy, forces as an array), as in MACE data."""
    frames = []
    for atoms in make_excited_configs(
        n_configs, 1, seed=seed, with_nacs=False, with_socs=False, with_dipoles=False
    ):
        energy = float(np.atleast_1d(atoms.info.pop("REF_energy"))[0])
        forces = np.asarray(atoms.info.pop("REF_forces")).reshape(len(atoms), -1, 3)
        atoms.info["REF_energy"] = energy
        atoms.arrays["REF_forces"] = forces[:, 0]
        frames.append(atoms)
    ase.io.write(path, frames)


def _heads(tmp_path, ground=True, **excited):
    heads = {
        "excited": {
            "train_file": str(tmp_path / "train.xyz"),
            "valid_fraction": 0.2,
            **excited,
        }
    }
    if ground:
        heads["ground"] = {
            "train_file": str(tmp_path / "ground.xyz"),
            "valid_fraction": 0.2,
            "ground_state": True,
        }
    return heads


@pytest.mark.parametrize("model", ["ExcitedMACE", "AutoencoderExcitedMACE"])
def test_mixed_ground_and_excited_heads(tmp_path, model):
    _write_data(tmp_path, with_socs=False)
    _ground_state_data(tmp_path / "ground.xyz", 16, seed=5)
    params = _params(tmp_path, model=model, max_num_epochs=3)
    params.pop("train_file")
    params.pop("valid_fraction")
    params["heads"] = repr(_heads(tmp_path))
    extra = ["--compute_nacs", "--compute_dipoles"]
    if model == "AutoencoderExcitedMACE":
        extra.append("--num_permutational_invariant=4")
    run_mace_train(params, extra_argv=extra)
    log = _log(tmp_path)
    assert "Head ground: ground state" in log
    trained = torch.load(tmp_path / "xmace.model", weights_only=False)
    assert trained.heads == ["excited", "ground"]
    assert trained.head_n_states == [3, 1]
    masks = trained.mask_tables()
    assert masks["nacs"][1].sum() == 0 and masks["nacs"][0].sum() == 3
    atoms = ase.io.read(tmp_path / "test.xyz", 0)
    excited = ExcitedMACECalculator(tmp_path / "xmace.model", head="excited")
    assert excited.predict(atoms)["energies"].shape == (3,)
    ground = ExcitedMACECalculator(tmp_path / "xmace.model", head="ground")
    results = ground.predict(atoms)
    assert results["energies"].shape == (1,)
    assert "nacs" not in results or results["nacs"] is None


@pytest.mark.parametrize("foundation", ["tiny_scaleshift", "tiny_mace", "tiny_magnetic"])
def test_foundation_backbones(tmp_path, foundation):
    """ScaleShiftMACE / MACELES / PolarMACE foundations are re-built for the training
    elements (as in MACE fine-tuning); others (plain MACE, magnetic) are used as they
    are. (tiny_maceles keeps l=1 skip connections in its last layer, which MACE's
    element transfer does not support, so it is not used here.)"""
    if foundation == "tiny_magnetic":
        pytest.importorskip("sphericart.torch")
    _write_data(tmp_path, with_socs=False, with_dipoles=False)
    if foundation == "tiny_magnetic":  # an O/Fe model: the same frames as Fe2O2
        for name in ("train.xyz", "test.xyz"):
            frames = ase.io.read(tmp_path / name, ":")
            for atoms in frames:
                atoms.set_chemical_symbols(["Fe", "O", "O", "Fe"])
                atoms.arrays["REF_magmom"] = np.zeros((len(atoms), 3))
            ase.io.write(tmp_path / name, frames)
    params = _params(
        tmp_path,
        foundation_model=str(GOLDEN / f"{foundation}.model"),
        max_num_epochs=2,
    )
    for key in ("hidden_irreps", "MLP_irreps", "num_interactions", "correlation", "r_max"):
        params.pop(key)
    run_mace_train(params, extra_argv=["--compute_nacs"])
    log = _log(tmp_path)
    assert "every state starts from the foundation energy" in log
    foundation_model = torch.load(GOLDEN / f"{foundation}.model", weights_only=False)
    trained = torch.load(tmp_path / "xmace.model", weights_only=False)
    assert trained.backbone_class == foundation_model.__class__.__name__
    atoms = ase.io.read(tmp_path / "test.xyz", 0)
    results = ExcitedMACECalculator(tmp_path / "xmace.model").predict(atoms)
    assert np.all(np.isfinite(results["energies"]))


@pytest.mark.parametrize("lora", [False, True])
def test_multihead_finetuning_with_excited_and_ground_heads(tmp_path, lora):
    """Fine-tune a foundation model on excited-state and ground-state data at once,
    with replay of the foundation's data (pt_head), optionally with LoRA."""
    _write_data(tmp_path, with_socs=False, with_dipoles=False)
    _ground_state_data(tmp_path / "ground.xyz", 16, seed=5)
    _ground_state_data(tmp_path / "replay.xyz", 16, seed=9)
    params = _params(
        tmp_path,
        foundation_model=str(GOLDEN_FOUNDATION),
        max_num_epochs=2,
        multiheads_finetuning=True,
        pt_train_file=str(tmp_path / "replay.xyz"),
        num_samples_pt=8,
        force_mh_ft_lr=True,
        E0s="foundation",
    )
    for key in ("hidden_irreps", "MLP_irreps", "num_interactions", "correlation", "r_max"):
        params.pop(key)
    params.pop("train_file")
    params.pop("valid_fraction")
    params["heads"] = repr(_heads(tmp_path))
    extra = ["--compute_nacs"]
    if lora:
        extra += ["--lora=True", "--lora_rank=2"]
    run_mace_train(params, extra_argv=extra)
    log = _log(tmp_path)
    assert "Head pt_head: ground state" in log
    trained = torch.load(tmp_path / "xmace.model", weights_only=False)
    assert trained.heads[0] == "pt_head"
    assert sorted(trained.heads) == ["excited", "ground", "pt_head"]
    n_states = dict(zip(trained.heads, trained.head_n_states))
    assert n_states == {"pt_head": 1, "excited": 3, "ground": 1}
    if lora:
        assert "LoRA" in log
    atoms = ase.io.read(tmp_path / "test.xyz", 0)
    calc = ExcitedMACECalculator(tmp_path / "xmace.model")  # default: first excited head
    assert calc.predict(atoms)["energies"].shape == (3,)


@pytest.mark.parametrize("backbone", ["MACE", "MACELES", "PolarMACE"])
@pytest.mark.parametrize("long_range", ["shared", "state"])
def test_backbones_from_scratch(tmp_path, backbone, long_range):
    if backbone == "MACELES":
        pytest.importorskip("les")
    if backbone == "PolarMACE":
        pytest.importorskip("graph_longrange")
    _write_data(tmp_path, with_socs=False)
    params = _params(tmp_path, max_num_epochs=2)
    run_mace_train(
        params,
        extra_argv=[
            "--compute_nacs",
            "--compute_dipoles",
            f"--excited_backbone={backbone}",
            f"--excited_long_range={long_range}",
        ],
    )
    trained = torch.load(tmp_path / "xmace.model", weights_only=False)
    # (MACE's --model=MACE builds a ScaleShiftMACE without shift)
    assert trained.backbone_class == {"MACE": "ScaleShiftMACE"}.get(backbone, backbone)
    assert trained.long_range_mode == long_range
    losses = _valid_losses(tmp_path)
    assert np.all(np.isfinite(losses))


# ------------------------------------------------------------------------------------
# data-parallel training (torchrun, two CPU processes with gloo)
# ------------------------------------------------------------------------------------


def _torchrun(params, extra_argv, nproc=2):
    import os  # pylint: disable=import-outside-toplevel
    import subprocess  # pylint: disable=import-outside-toplevel
    import sys  # pylint: disable=import-outside-toplevel

    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "1"
    argv = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(repo / "mace" / "cli" / "run_train.py"),
    ]
    argv += [f"--{k}={v}" if v is not None else f"--{k}" for k, v in params.items()]
    argv += list(extra_argv) + ["--distributed", "--launcher=torchrun"]
    subprocess.run(argv, check=True, env=env, cwd=params["work_dir"])


@pytest.mark.parametrize("model", ["ExcitedMACE", "AutoencoderExcitedMACE"])
def test_distributed_training_with_mixed_heads(tmp_path, model):
    """Two processes share each batch; the statistics are taken over all ranks, so the
    model starts exactly as in a single-process run, and training completes."""
    _write_data(tmp_path, with_socs=False)
    _ground_state_data(tmp_path / "ground.xyz", 20, seed=5)
    extra = ["--compute_nacs", "--compute_dipoles"]
    if model == "AutoencoderExcitedMACE":
        extra.append("--num_permutational_invariant=4")
    logs = {}
    for name, runner in (("single", run_mace_train), ("ddp", _torchrun)):
        work = tmp_path / name
        work.mkdir()
        params = _params(tmp_path, model=model, max_num_epochs=2, work_dir=str(work))
        params.pop("train_file")
        params.pop("valid_fraction")
        params["heads"] = repr(_heads(tmp_path))
        runner(params, extra)
        logs[name] = _log(work)
        trained = torch.load(work / "xmace.model", weights_only=False)
        assert trained.heads == ["excited", "ground"]
    assert "Processes: 2" in logs["ddp"]

    def statistics(log):
        return sorted(  # (the log and the debug log have the same lines)
            {
                line.split("INFO: ")[1]
                for line in log.splitlines()
                if "interaction energy scale" in line and "INFO: " in line
            }
        )

    assert statistics(logs["ddp"]) == statistics(logs["single"])
    assert len(statistics(logs["single"])) == 2


def test_mixed_heads_from_hdf5(tmp_path):
    """Preprocessed data of heads with different layouts share batches: the labels of
    each head are padded to the largest head as they are read."""
    _write_data(tmp_path, with_socs=False, with_dipoles=False)
    _ground_state_data(tmp_path / "ground.xyz", 16, seed=5)
    script = Path(__file__).resolve().parents[2] / "mace" / "cli" / "preprocess_data.py"
    common = {"valid_fraction": 0.2, "r_max": 4.0, "E0s": "average", "seed": 1,
              "num_process": 1}
    run_mace_train(
        {**common, "train_file": str(tmp_path / "train.xyz"),
         "h5_prefix": str(tmp_path / "ex") + "/", "n_energies": 3,
         "compute_nacs": None},
        script=script,
    )
    run_mace_train(
        {**common, "train_file": str(tmp_path / "ground.xyz"),
         "h5_prefix": str(tmp_path / "gs") + "/"},
        script=script,
    )
    heads = {
        name: {
            "train_file": str(tmp_path / prefix / "train" / "train_0.h5"),
            "valid_file": str(tmp_path / prefix / "val" / "val_0.h5"),
            "E0s": "{1: -1.0, 6: -3.0, 8: -5.0}",
            **extra,
        }
        for name, prefix, extra in (
            ("excited", "ex", {}),
            ("ground", "gs", {"ground_state": True}),
        )
    }
    params = _params(tmp_path, max_num_epochs=2)
    for key in ("train_file", "valid_fraction", "E0s"):
        params.pop(key)
    params["heads"] = repr(heads)
    params["atomic_numbers"] = "[1, 6, 8]"
    run_mace_train(params, extra_argv=["--compute_nacs"])
    trained = torch.load(tmp_path / "xmace.model", weights_only=False)
    assert trained.heads == ["excited", "ground"] and trained.head_n_states == [3, 1]
    assert np.all(np.isfinite(_valid_losses(tmp_path)))
