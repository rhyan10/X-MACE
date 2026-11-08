"""End-to-end training of excited-state models through the CLI (X-MACE workflows)."""

import json
from pathlib import Path

import ase.io
import numpy as np
import pytest
import torch

from mace.calculators import ExcitedMACECalculator
from tests.helpers import run_mace_train

from excited_data import make_excited_configs  # pylint: disable=import-error

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
    ase.io.write(tmp_path / "train.xyz", make_excited_configs(n_train, 3, seed=1, **kwargs))
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
        params, extra_argv=["--compute_nacs", "--compute_socs", "--compute_dipoles", "--pair_repulsion"]
    )
    losses = _valid_losses(tmp_path)
    assert losses[-1] < 0.5 * losses[0]
    log = _log(tmp_path)
    assert "NACs=3 (raw), SOCs=3, dipoles=6, pair repulsion=on" in log
    assert "RMSE NACs" in log and "Error-table on TEST" in log
    model = torch.load(tmp_path / "xmace.model", weights_only=False)
    assert model.__class__.__name__ == "ExcitedMACE"
    assert hasattr(model, "pair_repulsion_fn")
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
                [data.AtomicData.from_config(config, z_table=AtomicNumberTable([1, 6, 8]), cutoff=4.0)],
                batch_size=1,
            )
        )
    )
    out = compiled(batch.to_dict())
    np.testing.assert_allclose(out["energy"][0].detach().numpy(), ref["energies"], rtol=1e-10)


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
    configs = make_excited_configs(24, 3, seed=1, smooth_key="smooth_nacs", with_socs=False)
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
    assert "Architecture taken from the foundation model" in log
    assert "Every state starts from the foundation energy readout" in log
    foundation = torch.load(GOLDEN_FOUNDATION, weights_only=False)
    model = torch.load(tmp_path / "xmace.model", weights_only=False)
    assert float(model.r_max) == float(foundation.r_max)


def test_single_state_float32(tmp_path):
    configs = make_excited_configs(20, 1, seed=1, with_nacs=False, with_socs=False, with_dipoles=False)
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
