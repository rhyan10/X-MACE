"""Phase-free losses, weights, autoencoder terms, metrics and optimizer coverage."""

import argparse
import json

import numpy as np
import pytest
import torch

from mace.modules.excited import align_nacs
from mace.modules.excited_loss import (
    WeightedExcitedStatesLoss,
    phase_free_dipoles_loss,
    phase_free_nacs_loss,
    phase_free_socs_loss,
)
from mace.tools.scripts_utils import get_params_options
from mace.tools.train import evaluate

from conftest import atoms_to_batch, build_excited, full_spec  # pylint: disable=import-error
from excited_data import make_excited_configs  # pylint: disable=import-error


def _perfect_prediction(batch):
    return {
        "energy": batch.energy.clone(),
        "forces": batch.forces.clone(),
        "nacs": batch.nacs.clone(),
        "socs": batch.socs.clone(),
        "dipoles": batch.dipoles.clone(),
    }


@pytest.fixture(name="batch")
def fixture_batch(molecules):
    return atoms_to_batch(molecules, full_spec())


def _loss():
    return WeightedExcitedStatesLoss(
        energy_weight=1.0,
        forces_weight=1.0,
        nacs_weight=1.0,
        socs_weight=1.0,
        dipoles_weight=1.0,
        n_permanent_dipoles=3,
    )


def test_perfect_prediction_has_zero_loss(batch):
    loss = _loss()(batch, _perfect_prediction(batch))
    assert float(loss) < 1e-5  # sqrt(eps) of the RMSE terms


def test_loss_is_invariant_to_coupling_phases(batch):
    pred = _perfect_prediction(batch)
    rng = np.random.default_rng(0)
    n_graphs = batch.num_graphs
    pair_signs = torch.tensor(rng.choice([-1.0, 1.0], size=(n_graphs, 3)))
    pred["nacs"] = pred["nacs"] * pair_signs[batch.batch].unsqueeze(-1)
    pred["socs"] = pred["socs"] * torch.tensor(rng.choice([-1.0, 1.0], size=(n_graphs, 1)))
    transition_signs = torch.tensor(rng.choice([-1.0, 1.0], size=(n_graphs, 3, 1)))
    pred["dipoles"][:, 3:] = pred["dipoles"][:, 3:] * transition_signs
    assert float(_loss()(batch, pred)) < 1e-5


def test_permanent_dipole_sign_matters(batch):
    pred = _perfect_prediction(batch)
    pred["dipoles"][:, 0] = -pred["dipoles"][:, 0]
    assert float(phase_free_dipoles_loss(batch, pred, n_permanent=3)) > 1e-6


def test_nac_phase_is_per_configuration_not_per_atom(batch):
    """Flipping single atoms is not a phase change (X-MACE chose the sign per atom)."""
    pred = _perfect_prediction(batch)
    pred["nacs"][0] = -pred["nacs"][0]  # one atom only
    assert float(phase_free_nacs_loss(batch, pred)) > 1e-3


def test_align_nacs_picks_best_sign():
    ref = torch.randn(6, 2, 3)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    pred = ref.clone()
    pred[3:, 1] *= -1
    torch.testing.assert_close(align_nacs(ref, pred, batch, 2), ref)


def test_soc_phase_is_one_per_configuration(batch):
    pred = _perfect_prediction(batch)
    pred["socs"][0, 0] = -pred["socs"][0, 0]  # a single element
    assert float(phase_free_socs_loss(batch, pred)) > 1e-4


def test_zero_weight_frames_do_not_contribute(molecules):
    batch = atoms_to_batch(molecules, full_spec())
    pred = _perfect_prediction(batch)
    for key in ("energy", "nacs", "socs", "dipoles"):
        pred[key] = pred[key].clone()
    # corrupt everything of graph 1, then remove its labels via weights
    first = slice(batch.ptr[1], batch.ptr[2])
    pred["energy"][1] += 5.0
    pred["forces"][first] += 1.0
    pred["nacs"][first] += 1.0
    pred["socs"][1] += 1.0
    pred["dipoles"][1] += 1.0
    assert float(_loss()(batch, pred)) > 1.0
    for name in ("energy", "forces", "nacs", "socs", "dipoles"):
        batch[f"{name}_weight"][1] = 0.0
    assert float(_loss()(batch, pred)) < 1e-5
    # the global config_weight switches all terms off as well
    batch2 = atoms_to_batch(molecules, full_spec())
    batch2.weight[1] = 0.0
    assert float(_loss()(batch2, pred)) < 1e-5


def test_loss_skips_terms_the_model_does_not_predict(batch):
    pred = _perfect_prediction(batch)
    pred["energy"] = pred["energy"] + 1.0
    for key in ("nacs", "socs", "dipoles"):
        pred[key] = None
    torch.testing.assert_close(_loss()(batch, pred), torch.tensor(1.0))


def test_energy_loss_uses_total_energies(batch):
    pred = _perfect_prediction(batch)
    pred["energy"] = pred["energy"] + 2.0
    loss = WeightedExcitedStatesLoss(1.0, 0.0, 0.0, 0.0, 0.0)
    torch.testing.assert_close(loss(batch, pred), torch.tensor(4.0))


def test_autoencoder_loss_terms(molecules):
    model = build_excited(autoencoder=True)
    batch = atoms_to_batch(molecules, full_spec())
    out = model(batch.to_dict(), training=True)
    loss_fn = WeightedExcitedStatesLoss(1.0, 0.0, 0.0, 0.0, 0.0)
    total = loss_fn(batch, out)
    plain = WeightedExcitedStatesLoss(1.0, 0.0, 0.0, 0.0, 0.0)(
        batch, {"energy": out["energy"]}
    )
    assert float(total) > float(plain)  # reconstruction + latent terms are included
    total.backward()
    assert model.perm_encoder.post_aggregation_nn[-1].weight.grad is not None
    assert model.perm_decoder.decoder_nn[-1].weight.grad is not None


def test_metrics_use_phase_aligned_couplings(molecules):
    model = build_excited(soc_num=3)
    batch = atoms_to_batch(molecules, full_spec())
    loader = [batch]
    output_args = {"forces": True, "virials": False, "stress": False}
    _, metrics = evaluate(model, _loss(), loader, output_args, device="cpu")
    for key in ("mae_e", "mae_e_per_atom", "rmse_f", "mae_nacs", "rmse_socs", "mae_dipoles"):
        assert metrics[key] is not None and np.isfinite(metrics[key]), key
    # flipping the sign of the model's NAC readouts changes nothing in the metrics
    model.nacs_readouts[0].linear.weight.data.mul_(-1)
    last = model.nacs_readouts[1].linear_2.weight
    last.data.mul_(-1)
    _, flipped = evaluate(model, _loss(), loader, output_args, device="cpu")
    np.testing.assert_allclose(flipped["mae_nacs"], metrics["mae_nacs"], rtol=1e-10)


def test_metrics_ignore_unlabelled_frames(molecules):
    model = build_excited()
    atoms = molecules[1].copy()
    atoms.info = {"REF_energy": molecules[1].info["REF_energy"]}
    batch = atoms_to_batch([molecules[0], atoms], full_spec())
    output_args = {"forces": True, "virials": False, "stress": False}
    _, metrics = evaluate(model, _loss(), [batch], output_args, device="cpu")
    ref = atoms_to_batch([molecules[0]], full_spec())
    _, metrics_ref = evaluate(model, _loss(), [ref], output_args, device="cpu")
    np.testing.assert_allclose(metrics["mae_nacs"], metrics_ref["mae_nacs"], rtol=1e-10)
    np.testing.assert_allclose(metrics["mae_f"], metrics_ref["mae_f"], rtol=1e-10)


@pytest.mark.parametrize("autoencoder", [False, True])
def test_every_parameter_is_in_an_optimizer_group(autoencoder):
    """X-MACE never optimised the SOC readouts or the autoencoder's latent readouts."""
    model = build_excited(autoencoder=autoencoder, pair_repulsion=True)
    args = argparse.Namespace(
        lr=0.01,
        weight_decay=0.0,
        amsgrad=False,
        beta=0.9,
        freeze=None,
        lr_params_factors=json.dumps({}),
    )
    options = get_params_options(args, model)
    claimed = {id(p) for group in options["params"] for p in group["params"]}
    names = [n for n, p in model.named_parameters() if id(p) not in claimed]
    assert not names
    group_names = {group["name"] for group in options["params"]}
    assert {"nacs_readouts", "socs_readouts", "dipoles_readouts"} <= group_names
    if autoencoder:
        assert {"perm_encoder", "perm_decoder"} <= group_names


def test_training_steps_reduce_the_loss():
    torch.manual_seed(0)
    model = build_excited(pair_repulsion=True)
    batch = atoms_to_batch(make_excited_configs(8, 3, seed=11, random_phases=True), full_spec())
    loss_fn = WeightedExcitedStatesLoss(1.0, 10.0, 10.0, 10.0, 1.0, n_permanent_dipoles=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    losses = []
    for _ in range(60):
        optimizer.zero_grad()
        out = model(batch.to_dict(), training=True)
        loss = loss_fn(batch, out)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    assert losses[-1] < 0.3 * losses[0]
