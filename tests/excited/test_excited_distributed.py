"""Data-parallel (DDP) training of excited-state models.

Every rank gets part of the batch; DDP averages the gradients of the ranks. The tests
run two CPU processes (gloo) and check that the averaged gradient is the gradient of
the loss of the whole batch on one device, for every loss term (also the RMSE-type
coupling losses), every model type, mixed heads and state-specific charges, and that
the ranks stay identical after optimizer steps."""

import socket
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).parent))

# pylint: disable=wrong-import-position,import-error
from conftest import atoms_to_batch, build_excited, full_spec  # noqa: E402
from excited_data import make_excited_configs  # noqa: E402

from mace.modules import WeightedExcitedStatesLoss  # noqa: E402
from mace.modules.excited import (  # noqa: E402
    AutoencoderExcitedMACE,
    ExcitedMACE,
    excited_head_layout,
)

WORLD = 2
N_FRAMES = 8  # 4 per rank

CASES = {
    "excited": dict(),
    "autoencoder": dict(autoencoder=True),
    "state_charges": dict(long_range_mode="state"),
    "mixed_heads": dict(heads=True),
    "fixed_parity": dict(coupling_parity="fixed"),
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _model(case: str) -> torch.nn.Module:
    torch.manual_seed(7)
    options = dict(CASES[case])
    if options.pop("heads", False):
        from conftest import model_config  # pylint: disable=import-outside-toplevel

        from mace import modules  # pylint: disable=import-outside-toplevel

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
            for readout in list(model.readouts):
                last = getattr(readout, "linear_2", None) or readout.linear
                last.weight.normal_(0.0, 0.3)
        return model
    model = build_excited(**options)
    if options.get("long_range_mode") == "state":
        with torch.no_grad():
            for readout in model.charge_readouts:
                last = getattr(readout, "linear_2", None) or readout.linear
                last.weight.normal_(0.0, 0.3)
    return model


def _loss(model) -> WeightedExcitedStatesLoss:
    return WeightedExcitedStatesLoss(
        energy_weight=1.0,
        forces_weight=10.0,
        nacs_weight=10.0,
        socs_weight=10.0,
        dipoles_weight=1.0,
        n_permanent_dipoles=3,
        masks=model.mask_tables(),
    )


def _batch(case: str, frames: slice):
    atoms = make_excited_configs(N_FRAMES, 3, seed=11, random_phases=True)
    if case == "mixed_heads":
        # head 0 (ground) on rank 0's frames, head 1 (excited) on rank 1's: one
        # rank sees ground-state frames only
        heads = torch.tensor([0] * (N_FRAMES // 2) + [1] * (N_FRAMES // 2))
    else:
        heads = torch.zeros(N_FRAMES, dtype=torch.long)
    batch = atoms_to_batch(atoms[frames], full_spec())
    batch.head = heads[frames]
    return batch


def _gradients(model) -> dict:
    return {
        name: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def _worker(rank: int, port: int, case: str, out: str) -> None:
    import os  # pylint: disable=import-outside-toplevel

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    try:
        model = _model(case)
        loss_fn = _loss(model)
        ddp_model = DDP(model)
        per_rank = N_FRAMES // WORLD
        batch = _batch(case, slice(rank * per_rank, (rank + 1) * per_rank))
        out_dict = ddp_model(batch.to_dict(), training=True)
        loss = loss_fn(batch, out_dict, ddp=True)
        loss.backward()
        gradients = _gradients(model)
        losses = torch.tensor([float(loss)])
        dist.all_reduce(losses)
        # a few optimizer steps: the ranks must stay identical
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(3):
            optimizer.zero_grad()
            out_dict = ddp_model(batch.to_dict(), training=True)
            loss_fn(batch, out_dict, ddp=True).backward()
            optimizer.step()
        flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
        gathered = [torch.zeros_like(flat) for _ in range(WORLD)]
        dist.all_gather(gathered, flat)
        if rank == 0:
            torch.save(
                {
                    "gradients": gradients,
                    "loss": float(losses[0]) / WORLD,
                    "max_param_diff": float((gathered[0] - gathered[1]).abs().max()),
                },
                out,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.parametrize("case", list(CASES))
def test_ddp_gradients_equal_whole_batch_gradients(tmp_path, case):
    out = str(tmp_path / "ddp.pt")
    mp.spawn(_worker, args=(_free_port(), case, out), nprocs=WORLD, join=True)
    result = torch.load(out, weights_only=False)

    model = _model(case)
    loss_fn = _loss(model)
    batch = _batch(case, slice(0, N_FRAMES))
    loss = loss_fn(batch, model(batch.to_dict(), training=True), ddp=False)
    loss.backward()
    reference = _gradients(model)

    # the mean of the ranks' losses is the loss of the whole batch
    np.testing.assert_allclose(result["loss"], float(loss), rtol=1e-9)
    assert set(result["gradients"]) == set(reference)
    scale = max(float(g.abs().max()) for g in reference.values())
    for name, grad in reference.items():
        torch.testing.assert_close(
            result["gradients"][name], grad, rtol=1e-7, atol=1e-10 * scale, msg=name
        )
    assert result["max_param_diff"] == 0.0


def test_masked_rmse_without_ddp_is_plain_rmse():
    from mace.modules.excited_loss import (  # pylint: disable=import-outside-toplevel
        masked_rmse,
    )

    values = torch.rand(5, 3)
    mask = torch.ones(5, 3)
    torch.testing.assert_close(
        masked_rmse(values, mask, ddp=False), torch.sqrt(values.mean() + 1e-12)
    )
    assert AutoencoderExcitedMACE is not None  # both models are exercised above
