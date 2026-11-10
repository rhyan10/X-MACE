#!/usr/bin/env python
"""Oscillator strengths from a committee of X-MACE models, with uncertainties.

    python osc_committee.py --model "runs_osc/*/*_stagetwo.model" \
        --configs osc_data/test.xyz --out pred_osc.xyz

Writes every frame with MACE_oscillator_strengths [O] (committee mean) and
MACE_oscillator_strengths_std [O] (committee spread). If the frames carry
reference values (--ref_key, default REF_oscillator_strengths), prints the errors
per transition and how well the committee spread predicts them. Frames are
evaluated in batches without forces, so large test sets are quick.
"""

import argparse
import glob
import warnings

import numpy as np
import torch
from ase.io import read, write

from mace import data
from mace.calculators import ExcitedMACECalculator
from mace.tools import torch_geometric, torch_tools


def expand(patterns):
    """Model files matching the patterns; a TorchScript copy (*_compiled.model) is
    dropped when its original is there too, so no member counts twice."""
    paths = sorted(
        {p for pattern in patterns for p in (glob.glob(pattern) or [pattern])}
    )
    paths = [
        p
        for p in paths
        if not (
            p.endswith("_compiled.model")
            and p.replace("_compiled.model", ".model") in paths
        )
    ]
    if not paths:
        raise FileNotFoundError(f"no model files match {patterns}")
    return paths


def committee_predictions(calc, frames, batch_size):
    """[n_models, n_frames, O] oscillator strengths of every committee member."""
    configs = [
        data.config_from_atoms(atoms, key_specification=data.KeySpecification())
        for atoms in frames
    ]
    dtype = torch.float64 if calc.default_dtype == "float64" else torch.float32
    with torch_tools.default_dtype(dtype):
        dataset = [
            data.AtomicData.from_config(c, z_table=calc.z_table, cutoff=calc.r_max)
            for c in configs
        ]
    loader = torch_geometric.dataloader.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    n_osc = calc.head_view.osc_num
    if n_osc == 0:
        raise SystemExit("these models do not predict oscillator strengths")
    results = []
    for model in calc.models:
        values = []
        for batch in loader:
            batch = batch.to(calc.device)
            inputs = batch.to_dict()
            inputs["head"] = torch.full(
                (batch.num_graphs,), calc._head_indices(model), device=calc.device
            )
            out = model(inputs, training=False, compute_force=False)
            values.append(torch_tools.to_numpy(out["oscillator_strengths"])[:, :n_osc])
        results.append(np.concatenate(values))
    return np.stack(results)


def report(pred, std, ref):
    error = pred - ref
    n_osc = ref.shape[1]
    print(
        f"{'transition':>11} {'MAE':>8} {'RMSE':>8} {'R2':>6} {'mean ref':>9} "
        f"{'mean std':>9} {'|err|<2std':>10}"
    )
    for k in range(n_osc):
        e, r = error[:, k], ref[:, k]
        r2 = 1 - np.sum(e**2) / max(np.sum((r - r.mean()) ** 2), 1e-30)
        within = np.mean(np.abs(e) < 2 * std[:, k]) if std is not None else np.nan
        print(
            f"  S0 -> S{k + 1:<3} {np.abs(e).mean():8.4f} {np.sqrt((e**2).mean()):8.4f} "
            f"{r2:6.3f} {r.mean():9.4f} "
            f"{(std[:, k].mean() if std is not None else np.nan):9.4f} {within:10.2f}"
        )
    print(f"{'all':>11} {np.abs(error).mean():8.4f} {np.sqrt((error**2).mean()):8.4f}")
    if std is not None:
        rho = np.corrcoef(std.reshape(-1), np.abs(error).reshape(-1))[0, 1]
        print(
            f"correlation of committee std with |error|: {rho:.2f} "
            "(|err|<2std: share of values within twice the spread; ~0.95 if calibrated)"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model", nargs="+", required=True, help="files or glob patterns"
    )
    parser.add_argument("--configs", required=True)
    parser.add_argument("--out", default="pred_osc.xyz")
    parser.add_argument("--ref_key", default="REF_oscillator_strengths")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    paths = expand(args.model)
    print(f"committee of {len(paths)} model(s): {paths}")
    calc = ExcitedMACECalculator(paths, device=args.device, default_dtype=args.dtype)
    frames = read(args.configs, ":")
    members = committee_predictions(calc, frames, args.batch_size)
    mean = members.mean(axis=0)
    std = members.std(axis=0) if len(paths) > 1 else None
    for k, atoms in enumerate(frames):
        atoms.calc = None
        atoms.info["MACE_oscillator_strengths"] = mean[k]
        if std is not None:
            atoms.info["MACE_oscillator_strengths_std"] = std[k]
    write(args.out, frames)
    print(f"{len(frames)} frames with predictions written to {args.out}")

    if all(args.ref_key in atoms.info for atoms in frames):
        ref = np.array([np.reshape(atoms.info[args.ref_key], -1) for atoms in frames])
        report(mean, std, ref[:, : mean.shape[1]])


if __name__ == "__main__":
    main()
