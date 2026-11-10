#!/usr/bin/env python
"""Convert an XYZ file with oscillator strengths stored under REF_energy and split it
into training, validation and test sets.

    python prepare_osc_data.py pubchemSVP_osc.xyz --out_dir osc_data

writes osc_data/train.xyz, valid.xyz, test.xyz (default 90 / 5 / 5 %, random,
seeded) with the values under REF_oscillator_strengths and no REF_energy, and
osc_data/e0s.txt, an --E0s dictionary with every element of the data set (all 0.0:
energies are not trained). Frames with broken geometries (two atoms closer than
--min_distance, default 0.6 A) or unusable labels (not finite, below -0.01 or above
--max_f) are dropped and listed in osc_data/dropped.txt; slightly negative oscillator
strengths (numerical noise, down to -0.01) are set to 0. Reads the input twice, one frame at a time, so large
files are fine; copes with a UTF-8 byte-order mark, Windows line endings and an
incomplete last frame (skipped).
"""

import argparse
import io
import sys
from pathlib import Path

import ase.io
import numpy as np
from ase.data import atomic_numbers


def frames(path, warn=True):
    """(number, text) of every complete frame of an XYZ file."""
    with open(path, encoding="utf-8-sig", newline=None) as handle:  # strips the BOM
        number = 0
        while True:
            header = handle.readline()
            if not header:
                return
            if not header.strip():
                continue
            try:
                n_atoms = int(header.split()[0])
            except ValueError:
                sys.exit(
                    f"frame {number + 1}: expected the number of atoms, got {header!r}"
                )
            lines = [header, handle.readline()]
            for _ in range(n_atoms):
                line = handle.readline()
                if not line:
                    if not warn:
                        return
                    print(
                        f"warning: frame {number + 1} is incomplete "
                        f"({len(lines) - 2} of {n_atoms} atoms), skipped",
                        file=sys.stderr,
                    )
                    return
                lines.append(line)
            number += 1
            yield number, "".join(line.rstrip("\r\n") + "\n" for line in lines)


def problem(atoms, strengths, min_distance, max_f):
    """Why a frame cannot be used (None if it can)."""
    if not np.all(np.isfinite(atoms.positions)):
        return "positions not finite"
    if not np.all(np.isfinite(strengths)):
        return "oscillator strengths not finite"
    if strengths.size and strengths.max() > max_f:
        return f"oscillator strength {strengths.max():.3g} > {max_f}"
    if strengths.size and strengths.min() < -0.01:
        return f"oscillator strength {strengths.min():.3g} < -0.01"
    if len(atoms) > 1:
        distances = atoms.get_all_distances()
        np.fill_diagonal(distances, np.inf)
        if distances.min() < min_distance:
            i, j = np.unravel_index(np.argmin(distances), distances.shape)
            symbols = atoms.get_chemical_symbols()
            return (
                f"atoms {i} {symbols[i]} and {j} {symbols[j]} only "
                f"{distances.min():.3f} A apart"
            )
    return None


def convert(text, number, from_key, to_key):
    atoms = ase.io.read(io.StringIO(text), format="extxyz")
    if from_key not in atoms.info:
        sys.exit(f"frame {number}: no {from_key} in its info line")
    strengths = np.asarray(atoms.info.pop(from_key), dtype=float).reshape(-1)
    clean = ase.Atoms(
        atoms.get_chemical_symbols(),
        positions=atoms.positions,
        cell=atoms.cell,
        pbc=atoms.pbc,
    )
    clean.info.update(atoms.info)  # other info entries are kept
    clean.info[to_key] = strengths
    return clean, strengths


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input")
    parser.add_argument("--out_dir", default="osc_data")
    parser.add_argument("--valid_fraction", type=float, default=0.05)
    parser.add_argument("--test_fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--from_key", default="REF_energy")
    parser.add_argument("--to_key", default="REF_oscillator_strengths")
    parser.add_argument(
        "--min_distance",
        type=float,
        default=0.6,
        help="drop frames with atoms closer (A)",
    )
    parser.add_argument(
        "--max_f", type=float, default=10.0, help="drop frames with larger values"
    )
    args = parser.parse_args()

    # pass 1: check every frame; the usable ones are split
    dropped = []
    usable = []
    for number, text in frames(args.input, warn=False):
        atoms, strengths = convert(text, number, args.from_key, args.to_key)
        reason = problem(atoms, strengths, args.min_distance, args.max_f)
        if reason is None:
            usable.append(number)
        else:
            dropped.append((number, reason))
    n_frames = len(usable)
    if n_frames == 0:
        sys.exit("no usable frames found")
    position = {number: k for k, number in enumerate(usable)}
    n_valid = int(round(args.valid_fraction * n_frames))
    n_test = int(round(args.test_fraction * n_frames))
    if n_valid + n_test >= n_frames:
        sys.exit(f"{n_frames} frames are too few for this split")
    order = np.random.default_rng(args.seed).permutation(n_frames)
    split = np.zeros(n_frames, dtype=np.int8)  # 0 train, 1 valid, 2 test
    split[order[:n_valid]] = 1
    split[order[n_valid : n_valid + n_test]] = 2

    # pass 2: convert and write
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ("train", "valid", "test")
    files = [open(out_dir / f"{name}.xyz", "w", encoding="utf-8") for name in names]
    counts = [0, 0, 0]
    elements = set()
    sizes = set()
    sums, maxima, negative = None, None, 0
    try:
        for number, text in frames(args.input):
            if number not in position:
                continue
            atoms, strengths = convert(text, number, args.from_key, args.to_key)
            strengths = np.clip(strengths, 0.0, None)  # noise below 0
            atoms.info[args.to_key] = strengths
            part = int(split[position[number]])
            ase.io.write(files[part], atoms, format="extxyz")
            counts[part] += 1
            elements.update(atoms.get_chemical_symbols())
            sizes.add(strengths.size)
            if sums is None or strengths.size != sums.size:
                if sums is None:
                    sums, maxima = np.zeros(strengths.size), np.zeros(strengths.size)
                else:
                    continue
            sums += strengths
            maxima = np.maximum(maxima, strengths)
            negative += int((strengths < 0).sum())
    finally:
        for handle in files:
            handle.close()

    zs = sorted(atomic_numbers[symbol] for symbol in elements)
    e0s = "{" + ", ".join(f"{z}: 0.0" for z in zs) + "}"
    (out_dir / "e0s.txt").write_text(e0s + "\n")
    n_osc = max(sizes)
    (out_dir / "n_osc.txt").write_text(f"{n_osc}\n")

    with open(out_dir / "dropped.txt", "w", encoding="utf-8") as handle:
        for number, reason in dropped:
            handle.write(f"frame {number}: {reason}\n")
    if dropped:
        print(f"dropped {len(dropped)} frames (listed in {out_dir}/dropped.txt), e.g.:")
        for number, reason in dropped[:5]:
            print(f"  frame {number}: {reason}")
    print(
        f"{n_frames} usable frames: "
        + ", ".join(f"{name} {count}" for name, count in zip(names, counts))
        + f" -> {out_dir}/"
    )
    print(f"{args.to_key}: {n_osc} values per frame (use --n_energies={n_osc + 1})")
    print("mean per transition:", np.round(sums / max(sum(counts), 1), 4))
    print("max per transition: ", np.round(maxima, 4))
    print(f"elements: {' '.join(sorted(elements, key=lambda s: atomic_numbers[s]))}")
    print(f'--E0s="{e0s}" (also in {out_dir}/e0s.txt)')
    if len(sizes) > 1:
        print(
            f"warning: frames have different numbers of values: {sorted(sizes)}",
            file=sys.stderr,
        )
    if negative:
        print(f"warning: {negative} negative values", file=sys.stderr)


if __name__ == "__main__":
    main()
