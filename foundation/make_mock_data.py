"""Small mock data sets with the label layouts of a real excited-state foundation
model, for a dry run of the whole pipeline on a laptop or a login node.

    python foundation/make_mock_data.py mock/

writes mock/<name>/train.xyz and test.xyz for four data sets (singlets + triplets
with SOCs, singlets with NACs, energies only, doublets + quartet) and
mock/datasets.yaml. The labels are synthetic (smooth functions of the geometry),
only their shapes and keys matter here.
"""

import sys
from pathlib import Path

import ase.io
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

# pylint: disable=wrong-import-position,import-error
from excited_data import (  # noqa: E402
    make_excited_configs,
    make_multiplicity_configs,
)

DATASETS = {
    "sharc_st": dict(
        states=[2, 0, 2],
        compute_nacs=True,
        compute_socs=True,
        compute_dipoles=True,
        symbols=["C", "H", "H", "O"],
        charge=0,
        spin=0,
    ),
    "tddft_singlets": dict(
        n_energies=4,
        compute_nacs=True,
        compute_dipoles=True,
        symbols=["C", "N", "H", "H"],
        charge=0,
        spin=0,
    ),
    "public_energies": dict(
        n_energies=3,
        symbols=["C", "H", "H", "S"],
        charge=1,
        spin=1,
        weight=0.5,
        energies_only=True,
    ),
    "radicals": dict(
        states=[0, 2, 0, 1],
        compute_nacs=True,
        compute_socs=True,
        symbols=["C", "H", "H", "O"],
        charge=0,
        spin=1,
    ),
}


def _frames(name, spec, n, seed):
    if "states" in spec:
        frames = make_multiplicity_configs(n, spec["states"], seed=seed)
        if not spec.get("compute_dipoles"):
            for atoms in frames:
                atoms.info.pop("REF_dipoles", None)
    else:
        couplings = not spec.get("energies_only", False)
        frames = make_excited_configs(
            n,
            spec["n_energies"],
            seed=seed,
            with_nacs=couplings,
            with_socs=False,
            with_dipoles=spec.get("compute_dipoles", False),
        )
    for atoms in frames:
        atoms.set_chemical_symbols(spec["symbols"])
        atoms.info["total_charge"] = spec["charge"]
        atoms.info["total_spin"] = spec["spin"]
        if spec.get("energies_only"):
            atoms.info.pop("REF_forces", None)
    return frames


def main(out: str) -> None:
    out = Path(out).resolve()
    datasets = {}
    for i, (name, spec) in enumerate(DATASETS.items()):
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        ase.io.write(folder / "train.xyz", _frames(name, spec, 40, seed=10 + i))
        ase.io.write(folder / "test.xyz", _frames(name, spec, 6, seed=50 + i))
        entry = {
            "train_file": str(folder / "train.xyz"),
            "test_file": str(folder / "test.xyz"),
        }
        for key in (
            "states",
            "n_energies",
            "compute_nacs",
            "compute_socs",
            "compute_dipoles",
            "weight",
        ):
            if key in spec:
                entry[key] = spec[key]
        datasets[name] = entry
    config = {
        "elements": [1, 6, 7, 8, 16],
        "r_max": 4.0,
        "valid_fraction": 0.15,
        "num_process": 2,
        "seed": 1,
        "datasets": datasets,
    }
    (out / "datasets.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"mock data sets in {out}: {', '.join(datasets)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mock")
