"""Preprocess every data set of an excited-state foundation model and write the
training configuration.

    python foundation/prepare_datasets.py --datasets datasets.yaml \
        --template foundation/train_template.yaml --work_dir work

For every data set in datasets.yaml this runs mace_prepare_data (sharded HDF5,
per-data-set statistics: E0s, per-state shifts, scale, average number of
neighbours) into <work_dir>/h5/<name>/, then writes <work_dir>/train_config.yaml:
the template plus one head per data set. Each head keeps its own number of states,
multiplicities and couplings; heads with different layouts share batches (their
labels are padded as they are read).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

# options of a data set passed on to mace_prepare_data and kept on its head
LABEL_OPTIONS = (
    "n_energies",
    "states",
    "soc_mode",
    "compute_nacs",
    "compute_socs",
    "compute_dipoles",
    "compute_oscillator_strengths",
    "nac_num",
    "soc_num",
    "dipole_num",
    "osc_num",
    "nacs_format",
    "energy_key",
    "forces_key",
    "nacs_key",
    "socs_key",
    "dipoles_key",
    "oscillator_strengths_key",
)
# options that only concern the head in training
HEAD_OPTIONS = ("weight", "config_type_weights")
FLAGS = (
    "compute_nacs",
    "compute_socs",
    "compute_dipoles",
    "compute_oscillator_strengths",
)


def _cli_value(value) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


def prepare_dataset(name: str, spec: dict, common: dict, work_dir: Path) -> dict:
    prefix = work_dir / "h5" / name
    prefix.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        "-m",
        "mace.cli.preprocess_data",
        f"--train_file={spec['train_file']}",
        f"--h5_prefix={prefix}/",
        f"--r_max={common['r_max']}",
        f"--atomic_numbers={common['elements']}",
        f"--num_process={spec.get('num_process', common.get('num_process', 1))}",
        f"--seed={common.get('seed', 1)}",
        f"--E0s={spec.get('E0s', 'average')}",
        f"--scaling={common.get('scaling', 'rms_forces_scaling')}",
        "--compute_statistics",
    ]
    if spec.get("valid_file"):
        argv.append(f"--valid_file={spec['valid_file']}")
    else:
        argv.append(
            f"--valid_fraction={spec.get('valid_fraction', common.get('valid_fraction', 0.05))}"
        )
    for key in LABEL_OPTIONS:
        if key not in spec:
            continue
        if key in FLAGS:
            if spec[key]:
                argv.append(f"--{key}")
        else:
            argv.append(f"--{key}={_cli_value(spec[key])}")
    print(f">> {name}: mace_prepare_data -> {prefix}", flush=True)
    subprocess.run(argv, check=True)

    head = {
        "train_file": str(prefix / "train"),
        "valid_file": str(prefix / "val"),
        "statistics_file": str(prefix / "statistics.json"),
    }
    if spec.get("test_file"):
        head["test_file"] = str(spec["test_file"])
    for key in LABEL_OPTIONS + HEAD_OPTIONS:
        if key in spec:
            head[key] = spec[key]
    if "n_energies" not in head and "states" not in head:
        head["n_energies"] = 1
    return head


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", required=True, help="datasets.yaml")
    parser.add_argument("--template", required=True, help="training config template")
    parser.add_argument("--work_dir", required=True)
    parser.add_argument(
        "--only", default=None, help="comma-separated data sets to (re)process"
    )
    args = parser.parse_args(argv)

    spec = yaml.safe_load(Path(args.datasets).read_text(encoding="utf-8"))
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    common = {k: v for k, v in spec.items() if k != "datasets"}
    if "elements" not in common or "r_max" not in common:
        raise SystemExit("datasets.yaml needs 'elements' and 'r_max'")
    only = set(args.only.split(",")) if args.only else None

    heads = {}
    for name, dataset in spec["datasets"].items():
        prefix = work_dir / "h5" / name
        if (
            only is not None
            and name not in only
            and (prefix / "statistics.json").exists()
        ):
            print(f">> {name}: already prepared", flush=True)
            head = {
                "train_file": str(prefix / "train"),
                "valid_file": str(prefix / "val"),
                "statistics_file": str(prefix / "statistics.json"),
            }
            for key in LABEL_OPTIONS + HEAD_OPTIONS:
                if key in dataset:
                    head[key] = dataset[key]
            if dataset.get("test_file"):
                head["test_file"] = str(dataset["test_file"])
            heads[name] = head
            continue
        heads[name] = prepare_dataset(name, dataset, common, work_dir)

    config = yaml.safe_load(Path(args.template).read_text(encoding="utf-8")) or {}
    config["heads"] = heads
    config["atomic_numbers"] = str(list(common["elements"]))
    config["r_max"] = common["r_max"]
    config.setdefault("work_dir", str(work_dir / "run"))
    out = work_dir / "train_config.yaml"
    out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    summary = {
        name: json.loads(Path(head["statistics_file"]).read_text())["avg_num_neighbors"]
        for name, head in heads.items()
    }
    print(f">> wrote {out}; average number of neighbours per data set: {summary}")


if __name__ == "__main__":
    main()
