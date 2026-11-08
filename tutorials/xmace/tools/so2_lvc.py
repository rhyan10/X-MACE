"""Helpers for the X-MACE tutorials: a reference model of SO2 and SHARC runs.

The reference is SHARC's linear vibronic coupling (LVC) model of SO2, shipped with
SHARC 4 in examples/SHARC_LVC (4 singlets and 3 triplets, with spin-orbit couplings
and dipoles). It is cheap, so it can stand in for an ab initio method: we sample
geometries, compute "reference" energies, gradients, NACs, SOCs and dipoles with it,
train X-MACE on them and compare dynamics.

SHARC's Python modules have generic names (utils, constants, logger, ...), so they
are only ever imported in a separate process. This file is that worker too:

    python so2_lvc.py lvc  WORKDIR GEOMETRIES.npz OUT.npz

Everything else (sampling, files, parsing) runs in the notebook's process.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903
AMU_TO_ME = 1822.888486209
FS_TO_AU = 41.341373335
STATES = [4, 0, 3]  # SHARC: 4 singlets, 0 doublets, 3 triplets
SYMBOLS = ["S", "O", "O"]


# --------------------------------------------------------------------------------------
# SHARC installation
# --------------------------------------------------------------------------------------


def sharc_root() -> Optional[Path]:
    """Root of the SHARC installation ($SHARC is its bin directory), or None."""
    sharc = os.environ.get("SHARC")
    if not sharc or not Path(sharc, "sharc.x").is_file():
        return None
    return Path(sharc).resolve().parent


def sharc_env(extra_paths: Sequence[str] = ()) -> Dict[str, str]:
    """Environment for SHARC subprocesses (as sharcvars.sh sets it up)."""
    root = sharc_root()
    if root is None:
        raise RuntimeError("SHARC not found: set $SHARC to SHARC's bin directory")
    env = dict(os.environ)
    paths = [*extra_paths, str(root / "lib"), str(root / "bin"), str(root / "pysharc")]
    env["PYTHONPATH"] = os.pathsep.join(paths + [env.get("PYTHONPATH", "")])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(root / "lib"), env.get("LD_LIBRARY_PATH", "")]
    )
    env.setdefault("HOSTNAME", "localhost")
    return env


# --------------------------------------------------------------------------------------
# The LVC model
# --------------------------------------------------------------------------------------


def read_v0(path: Path) -> Dict[str, np.ndarray]:
    """Reference geometry (bohr), masses (amu), frequencies (Hartree) and mass-weighted
    normal modes (columns) from an LVC V0.txt file."""
    lines = Path(path).read_text().splitlines()
    i = lines.index("Geometry") + 1
    symbols, coords, masses = [], [], []
    while lines[i].strip() != "Frequencies":
        parts = lines[i].split()
        symbols.append(parts[0])
        coords.append([float(x) for x in parts[2:5]])
        masses.append(float(parts[5]))
        i += 1
    frequencies = np.array(lines[i + 1].split(), dtype=float)
    n = 3 * len(symbols)
    start = lines.index("Mass-weighted normal modes") + 1
    modes = np.array([lines[start + r].split() for r in range(n)], dtype=float)
    return {
        "symbols": symbols,
        "coords": np.array(coords),
        "masses": np.array(masses),
        "frequencies": frequencies,
        "modes": modes,
    }


def _matrix_block(lines: List[str], header: str, n: int) -> Optional[np.ndarray]:
    for i, line in enumerate(lines):
        if line.split() == header.split():
            return np.array([lines[i + 1 + r].split() for r in range(n)], dtype=float)
    return None


def make_lvc_model(directory: Path) -> Dict[str, float]:
    """Copy SHARC's SO2 LVC model into `directory`, with spin-orbit couplings that obey
    the Wigner-Eckart theorem.

    The SOC matrix of SHARC's example template is real and symmetric and does not
    have the Wigner-Eckart form of real spin-free states: for instance its
    singlet-triplet M_S = 0 elements are real, but they must be purely imaginary
    (i a_z). We fit one coupling vector per pair of states to it (least squares) and
    rescale each vector so the pair keeps its coupling strength (the Frobenius norm of
    its block). Returns the relative residual of that fit for the original matrix."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from mace.modules.spin_states import (  # pylint: disable=import-outside-toplevel
        ms_resolved_map,
        n_mstates,
        soc_matrix_from_vectors,
        soc_vectors_from_matrix,
        spin_orbit_pairs,
    )

    root = sharc_root()
    if root is None:
        raise RuntimeError("SHARC not found: set $SHARC to SHARC's bin directory")
    example = root / "examples" / "SHARC_LVC"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(example / "V0.txt", directory / "V0.txt")
    lines = (example / "LVC.template").read_text().splitlines()
    nm = n_mstates(STATES)
    soc_real = _matrix_block(lines, "SOC R", nm)
    soc_imag = _matrix_block(lines, "SOC I", nm)
    original = soc_real + 1j * (soc_imag if soc_imag is not None else 0.0)
    vectors, residual = soc_vectors_from_matrix(original, STATES)
    owner = np.zeros(nm, dtype=int)
    for k, components in enumerate(ms_resolved_map(STATES)):
        for index, _ in components:
            owner[index] = k
    for p, (i, j) in enumerate(spin_orbit_pairs(STATES)):
        block = original[np.ix_(owner == i, owner == j)]
        unit = soc_matrix_from_vectors(
            np.eye(len(vectors))[p][:, None] * vectors[p][None, :], STATES
        )[np.ix_(owner == i, owner == j)]
        norm_unit = np.linalg.norm(unit)
        if norm_unit > 0:
            vectors[p] *= np.linalg.norm(block) / norm_unit
    soc = soc_matrix_from_vectors(vectors, STATES)

    out, skip = [], 0
    for line in lines:
        if skip:
            skip -= 1
            continue
        if line.split() in (["SOC", "R"], ["SOC", "I"]):
            skip = nm
            continue
        out.append(line)
    for name, part in (("SOC R", soc.real), ("SOC I", soc.imag)):
        out.append(name)
        out += ["".join(f"{x:16.8e}" for x in row) for row in part]
    (directory / "LVC.template").write_text("\n".join(out) + "\n")
    (directory / "LVC.resources").write_text("do_kabsch true\n")
    return {"original_residual": float(residual)}


# --------------------------------------------------------------------------------------
# Initial conditions
# --------------------------------------------------------------------------------------


def wigner_sample(
    v0: Dict[str, np.ndarray], n: int, seed: int = 0, spread: float = 1.0
) -> Dict[str, np.ndarray]:
    """Geometries (bohr) and velocities (bohr / a.u. time) from the harmonic Wigner
    distribution of the vibrational ground state.

    For a mode of angular frequency w (a.u.), the mass-weighted coordinate Q and
    momentum P are Gaussian with variances 1 / (2 w) and w / 2. `spread` > 1 widens
    the geometry distribution, to sample further from the minimum."""
    rng = np.random.default_rng(seed)
    masses = np.repeat(v0["masses"] * AMU_TO_ME, 3)  # per Cartesian coordinate
    vibrations = v0["frequencies"] > 1e-6
    w = v0["frequencies"][vibrations]
    modes = v0["modes"][:, vibrations]
    q = rng.normal(size=(n, len(w))) * np.sqrt(1.0 / (2.0 * w)) * spread
    p = rng.normal(size=(n, len(w))) * np.sqrt(w / 2.0)
    displacement = (q @ modes.T) / np.sqrt(masses)
    velocities = (p @ modes.T) / np.sqrt(masses)
    n_atoms = len(v0["symbols"])
    return {
        "coords": v0["coords"][None] + displacement.reshape(n, n_atoms, 3),
        "velocities": velocities.reshape(n, n_atoms, 3),
    }


# --------------------------------------------------------------------------------------
# Reference calculations
# --------------------------------------------------------------------------------------


def lvc_reference(
    model_dir: Path, coords_bohr: np.ndarray, symbols: Sequence[str] = SYMBOLS
) -> Dict[str, np.ndarray]:
    """h, grad, nacdr and dm (SHARC's atomic units, M_S resolved) for every geometry,
    computed by SHARC's LVC interface in a separate process."""
    model_dir = Path(model_dir).resolve()
    work = model_dir / "calc"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    for name in ("V0.txt", "LVC.template", "LVC.resources"):
        shutil.copy(model_dir / name, work / name)
    np.savez(work / "geometries.npz", coords=np.asarray(coords_bohr), symbols=symbols)
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "lvc",
            str(work),
            "geometries.npz",
            "results.npz",
        ],
        env=sharc_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout[-2000:] + result.stderr[-2000:])
    stored = np.load(work / "results.npz")
    return {key: stored[key] for key in stored.files}


def _lvc_worker(work: Path, geometries: str, output: str) -> None:
    """Runs inside SHARC's environment: the LVC interface stays in memory, as in
    pysharc, and is called once per geometry."""
    os.chdir(work)
    from SHARC_LVC import (
        SHARC_LVC,
    )  # pylint: disable=import-outside-toplevel,import-error

    data = np.load(geometries)
    coords, symbols = data["coords"], [str(s) for s in data["symbols"]]
    states = " ".join(map(str, STATES))

    def write_qmin(step, xyz):
        lines = [str(len(symbols)), ""]
        lines += [
            f"{s} {x:.12f} {y:.12f} {z:.12f}" for s, (x, y, z) in zip(symbols, xyz)
        ]
        lines += [
            "unit bohr",
            f"states {states}",
            "charge 0 1 0",
            "savedir ./save",
            f"step {step}",
            "h",
            "soc",
            "dm",
            "grad",
            "nacdr",
        ]
        Path("QM.in").write_text("\n".join(lines) + "\n")

    interface = SHARC_LVC(persistent=True, loglevel=40)
    write_qmin(0, coords[0])
    interface.setup_mol("QM.in")
    interface.read_resources("LVC.resources")
    interface.read_template("LVC.template")
    interface.setup_interface()
    out = {"h": [], "grad": [], "nacdr": [], "dm": []}
    for step, xyz in enumerate(coords):
        write_qmin(step, xyz)
        interface.read_requests("QM.in")
        interface.set_coords("QM.in")
        interface.run()
        qmout = interface.getQMout()
        interface.write_step_file()
        for key in out:
            out[key].append(np.array(qmout[key]))
    np.savez(output, **{key: np.array(value) for key, value in out.items()})


# --------------------------------------------------------------------------------------
# SHARC trajectories
# --------------------------------------------------------------------------------------


def write_trajectory(
    directory: Path,
    coords_bohr: np.ndarray,
    velocities: np.ndarray,
    masses_amu: Sequence[float],
    interface_files: Dict[str, str],
    tmax: float = 50.0,
    stepsize: float = 0.5,
    state: int = 2,
    seed: int = 1,
    symbols: Sequence[str] = SYMBOLS,
    states: Sequence[int] = STATES,
) -> Path:
    """A SHARC trajectory directory: input, geom, veloc and QM/ with the interface's
    files (name -> text). Surface hopping in SHARC's diagonal representation with
    NACs, spin-orbit couplings and energy-based decoherence."""
    numbers = {"S": 16.0, "O": 8.0, "C": 6.0, "H": 1.0, "N": 7.0}
    directory = Path(directory)
    if directory.exists():
        shutil.rmtree(directory)
    (directory / "QM").mkdir(parents=True)
    (directory / "geom").write_text(
        "".join(
            f"{s:>3} {numbers[s]:5.1f} {x:16.10f} {y:16.10f} {z:16.10f} {m:14.8f}\n"
            for s, (x, y, z), m in zip(symbols, coords_bohr, masses_amu)
        )
    )
    (directory / "veloc").write_text(
        "".join(f"{a:16.10f} {b:16.10f} {c:16.10f}\n" for a, b, c in velocities)
    )
    layout = " ".join(map(str, states))
    charges = " ".join("0" for _ in states)
    (directory / "input").write_text(f"""geomfile "geom"
veloc external
velocfile "veloc"

nstates {layout}
charge {charges}
actstates {layout}
state {state} mch
coeff auto
rngseed {seed}

ezero 0.0
tmax {tmax}
stepsize {stepsize}
nsubsteps 25

surf sharc
coupling nacdr
gradcorrect
ekincorrect parallel_vel
decoherence_scheme edc
nophases_from_interface
output_format ascii
""")
    for name, text in interface_files.items():
        (directory / "QM" / name).write_text(text)
    return directory


def run_pysharc(directory: Path, interface: str, extra_paths: Sequence[str] = ()):
    """Run a trajectory with pysharc ($SHARC/driver.py -i <interface> input)."""
    root = sharc_root()
    result = subprocess.run(
        [sys.executable, str(root / "bin" / "driver.py"), "-i", interface, "input"],
        cwd=directory,
        env=sharc_env(extra_paths),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        log = Path(directory, "output.log")
        tail = log.read_text()[-1500:] if log.exists() else ""
        raise RuntimeError(f"{directory}: {tail}\n{result.stderr[-1500:]}")
    return result


def read_output_lis(directory: Path) -> Dict[str, np.ndarray]:
    """Time (fs), active state (diagonal and MCH, 1-based, M_S resolved) and kinetic,
    potential and total energy (eV) from SHARC's output.lis."""
    rows = [
        line.split()
        for line in Path(directory, "output.lis").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    data = np.array([[float(x) for x in row[1:7]] for row in rows])
    return {
        "time": data[:, 0],
        "state_diag": data[:, 1].astype(int),
        "state_mch": data[:, 2].astype(int),
        "ekin": data[:, 3],
        "epot": data[:, 4],
        "etot": data[:, 5],
    }


def mch_multiplicity(states: Sequence[int] = STATES) -> np.ndarray:
    """Multiplicity of every M_S-resolved MCH state (index 0 = SHARC state 1)."""
    out = []
    for m, n in enumerate(states):
        out += [m + 1] * (n * (m + 1))
    return np.array(out)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "lvc":
        _lvc_worker(Path(sys.argv[2]), sys.argv[3], sys.argv[4])
    else:
        print(__doc__)
        sys.exit(1)
