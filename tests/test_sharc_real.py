"""SHARC_MACE.py with a real SHARC 4 installation (skipped without $SHARC).

Set up SHARC (https://github.com/sharc-md/sharc4) and its environment (sharcvars.sh)
so that $SHARC/sharc.x exists and $SHARC/../lib is importable. The tests run the
interface the ways SHARC does: a single point (SHARC_MACE.py QM.in), a sharc.x
trajectory (QM/runQM.sh, one process per step) and a pysharc trajectory
($SHARC/driver.py -i MACE), with singlets, triplets and quintets, NACs and
spin-orbit couplings.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mace.calculators.sharc_calculator import BOHR_TO_ANGSTROM, SharcCalculator

sys.path.insert(0, str(Path(__file__).parent))
from conftest import build_excited  # noqa: E402  pylint: disable=wrong-import-position

SHARC = os.environ.get("SHARC")
REPO = Path(__file__).resolve().parents[1]
INTERFACE = REPO / "scripts" / "sharc" / "SHARC_MACE.py"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not SHARC or not Path(SHARC, "sharc.x").is_file(),
        reason="needs a SHARC 4 installation ($SHARC/sharc.x)",
    ),
]

STATES = [2, 0, 2, 0, 1]  # 2 singlets, 2 triplets, 1 quintet: 13 M_S states
SYMBOLS = ["C", "H", "H", "O"]
POSITIONS = np.array(
    [[0.0, 0.0, 0.0], [1.09, 0.05, 0.0], [-0.4, 1.0, 0.1], [-0.3, -0.6, 1.0]]
)
MASSES = {"C": 12.0, "H": 1.00782503, "O": 15.99491462}
NUMBERS = {"C": 6.0, "H": 1.0, "O": 8.0}
PROPERTIES = "energy forces nacs socs dipoles"


def _env():
    env = dict(os.environ)
    lib = str(Path(SHARC).parent / "lib")
    pysharc = str(Path(SHARC).parent / "pysharc")
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(REPO),
            str(INTERFACE.parent),
            lib,
            SHARC,
            pysharc,
            env.get("PYTHONPATH", ""),
        ]
    )
    env["LD_LIBRARY_PATH"] = os.pathsep.join([lib, env.get("LD_LIBRARY_PATH", "")])
    env.setdefault("HOSTNAME", "localhost")
    return env


def _run(cmd, cwd, timeout=600):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return result


@pytest.fixture(name="model_file", scope="module")
def fixture_model_file(tmp_path_factory):
    torch.manual_seed(3)
    model = build_excited(
        n_states=sum(STATES), states=STATES, soc_mode="spin_orbit"
    ).double()
    path = tmp_path_factory.mktemp("model") / "stq.model"
    torch.save(model, path)
    return path


def _write_template(directory: Path, model_file: Path):
    (directory / "MACE.template").write_text(
        f"model_file    {model_file}\nproperties    {PROPERTIES}\n"
    )


def _read_qmout(directory: Path) -> dict:
    """QM.out parsed by SHARC's own reader (in a subprocess: SHARC's modules have
    generic names that should not leak into the test process)."""
    script = (
        "import numpy as np, sys\n"
        "from qmout import QMout\n"
        "q = QMout(filepath='QM.out')\n"
        "np.savez('qmout.npz', **{k: np.asarray(q[k]) for k in ('h', 'grad', 'nacdr', 'dm')})\n"
    )
    _run([sys.executable, "-c", script], cwd=directory)
    stored = np.load(directory / "qmout.npz")
    return {key: stored[key] for key in stored.files}


def test_single_point(tmp_path, model_file):
    lines = [str(len(SYMBOLS)), ""]
    lines += [
        f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, (x, y, z) in zip(SYMBOLS, POSITIONS)
    ]
    lines += [
        "init",
        "unit angstrom",
        "states " + " ".join(map(str, STATES)),
        "charge 0 0 0 0 0",
        f"savedir {tmp_path / 'save'}",
        "h",
        "soc",
        "dm",
        "grad",
        "nacdr",
    ]
    (tmp_path / "QM.in").write_text("\n".join(lines) + "\n")
    _write_template(tmp_path, model_file)
    _run([sys.executable, str(INTERFACE), "QM.in"], cwd=tmp_path)
    qmout = _read_qmout(tmp_path)

    sharc = SharcCalculator(
        SYMBOLS, str(model_file), states=STATES, properties=PROPERTIES.split()
    )
    expected = sharc.calculate(POSITIONS / BOHR_TO_ANGSTROM)
    for key in ("h", "grad", "nacdr", "dm"):
        scale = max(np.abs(expected[key]).max(), 1e-12)
        np.testing.assert_allclose(qmout[key], expected[key], atol=1e-9 * scale + 1e-11)
    h = qmout["h"]
    assert h.shape == (13, 13)
    np.testing.assert_allclose(h, h.conj().T, atol=1e-12)
    assert np.any(np.abs(h - np.diag(np.diag(h))) > 0)  # SOCs present
    # singlet-quintet and singlet-singlet blocks carry no spin-orbit coupling
    assert np.all(h[:2, 8:] == 0) and h[0, 1] == 0


def _trajectory(directory: Path, model_file: Path, tmax=3.0):
    rng = np.random.default_rng(0)
    bohr = POSITIONS / BOHR_TO_ANGSTROM
    (directory / "geom").write_text(
        "".join(
            f"{s:>4} {NUMBERS[s]:5.1f} {x:14.8f} {y:14.8f} {z:14.8f} {MASSES[s]:14.8f}\n"
            for s, (x, y, z) in zip(SYMBOLS, bohr)
        )
    )
    (directory / "veloc").write_text(
        "".join(
            f"{a:14.8f} {b:14.8f} {c:14.8f}\n"
            for a, b, c in rng.normal(0, 2e-4, (4, 3))
        )
    )
    states = " ".join(map(str, STATES))
    (directory / "input").write_text(f"""geomfile "geom"
veloc external
velocfile "veloc"
nstates {states}
charge 0 0 0 0 0
actstates {states}
state 3 mch
coeff auto
rngseed 26933
ezero 0.0
tmax {tmax}
stepsize 0.5
nsubsteps 25
surf sharc
coupling nacdr
gradcorrect
ekincorrect parallel_vel
decoherence_scheme edc
nophases_from_interface
output_format ascii
""")
    qm_dir = directory / "QM"
    qm_dir.mkdir()
    _write_template(qm_dir, model_file)
    (qm_dir / "runQM.sh").write_text(
        f"cd QM\n{sys.executable} {INTERFACE} QM.in >> QM.log 2>> QM.err\n"
    )


def _energies(directory: Path) -> np.ndarray:
    rows = [
        line.split()
        for line in (directory / "output.lis").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return np.array([[float(r[1]), float(r[6])] for r in rows])  # time, E_tot (eV)


def test_sharc_x_trajectory(tmp_path, model_file):
    _trajectory(tmp_path, model_file)
    _run([str(Path(SHARC) / "sharc.x"), "input"], cwd=tmp_path, timeout=1200)
    energies = _energies(tmp_path)
    assert energies[-1, 0] == pytest.approx(3.0)
    # total energy conserved with SOCs, NACs and gradient correction
    assert np.ptp(energies[:, 1]) < 1e-3
    # couplings stay phase continuous across the separate interface calls
    assert list((tmp_path / "restart").glob("mace_phases.*"))


def test_pysharc_trajectory_matches_sharc_x(tmp_path, model_file):
    pytest.importorskip("netCDF4")
    try:
        _run([sys.executable, "-c", "import sharc.sharc"], cwd=tmp_path)
    except AssertionError:
        pytest.skip("pysharc is not built")
    reference = tmp_path / "sharc_x"
    pysharc = tmp_path / "pysharc"
    for directory in (reference, pysharc):
        directory.mkdir()
        _trajectory(directory, model_file, tmax=2.0)
    _run([str(Path(SHARC) / "sharc.x"), "input"], cwd=reference, timeout=1200)
    result = subprocess.run(
        [sys.executable, str(Path(SHARC) / "driver.py"), "-i", "MACE", "input"],
        cwd=pysharc,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )
    if result.returncode != 0:
        log = (
            (pysharc / "output.log").read_text()
            if (pysharc / "output.log").exists()
            else ""
        )
        if "Keyword CHARGE not found" in log:
            pytest.skip(
                "this SHARC build lost the input file: source/input_list.f90 "
                "next_keyword never sets stat on success; set 'stat = 0' and rebuild"
            )
        raise AssertionError(result.stdout[-3000:] + result.stderr[-3000:])
    np.testing.assert_allclose(_energies(pysharc), _energies(reference), atol=2e-6)
