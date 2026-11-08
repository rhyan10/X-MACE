"""ASE calculator, batch evaluation and the SHARC layer for excited-state models."""

import importlib.util
import sys
import types
from pathlib import Path

import ase
import numpy as np
import pytest
import torch

from mace.calculators import ExcitedMACECalculator, MACECalculator
from mace.calculators.excited import evaluate_excited_configs, raw_nacs_from_smooth
from mace.calculators.sharc_calculator import (
    BOHR_TO_ANGSTROM,
    HARTREE_TO_EV,
    SharcCalculator,
)

from conftest import atoms_to_batch, build_excited  # pylint: disable=import-error

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(name="model_file")
def fixture_model_file(tmp_path):
    torch.manual_seed(5)
    model = build_excited(pair_repulsion=True)
    path = tmp_path / "excited.model"
    torch.save(model, path)
    return path


def _model_output(path, atoms):
    model = torch.load(path, weights_only=False)
    return model(atoms_to_batch([atoms], None).to_dict())


def test_ase_calculator_matches_model(model_file, molecules):
    atoms = molecules[0].copy()
    calc = ExcitedMACECalculator(model_file, state=2)
    atoms.calc = calc
    out = _model_output(model_file, atoms)
    np.testing.assert_allclose(atoms.get_potential_energy(), out["energy"][0, 2].item())
    np.testing.assert_allclose(atoms.get_forces(), out["forces"][:, 2].detach().numpy())
    np.testing.assert_allclose(calc.results["energies"], out["energy"][0].detach().numpy())
    np.testing.assert_allclose(calc.results["smooth_nacs"], out["nacs"].detach().numpy())
    np.testing.assert_allclose(calc.results["socs"], out["socs"][0].detach().numpy())
    np.testing.assert_allclose(calc.results["dipoles"], out["dipoles"][0].detach().numpy())
    np.testing.assert_allclose(
        calc.results["nacs"],
        raw_nacs_from_smooth(calc.results["smooth_nacs"], calc.results["energies"]),
    )


def test_raw_nacs_divide_by_gap():
    smooth = np.ones((2, 3, 3))
    nacs = raw_nacs_from_smooth(smooth, np.array([0.0, 1.0, 3.0]))
    np.testing.assert_allclose(nacs[0, :, 0], [1.0, 1 / 3, 1 / 2])


def test_ase_md_step_conserves_energy_roughly(model_file, molecules):
    from ase import units
    from ase.md.verlet import VelocityVerlet

    atoms = molecules[0].copy()
    atoms.calc = ExcitedMACECalculator(model_file, state=0)
    atoms.set_momenta(np.zeros((4, 3)))
    dyn = VelocityVerlet(atoms, 0.1 * units.fs)
    e_start = atoms.get_total_energy()
    dyn.run(20)
    assert abs(atoms.get_total_energy() - e_start) < 1e-3


def test_committee_aligns_coupling_signs(tmp_path, molecules):
    torch.manual_seed(6)
    model = build_excited()
    flipped = build_excited()
    flipped.load_state_dict(model.state_dict())
    flipped.nacs_readouts[0].linear.weight.data.mul_(-1)
    flipped.nacs_readouts[1].linear_2.weight.data.mul_(-1)
    for readout in flipped.socs_readouts:
        last = readout.linear_2 if hasattr(readout, "linear_2") else readout.linear
        last.weight.data.mul_(-1)
    torch.save(model, tmp_path / "a.model")
    torch.save(flipped, tmp_path / "b.model")
    single = ExcitedMACECalculator(tmp_path / "a.model")
    committee = ExcitedMACECalculator([tmp_path / "a.model", tmp_path / "b.model"])
    atoms = molecules[1]
    ref = single.predict(atoms)
    both = committee.predict(atoms)
    np.testing.assert_allclose(both["smooth_nacs"], ref["smooth_nacs"], atol=1e-12)
    np.testing.assert_allclose(both["socs"], ref["socs"], atol=1e-12)
    np.testing.assert_allclose(both["energies_var"], 0.0, atol=1e-20)


def test_glob_pattern_and_errors(tmp_path, model_file):
    calc = ExcitedMACECalculator(str(model_file.parent / "*.model"))
    assert calc.num_models == 1
    with pytest.raises(ValueError, match="Couldn't find"):
        ExcitedMACECalculator(str(tmp_path / "nothing*.model"))
    with pytest.raises(ValueError, match="state must be"):
        ExcitedMACECalculator(model_file, state=3)


def test_ground_state_models_are_refused(tmp_path):
    from mace import modules

    from conftest import model_config  # pylint: disable=import-error

    model = modules.ScaleShiftMACE(**model_config(), atomic_inter_scale=1.0, atomic_inter_shift=0.0)
    torch.save(model, tmp_path / "gs.model")
    with pytest.raises(ValueError, match="not an excited-state model"):
        ExcitedMACECalculator(tmp_path / "gs.model")
    MACECalculator(model_paths=str(tmp_path / "gs.model"), device="cpu")  # still works


def test_float32_calculator(model_file, molecules):
    calc = ExcitedMACECalculator(model_file, default_dtype="float32")
    atoms = molecules[0].copy()
    atoms.calc = calc
    ref = _model_output(model_file, atoms)["energy"][0, 0].item()
    assert abs(atoms.get_potential_energy() - ref) < 1e-4


def test_evaluate_excited_configs(model_file, molecules):
    model = torch.load(model_file, weights_only=False)
    atoms_list = [a.copy() for a in molecules]
    evaluate_excited_configs(model, atoms_list, batch_size=3, info_prefix="P_")
    for atoms in atoms_list:
        out = _model_output(model_file, atoms)
        np.testing.assert_allclose(atoms.info["P_energies"], out["energy"][0].detach().numpy())
        np.testing.assert_allclose(atoms.info["P_forces"], out["forces"].detach().numpy())
        assert atoms.info["P_nacs"].shape == (4, 3, 3)
        assert atoms.info["P_socs"].shape == (3,)
        assert atoms.info["P_dipoles"].shape == (6, 3)
        assert atoms.info["P_energy"] == pytest.approx(out["energy"][0, 0].item())


# ------------------------------------------------------------------------------------
# SHARC
# ------------------------------------------------------------------------------------


def _sharc(model_file, singlets=3, triplets=0, properties=None, **kwargs):
    properties = properties or ["energy", "forces", "nacs", "dipoles", "socs"]
    return SharcCalculator(
        ["C", "H", "H", "O"],
        str(model_file),
        n_states={"n_singlets": singlets, "n_triplets": triplets},
        properties=[properties],  # SHARC's nested template format
        **kwargs,
    )


def test_sharc_singlets_units(model_file, molecules):
    atoms = molecules[0]
    sharc = _sharc(model_file)
    qm = sharc.calculate(atoms.positions / BOHR_TO_ANGSTROM)
    out = _model_output(model_file, atoms)
    energies = out["energy"][0].detach().numpy()
    forces = out["forces"].detach().numpy()
    np.testing.assert_allclose(np.diag(qm["h"]).real, energies / HARTREE_TO_EV)
    np.testing.assert_allclose(
        qm["grad"], -np.transpose(forces, (1, 0, 2)) / HARTREE_TO_EV * BOHR_TO_ANGSTROM
    )
    nacs = raw_nacs_from_smooth(out["nacs"].detach().numpy(), energies) * BOHR_TO_ANGSTROM
    np.testing.assert_allclose(qm["nacdr"][0, 1], nacs[:, 0])
    np.testing.assert_allclose(qm["nacdr"][1, 2], nacs[:, 2])
    np.testing.assert_allclose(qm["nacdr"], -np.transpose(qm["nacdr"], (1, 0, 2, 3)))
    dipoles = out["dipoles"][0].detach().numpy() / BOHR_TO_ANGSTROM
    np.testing.assert_allclose(qm["dm"][:, 1, 1], dipoles[1])
    np.testing.assert_allclose(qm["dm"][:, 0, 2], dipoles[3 + 1])
    np.testing.assert_allclose(qm["dm"], np.transpose(qm["dm"], (0, 2, 1)))
    # singlets only: no SOCs in h
    np.testing.assert_allclose(qm["h"], np.diag(np.diag(qm["h"])))


def test_sharc_smooth_nacs(model_file, molecules):
    sharc = _sharc(model_file, properties=["energy", "forces", "smooth_nacs"])
    qm = sharc.calculate(molecules[0].positions / BOHR_TO_ANGSTROM)
    out = _model_output(model_file, molecules[0])
    expected = out["nacs"].detach().numpy()[:, 0] / HARTREE_TO_EV * BOHR_TO_ANGSTROM
    np.testing.assert_allclose(qm["nacdr"][0, 1], expected)


def test_sharc_uses_lowest_singlets_of_a_bigger_model(model_file, molecules):
    sharc = _sharc(model_file, singlets=2)
    qm = sharc.calculate(molecules[0].positions / BOHR_TO_ANGSTROM)
    assert qm["h"].shape == (2, 2) and qm["nacdr"].shape == (2, 2, 4, 3)


def _spatial_triplet_model(tmp_path, soc_num, n_states=3):
    torch.manual_seed(7)
    model = build_excited(n_states=n_states, soc_num=soc_num)
    path = tmp_path / f"spatial_{soc_num}.model"
    torch.save(model, path)
    return path


@pytest.mark.parametrize("complex_socs", [False, True])
def test_sharc_spatial_singlets_and_triplets(tmp_path, molecules, complex_socs):
    # 2 singlets + 1 triplet: 3 spatial states, nmstates = 5
    n_upper = 5 * 4 // 2
    path = _spatial_triplet_model(tmp_path, 2 * n_upper if complex_socs else n_upper)
    sharc = _sharc(path, singlets=2, triplets=1)
    assert sharc.mode == "spatial"
    qm = sharc.calculate(molecules[0].positions / BOHR_TO_ANGSTROM)
    out = _model_output(path, molecules[0])
    energies = out["energy"][0].detach().numpy() / HARTREE_TO_EV
    h = qm["h"]
    assert h.shape == (5, 5)
    np.testing.assert_allclose(np.diag(h).real, [energies[0], energies[1]] + [energies[2]] * 3)
    np.testing.assert_allclose(h, h.conj().T)  # Hermitian
    socs = out["socs"][0].detach().numpy() / HARTREE_TO_EV
    if complex_socs:
        assert abs(h[0, 1].imag - socs[1]) < 1e-12 and np.any(np.abs(h.imag) > 0)
    else:
        assert abs(h[0, 1] - socs[0]) < 1e-12 and np.all(h.imag == 0)
    # gradients of the three triplet components are the same
    np.testing.assert_allclose(qm["grad"][2], qm["grad"][3])
    np.testing.assert_allclose(qm["grad"][2], qm["grad"][4])
    # NACs: singlet-singlet set, singlet-triplet zero, triplets only within the same M_S
    nacdr = qm["nacdr"]
    assert np.any(nacdr[0, 1] != 0)
    assert np.all(nacdr[0, 2:] == 0) and np.all(nacdr[1, 2:] == 0)
    assert np.all(nacdr[2:, 2:] == 0)  # one triplet: no triplet-triplet pairs
    dm = qm["dm"]
    np.testing.assert_allclose(dm[:, 2, 2], dm[:, 4, 4])
    assert np.all(dm[:, 0, 2:] == 0)


def test_sharc_ms_resolved_model(tmp_path, molecules):
    torch.manual_seed(8)
    model = build_excited(n_states=4, soc_num=6)  # 1 singlet + 1 triplet, M_S resolved
    path = tmp_path / "ms.model"
    torch.save(model, path)
    sharc = _sharc(path, singlets=1, triplets=1)
    assert sharc.mode == "ms_resolved"
    qm = sharc.calculate(molecules[0].positions / BOHR_TO_ANGSTROM)
    assert qm["h"].shape == (4, 4) and qm["nacdr"].shape == (4, 4, 4, 3)


def test_sharc_error_messages(tmp_path, model_file):
    with pytest.raises(ValueError, match="singlets and"):
        _sharc(model_file, singlets=2, triplets=2)
    no_nacs = build_excited(nacs=False)
    torch.save(no_nacs, tmp_path / "no_nacs.model")
    with pytest.raises(ValueError, match="without NACs"):
        _sharc(tmp_path / "no_nacs.model")
    wrong_socs = _spatial_triplet_model(tmp_path, soc_num=3)
    with pytest.raises(ValueError, match="upper triangle"):
        _sharc(wrong_socs, singlets=2, triplets=1)
    with pytest.raises(ValueError, match="Unknown properties"):
        _sharc(model_file, properties=["energy", "hessian"])


def test_sharc_phase_continuity(tmp_path, molecules):
    torch.manual_seed(9)
    model = build_excited()
    path = tmp_path / "phase.model"
    torch.save(model, path)
    sharc = _sharc(path)
    coords = molecules[0].positions / BOHR_TO_ANGSTROM
    first = sharc.calculate(coords)
    # the model flips the sign of its couplings between two (identical) steps
    flipped = build_excited()
    flipped.load_state_dict(model.state_dict())
    flipped.nacs_readouts[0].linear.weight.data.mul_(-1)
    flipped.nacs_readouts[1].linear_2.weight.data.mul_(-1)
    sharc.calc.models = [flipped]
    second = sharc.calculate(coords)
    np.testing.assert_allclose(second["nacdr"], first["nacdr"], atol=1e-12)
    sharc.reset_phases()
    third = sharc.calculate(coords)
    np.testing.assert_allclose(third["nacdr"], -first["nacdr"], atol=1e-12)


def _load_sharc_interface(monkeypatch):
    """Import scripts/sharc/SHARC_MACE.py with minimal stand-ins for SHARC's modules."""

    class _Fields(dict):
        def __init__(self):
            super().__init__()
            self.types, self.data = {}, {}

        def __getitem__(self, key):
            return self.data[key]

    class _QMin:
        def __init__(self):
            self.template = _Fields()
            self.resources = _Fields()
            self.molecule = {}
            self.coords = {}
            self.requests = {}

    class SHARC_FAST:  # pylint: disable=invalid-name
        def __init__(self, *args, **kwargs):
            self.QMin = _QMin()
            self.QMout = types.SimpleNamespace()
            self.log = types.SimpleNamespace(warning=lambda *a: None, info=lambda *a: None)

        def setup_interface(self):
            pass

    sharc_fast = types.ModuleType("SHARC_FAST")
    sharc_fast.SHARC_FAST = SHARC_FAST
    utils = types.ModuleType("utils")
    utils.expand_path = utils.link = utils.question = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "SHARC_FAST", sharc_fast)
    monkeypatch.setitem(sys.modules, "utils", utils)
    spec = importlib.util.spec_from_file_location(
        "SHARC_MACE", REPO_ROOT / "scripts" / "sharc" / "SHARC_MACE.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sharc_interface_run(monkeypatch, model_file, molecules):
    module = _load_sharc_interface(monkeypatch)
    interface = module.SHARC_MACE()
    qmin = interface.QMin
    qmin.template.data.update(
        {"model_file": str(model_file), "properties": [["energy", "forces", "nacs", "dipoles"]]}
    )
    qmin.molecule.update(
        {
            "states": [3],
            "elements": ["C", "H", "H", "O"],
            "nmstates": 3,
            "nstates": 3,
            "natom": 4,
            "npc": 0,
        }
    )
    qmin.coords["coords"] = molecules[0].positions / BOHR_TO_ANGSTROM
    qmin.requests.update({"h": True, "grad": True, "nacdr": True, "dm": True, "phases": True})
    interface.setup_interface()
    interface.run()
    qmout = interface.getQMout()
    assert qmout.h.shape == (3, 3)
    assert qmout.grad.shape == (3, 4, 3)
    assert qmout.nacdr.shape == (3, 3, 4, 3)
    assert qmout.dm.shape == (3, 3, 3)
    np.testing.assert_allclose(qmout.phases, np.ones(3))
    with pytest.raises(ValueError, match="doublets"):
        qmin.molecule["states"] = [1, 1]
        interface.setup_interface()
