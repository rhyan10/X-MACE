"""Shared test setup for X-MACE.

* Capabilities: optional requirements (a GPU, an optional dependency, network
  access) are pytest markers; a test whose capability is unavailable is skipped,
  unless MACE_REQUIRE_CAPS names it (then it fails: CI promised it).
* Builders of small excited-state models, batches and label specs used across the
  test files.
"""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import data, modules, tools
from mace.data.excited import ExcitedStateSpec, standardise_excited_configs
from mace.modules.excited import AutoencoderExcitedMACE, ExcitedMACE
from mace.tools import torch_geometric

sys.path.insert(0, str(Path(__file__).parent))
from excited_data import make_excited_configs  # noqa: E402  pylint: disable=wrong-import-position

# ---------------------------------------------------------------------------
# Capability model
#
# Each optional requirement of a test (a GPU, an optional dependency, network
# access, an external binary) is a "capability", expressed as a pytest marker
# (registered in pyproject.toml). The contract is:
#
#   * locally, a test whose capability is unavailable is SKIPPED;
#   * in CI, each job exports MACE_REQUIRE_CAPS with the capabilities it
#     guarantees (because it installed them / runs on a GPU host); a test that
#     would skip for one of those capabilities FAILS instead. A job can never
#     again go green while silently skipping the thing it exists to test.
#
# Vendor is deliberately NOT a capability: tests always use device="cuda"
# (valid on ROCm too) and GPU jobs select by marker expression per vendor
# (e.g. "gpu and not cueq" on AMD hosts, since cuEquivariance is NVIDIA-only).
# ---------------------------------------------------------------------------


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _gpu_available() -> bool:
    try:
        import torch  # pylint: disable=import-outside-toplevel

        # True on ROCm builds as well; torch.version.hip distinguishes vendors.
        return torch.cuda.is_available()
    except ImportError:
        return False


def _lammps_available() -> bool:
    # Real import, not find_spec: a broken wheel (e.g. missing libmpi) must
    # read as unavailable.
    try:
        import lammps  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import

        return True
    except (ImportError, OSError):
        return False


CAPABILITY_PROBES = {
    "gpu": _gpu_available,
    "cueq": lambda: _module_available("cuequivariance"),
    "oeq": lambda: _module_available("openequivariance"),
    "polar": lambda: _module_available("graph_longrange"),
    "les": lambda: _module_available("les"),
    "magnetic": lambda: _module_available("sphericart"),
    "torchsim": lambda: _module_available("torch_sim"),
    "schedulefree": lambda: _module_available("schedulefree"),
    "wandb": lambda: _module_available("wandb"),
    "bin_lammps": _lammps_available,
    # Never autodetected: network access is an explicit opt-in so that PR jobs
    # stay deterministic and downloads live only in jobs that chose them.
    "network": lambda: os.environ.get("MACE_CI_ALLOW_NETWORK", "") == "1",
}

_capability_cache: dict = {}


def capability_available(cap: str) -> bool:
    if cap not in _capability_cache:
        _capability_cache[cap] = CAPABILITY_PROBES[cap]()
    return _capability_cache[cap]


def required_caps() -> set:
    return {
        cap.strip()
        for cap in os.environ.get("MACE_REQUIRE_CAPS", "").split(",")
        if cap.strip()
    }


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    # tryfirst: must run before the builtin skipping plugin, so that a
    # required-but-unavailable capability fails even if a legacy
    # @pytest.mark.skipif on the same test would have skipped it.
    missing = [
        cap
        for cap in CAPABILITY_PROBES
        if item.get_closest_marker(cap) is not None and not capability_available(cap)
    ]
    if not missing:
        return
    # A guaranteed-but-missing capability outranks any other skip reason: the
    # job promised it, so its absence is a broken job, not a skippable test.
    broken = sorted(set(missing) & required_caps())
    if broken:
        pytest.fail(
            f"MACE_REQUIRE_CAPS guarantees {broken} but unavailable on this "
            f"host — the CI job is broken (bad install or wrong runner), "
            f"not the test.",
            pytrace=False,
        )
    pytest.skip(f"requires capabilities {sorted(missing)}")



def pytest_configure(config):
    """Validate MACE_REQUIRE_CAPS, isolate cache per xdist worker, and guard
    known plugin reseed issues."""
    unknown = required_caps() - set(CAPABILITY_PROBES)
    if unknown:
        raise pytest.UsageError(
            f"MACE_REQUIRE_CAPS contains unknown capabilities: {sorted(unknown)}; "
            f"known: {sorted(CAPABILITY_PROBES)}"
        )

    os.environ.setdefault("TORCHINDUCTOR_DISABLE_PCH", "1")
    # Ensure inductor does not use stale precompiled headers across test runs.
    try:
        import torch._inductor.config as inductor_config

        inductor_config.cpp_cache_precompile_headers = False
    except (ImportError, AttributeError):
        pass
    if hasattr(config.option, "randomly_reset_seed"):
        # Some environments with pytest-randomly + thinc can produce invalid seeds.
        config.option.randomly_reset_seed = False

    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker:
        cache_root = Path(tempfile.gettempdir()) / f"mace_cache_{worker}"
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(cache_root)
        print(f"[XDIST] Worker {worker} using cache {cache_root}")




# ---------------------------------------------------------------------------
# X-MACE builders
# ---------------------------------------------------------------------------


Z_TABLE = tools.AtomicNumberTable([1, 6, 8])
R_MAX = 4.0


def model_config(**overrides):
    config = dict(
        r_max=R_MAX,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=2,
        num_elements=len(Z_TABLE),
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=np.array([-1.0, -3.0, -5.0]),
        avg_num_neighbors=3.0,
        atomic_numbers=Z_TABLE.zs,
        correlation=2,
        gate=torch.nn.functional.silu,
    )
    config.update(overrides)
    return config


def build_excited(
    n_states=3, nacs=True, socs=True, dipoles=True, autoencoder=False, **kwargs
):
    cls = AutoencoderExcitedMACE if autoencoder else ExcitedMACE
    extra = {"num_permutational_invariant": 6} if autoencoder else {}
    extra.update(kwargs)
    perturb = extra.pop("perturb", True)
    model = cls(
        **model_config(),
        n_energies=n_states,
        compute_nacs=nacs,
        compute_socs=socs,
        compute_dipoles=dipoles,
        **extra,
    )
    if perturb:
        perturb_energy_readouts(model)
    return model


def perturb_energy_readouts(model, size=0.3):
    """Energy readouts start at zero (every state on the backbone's surface); give
    them random weights so that the states of a test model differ, as after training."""
    generator = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for readout in model.readouts:
            last = getattr(readout, "linear_2", None) or getattr(readout, "linear", None)
            if last is not None and float(last.weight.abs().max()) == 0.0:
                last.weight.copy_(
                    size * torch.randn(last.weight.shape, generator=generator)
                )


KEYSPEC = data.KeySpecification(
    info_keys={
        "energy": "REF_energy",
        "socs": "REF_socs",
        "dipoles": "REF_dipoles",
    },
    arrays_keys={"forces": "REF_forces", "nacs": "REF_nacs"},
)


def atoms_to_batch(atoms_list, spec=None, keyspec=KEYSPEC):
    configs = [data.config_from_atoms(a, key_specification=keyspec) for a in atoms_list]
    if spec is not None:
        standardise_excited_configs(configs, spec)
    dataset = [data.AtomicData.from_config(c, z_table=Z_TABLE, cutoff=R_MAX) for c in configs]
    loader = torch_geometric.dataloader.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    return next(iter(loader))


def full_spec(n_states=3, **kwargs):
    n_pairs = n_states * (n_states - 1) // 2
    values = dict(
        n_states=n_states,
        n_nacs=n_pairs,
        n_socs=max(n_pairs, 1),
        n_dipoles=n_states + n_pairs,
        nacs_format="raw",
    )
    values.update(kwargs)
    return ExcitedStateSpec(**values)


@pytest.fixture(autouse=True)
def _float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(previous)


@pytest.fixture(name="molecules")
def fixture_molecules():
    return make_excited_configs(4, n_states=3, seed=3)


def flip_nac_readouts(model):
    """Change the sign of a model's NACs (all its polar and axial NAC readouts)."""
    readouts = list(model.nacs_readouts) + list(getattr(model, "nacs_axial_readouts", []))
    for readout in readouts:
        readout.register_forward_hook(_negate)


def _negate(_module, _inputs, output):
    return -output
