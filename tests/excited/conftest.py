"""Shared builders for the X-MACE (excited-state) tests."""

from __future__ import annotations

import sys
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
    return cls(
        **model_config(),
        n_energies=n_states,
        compute_nacs=nacs,
        compute_socs=socs,
        compute_dipoles=dipoles,
        **extra,
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
