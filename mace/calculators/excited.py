###########################################################################################
# ASE calculator and batch evaluation for excited-state MACE models (X-MACE)
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import ase
import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from mace import data
from mace.modules.spin_states import (
    same_multiplicity_pairs,
    soc_matrix_from_vectors,
    spin_orbit_pairs,
    state_pairs,
)
from mace.tools import torch_geometric, torch_tools, utils

EXCITED_MODEL_CLASSES = ("ExcitedMACE", "AutoencoderExcitedMACE")


def is_excited_model(model: torch.nn.Module) -> bool:
    return model.__class__.__name__ in EXCITED_MODEL_CLASSES or (
        hasattr(model, "n_energies") and hasattr(model, "nacs_readouts")
    )


def _model_dtype(model: torch.nn.Module) -> str:
    return "float64" if next(model.parameters()).dtype == torch.float64 else "float32"


def raw_nacs_from_smooth(
    smooth_nacs: np.ndarray,
    energies: np.ndarray,
    min_gap: float = 1e-8,
    pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> np.ndarray:
    """NACs [n_atoms, P, 3] = smooth NACs / |E_j - E_i| for the given state pairs
    (default: all pairs (0,1), (0,2), ...)."""
    energies = np.asarray(energies).reshape(-1)
    if pairs is None:
        pairs = state_pairs(energies.size)
    i = np.array([p[0] for p in pairs], dtype=int)
    j = np.array([p[1] for p in pairs], dtype=int)
    gaps = np.maximum(np.abs(energies[j] - energies[i]), min_gap)
    return smooth_nacs / gaps[None, :, None]


def soc_upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Upper triangle of a complex SOC matrix as interleaved (re, im) values."""
    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    return np.stack([upper.real, upper.imag], axis=-1).reshape(-1)


@dataclass
class HeadView:
    """What one head of an excited-state model predicts (sizes of its outputs)."""

    index: int
    name: str
    n_energies: int
    multiplicity_states: List[int]
    compute_nacs: bool
    nac_num: int
    compute_socs: bool
    soc_num: int
    soc_mode: str
    soc_vector_num: int
    compute_dipoles: bool
    dipole_num: int

    @classmethod
    def from_model(
        cls, model: torch.nn.Module, head: Optional[Union[str, int]] = None
    ) -> "HeadView":
        """head: name or index; default the model's default head (the first head that
        is not pt_head)."""
        heads = [str(h) for h in getattr(model, "heads", ["Default"])]
        layouts = getattr(model, "head_layouts", None)
        if head is None:
            index = int(getattr(model, "default_head", 0))
        elif isinstance(head, int):
            index = head
        elif head in heads:
            index = heads.index(head)
        else:
            raise ValueError(f"Head {head!r} not in the model's heads {heads}")
        if not 0 <= index < len(heads):
            raise ValueError(f"Head index {index} out of range for heads {heads}")
        if layouts is None:  # single-head model of an older version
            n_states = int(model.n_energies)
            return cls(
                index=index,
                name=heads[index],
                n_energies=n_states,
                multiplicity_states=[
                    int(x) for x in getattr(model, "multiplicity_states", [n_states])
                ],
                compute_nacs=bool(model.compute_nacs),
                nac_num=int(model.nac_num) if bool(model.compute_nacs) else 0,
                compute_socs=bool(model.compute_socs),
                soc_num=int(getattr(model, "soc_num", 0)),
                soc_mode=str(getattr(model, "soc_mode", "free")),
                soc_vector_num=int(getattr(model, "soc_vector_num", 0)),
                compute_dipoles=bool(model.compute_dipoles),
                dipole_num=int(model.dipole_num) if bool(model.compute_dipoles) else 0,
            )
        layout = layouts[index]
        soc_vectors = int(layout["soc_vector_num"])
        return cls(
            index=index,
            name=heads[index],
            n_energies=int(layout["n_states"]),
            multiplicity_states=[int(x) for x in layout["states"]],
            compute_nacs=int(layout["nac_num"]) > 0,
            nac_num=int(layout["nac_num"]),
            compute_socs=int(layout["soc_num"]) > 0 or soc_vectors > 0,
            soc_num=int(layout["soc_num"]),
            soc_mode="spin_orbit" if soc_vectors > 0 else "free",
            soc_vector_num=soc_vectors,
            compute_dipoles=int(layout["dipole_num"]) > 0,
            dipole_num=int(layout["dipole_num"]),
        )

    def select(self, out: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Cut the (padded) outputs of a batch of this head to its own sizes."""
        sizes = {
            "energy": (1, self.n_energies),
            "forces": (1, self.n_energies),
            "nacs": (1, self.nac_num),
            "socs": (1, self.soc_num),
            "soc_vectors": (1, self.soc_vector_num),
            "dipoles": (1, self.dipole_num),
        }
        selected = {}
        for key, value in out.items():
            if key not in sizes:
                selected[key] = value
                continue
            axis, size = sizes[key]
            if size == 0:
                continue
            selected[key] = np.take(value, np.arange(size), axis=axis)
        return selected


@dataclass
class ExcitedModelLayout:
    """What the outputs of an excited-state model mean.

    states       states per multiplicity ([n_singlets, n_doublets, ...])
    nac_pairs    state pair of every NAC (None: custom layout)
    n_permanent  number of permanent dipoles (then transition dipoles of dipole_pairs)
    dipole_pairs state pair of every transition dipole
    soc_mode     "free" (soc_num numbers) or "spin_orbit" (coupling vectors)
    soc_pairs    spatial state pair of every SOC vector (spin_orbit)
    """

    states: List[int]
    nac_pairs: Optional[List[Tuple[int, int]]]
    n_permanent: int
    dipole_pairs: List[Tuple[int, int]]
    soc_mode: str
    soc_pairs: List[Tuple[int, int]]

    @classmethod
    def from_model(
        cls, model: torch.nn.Module, head: Optional[Union[str, int]] = None
    ) -> "ExcitedModelLayout":
        if not isinstance(model, HeadView):
            model = HeadView.from_model(model, head)
        n_states = int(model.n_energies)
        states = [int(s) for s in getattr(model, "multiplicity_states", [n_states])]
        if sum(states) != n_states:
            states = [n_states]
        coupled = same_multiplicity_pairs(states)
        every = state_pairs(n_states)
        nac_pairs = None
        if bool(model.compute_nacs):
            nac_num = int(model.nac_num)
            if nac_num == len(coupled):
                nac_pairs = coupled
            elif nac_num == len(every):
                nac_pairs = every  # X-MACE models without multiplicities: all pairs
        n_dipoles = int(model.dipole_num) if bool(model.compute_dipoles) else 0
        n_permanent, dipole_pairs = n_dipoles, []
        if n_dipoles == n_states + len(coupled):
            n_permanent, dipole_pairs = n_states, coupled
        elif n_dipoles == n_states + len(every):
            n_permanent, dipole_pairs = n_states, every
        soc_mode = str(getattr(model, "soc_mode", "free"))
        soc_pairs = spin_orbit_pairs(states) if soc_mode == "spin_orbit" else []
        return cls(states, nac_pairs, n_permanent, dipole_pairs, soc_mode, soc_pairs)


def coupling_signs(
    reference: np.ndarray, value: np.ndarray, axis_pairs: int
) -> np.ndarray:
    """+1/-1 per coupling (along axis_pairs) so that value * sign best matches reference."""
    other = tuple(a for a in range(value.ndim) if a != axis_pairs)
    minus = np.sum((reference - value) ** 2, axis=other)
    plus = np.sum((reference + value) ** 2, axis=other)
    return np.where(plus < minus, -1.0, 1.0)


def _align_sign(
    reference: np.ndarray, value: np.ndarray, axis_pairs: int
) -> np.ndarray:
    signs = coupling_signs(reference, value, axis_pairs)
    shape = [1] * value.ndim
    shape[axis_pairs] = -1
    return value * signs.reshape(shape)


class ExcitedMACECalculator(Calculator):
    """ASE calculator for ExcitedMACE / AutoencoderExcitedMACE models.

    model_paths  a path, a glob pattern ("model_*.model") or a list for a committee
    state        electronic state whose energy and forces ASE uses ("energy", "forces")
    head         head of a multi-head model (name or index); default the first head
                 that is not pt_head. A ground-state head predicts one state.

    results (S states, P coupled state pairs, L SOCs, D dipoles, Q spin-orbit pairs):
      energy, free_energy   energy of `state` (eV)
      forces                forces of `state` [n_atoms, 3] (eV/A)
      energies              [S]
      forces_states         [n_atoms, S, 3]
      smooth_nacs           [n_atoms, P, 3]  NAC * dE, as predicted
      nacs                  [n_atoms, P, 3]  smooth NACs / dE  (one NAC per coupled pair)
      socs                  [L]              (soc_mode="free")
      soc_vectors           [Q, 3]           (soc_mode="spin_orbit")
      soc_matrix            [nm, nm] complex M_S-resolved SOC matrix (spin_orbit)
      dipoles               [D, 3]
      energies_var, forces_states_var   committee variances (several models)

    self.layout (ExcitedModelLayout) says which states the couplings belong to.

    Couplings are in the units of the training data. For a committee, the
    couplings of every model are sign-aligned to the first model before averaging.
    """

    implemented_properties = [
        "energy",
        "free_energy",
        "forces",
        "energies",
        "forces_states",
        "nacs",
        "smooth_nacs",
        "socs",
        "soc_vectors",
        "soc_matrix",
        "dipoles",
    ]

    def __init__(
        self,
        model_paths: Union[str, Path, Sequence[Union[str, Path]]],
        device: str = "cpu",
        state: int = 0,
        default_dtype: str = "",
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        charges_key: str = "Qs",
        models: Optional[List[torch.nn.Module]] = None,
        head: Optional[Union[str, int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if models is None:
            models = [
                torch.load(f=path, map_location=device, weights_only=False)
                for path in self._expand_paths(model_paths)
            ]
        for model in models:
            if not is_excited_model(model):
                raise ValueError(
                    f"{model.__class__.__name__} is not an excited-state model; "
                    "use mace.calculators.MACECalculator"
                )
        self.device = torch_tools.init_device(device)
        model_dtype = _model_dtype(models[0])
        if default_dtype == "":
            default_dtype = model_dtype
        if default_dtype not in ("float32", "float64"):
            raise ValueError(f"Unsupported default_dtype {default_dtype}")
        self.default_dtype = default_dtype
        dtype = torch.float64 if default_dtype == "float64" else torch.float32
        self.models = [model.to(self.device).to(dtype) for model in models]
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False
        reference = self.models[0]
        self.head_view = HeadView.from_model(reference, head)
        self.head = self.head_view.name
        for model in self.models[1:]:
            view = HeadView.from_model(model, self.head)
            if view.n_energies != self.head_view.n_energies:
                raise ValueError(
                    "Committee members predict different numbers of states"
                )
            if float(model.r_max) != float(reference.r_max):
                raise ValueError("Committee members have different r_max")
            if ExcitedModelLayout.from_model(view) != ExcitedModelLayout.from_model(
                self.head_view
            ):
                raise ValueError("Committee members predict different couplings")
        self.num_models = len(self.models)
        self.n_states = self.head_view.n_energies
        if not 0 <= state < self.n_states:
            raise ValueError(f"state must be in [0, {self.n_states - 1}], got {state}")
        self.state = state
        self.r_max = float(reference.r_max)
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in reference.atomic_numbers]
        )
        self.energy_units_to_eV = float(energy_units_to_eV)
        self.length_units_to_A = float(length_units_to_A)
        self.charges_key = charges_key
        self.layout = ExcitedModelLayout.from_model(self.head_view)
        self.nacs_per_pair = self.layout.nac_pairs is not None

    def _head_indices(self, model: torch.nn.Module) -> int:
        heads = [str(h) for h in getattr(model, "heads", ["Default"])]
        return heads.index(self.head) if self.head in heads else 0

    @staticmethod
    def _expand_paths(model_paths) -> List[str]:
        if isinstance(model_paths, (str, Path)):
            paths = sorted(glob(str(model_paths)))
            if not paths:
                raise ValueError(f"Couldn't find MACE model files: {model_paths}")
            return paths
        paths = [str(path) for path in model_paths]
        if not paths:
            raise ValueError("No model files given")
        return paths

    def _atoms_to_batch(self, atoms: ase.Atoms):
        keyspec = data.KeySpecification(
            info_keys={}, arrays_keys={"charges": self.charges_key}
        )
        config = data.config_from_atoms(atoms, key_specification=keyspec)
        with torch_tools.default_dtype(
            torch.float64 if self.default_dtype == "float64" else torch.float32
        ):
            dataset = [
                data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=self.r_max
                )
            ]
        loader = torch_geometric.dataloader.DataLoader(
            dataset, batch_size=1, shuffle=False
        )
        return next(iter(loader)).to(self.device)

    def predict(self, atoms: ase.Atoms) -> Dict[str, np.ndarray]:
        """All predictions for atoms, without touching self.results."""
        batch = self._atoms_to_batch(atoms)
        per_model = []
        for model in self.models:
            inputs = batch.clone().to_dict()
            inputs["head"] = torch.full_like(inputs["head"], self._head_indices(model))
            out = model(inputs, training=False, compute_force=True)
            per_model.append(
                self.head_view.select(
                    {
                        key: torch_tools.to_numpy(value).copy()
                        for key, value in out.items()
                        if key
                        in ("energy", "forces", "nacs", "socs", "dipoles", "soc_vectors")
                        and value is not None
                    }
                )
            )
        return self._combine(per_model)

    def _combine(self, per_model: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        e_conv = self.energy_units_to_eV
        f_conv = self.energy_units_to_eV / self.length_units_to_A
        energies = np.stack([p["energy"][0] for p in per_model]) * e_conv  # [M, S]
        forces = np.stack([p["forces"] for p in per_model]) * f_conv  # [M, N, S, 3]
        results: Dict[str, np.ndarray] = {
            "energies": energies.mean(axis=0),
            "forces_states": forces.mean(axis=0),
        }
        if self.num_models > 1:
            results["energies_var"] = energies.var(axis=0)
            results["forces_states_var"] = forces.var(axis=0)
        first = per_model[0]
        if "nacs" in first:
            aligned = [first["nacs"]] + [
                _align_sign(first["nacs"], p["nacs"], axis_pairs=1)
                for p in per_model[1:]
            ]
            smooth = np.mean(aligned, axis=0)
            results["smooth_nacs"] = smooth
            if self.nacs_per_pair:
                model_energies = np.mean([p["energy"][0] for p in per_model], axis=0)
                results["nacs"] = raw_nacs_from_smooth(
                    smooth, model_energies, pairs=self.layout.nac_pairs
                )
        if "socs" in first:
            socs = [first["socs"][0]]
            for p in per_model[1:]:
                value = p["socs"][0]
                if np.sum((socs[0] + value) ** 2) < np.sum((socs[0] - value) ** 2):
                    value = -value
                socs.append(value)
            results["socs"] = np.mean(socs, axis=0)
        if "soc_vectors" in first:
            vectors = [first["soc_vectors"][0]] + [
                _align_sign(first["soc_vectors"][0], p["soc_vectors"][0], axis_pairs=0)
                for p in per_model[1:]
            ]
            results["soc_vectors"] = np.mean(vectors, axis=0)
            results["soc_matrix"] = soc_matrix_from_vectors(
                results["soc_vectors"], self.layout.states
            )
        if "dipoles" in first:
            n_dipoles = first["dipoles"].shape[1]
            n_permanent = self.layout.n_permanent
            dipoles = [first["dipoles"][0]]
            for p in per_model[1:]:
                value = p["dipoles"][0].copy()
                if n_permanent < n_dipoles:
                    value[n_permanent:] = _align_sign(
                        dipoles[0][n_permanent:], value[n_permanent:], axis_pairs=0
                    )
                dipoles.append(value)
            results["dipoles"] = np.mean(dipoles, axis=0)
        return results

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        results = self.predict(self.atoms)
        results["energy"] = float(results["energies"][self.state])
        results["free_energy"] = results["energy"]
        results["forces"] = results["forces_states"][:, self.state, :].copy()
        self.results = results


def evaluate_excited_configs(
    model: torch.nn.Module,
    atoms_list: List[ase.Atoms],
    batch_size: int = 16,
    device: str = "cpu",
    info_prefix: str = "MACE_",
    head: Optional[Union[str, int]] = None,
) -> List[ase.Atoms]:
    """Write the predictions of an excited-state model into atoms.info (in place).

    <prefix>energy (state 0), <prefix>energies [S], <prefix>forces [n_atoms, S, 3],
    and if predicted <prefix>smooth_nacs, <prefix>nacs [n_atoms, P, 3],
    <prefix>socs [L] (free) or <prefix>soc_vectors [Q, 3] and <prefix>socs, the upper
    triangle of the M_S-resolved SOC matrix as (re, im) pairs [nm (nm - 1)]
    (spin_orbit; the layout --socs_key reads back), <prefix>dipoles [D, 3].
    head selects the head of a multi-head model (default: the first excited head)."""
    calc = ExcitedMACECalculator(
        model_paths=None, models=[model], device=device, head=head
    )
    z_table, r_max = calc.z_table, calc.r_max
    configs = [
        data.config_from_atoms(atoms, key_specification=data.KeySpecification())
        for atoms in atoms_list
    ]
    loader = torch_geometric.dataloader.DataLoader(
        [
            data.AtomicData.from_config(c, z_table=z_table, cutoff=r_max)
            for c in configs
        ],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    index = 0
    for batch in loader:
        batch = batch.to(calc.device)
        inputs = batch.to_dict()
        inputs["head"] = torch.full(
            (batch.num_graphs,), calc._head_indices(calc.models[0]), device=calc.device
        )
        out = calc.models[0](inputs, training=False, compute_force=True)
        out = calc.head_view.select(
            {
                key: torch_tools.to_numpy(value)
                for key, value in out.items()
                if key in ("energy", "forces", "nacs", "socs", "dipoles", "soc_vectors")
                and value is not None
            }
        )
        ptr = torch_tools.to_numpy(batch.ptr)
        energies = out["energy"]
        forces = out["forces"]
        nacs = out.get("nacs")
        socs = out.get("socs")
        soc_vectors = out.get("soc_vectors")
        dipoles = out.get("dipoles")
        for g in range(len(ptr) - 1):
            atoms = atoms_list[index]
            atoms.calc = None
            start, stop = ptr[g], ptr[g + 1]
            atoms.info[info_prefix + "energy"] = float(energies[g, 0])
            atoms.info[info_prefix + "energies"] = energies[g]
            atoms.info[info_prefix + "forces"] = forces[start:stop]
            if nacs is not None:
                atoms.info[info_prefix + "smooth_nacs"] = nacs[start:stop]
                if calc.nacs_per_pair:
                    atoms.info[info_prefix + "nacs"] = raw_nacs_from_smooth(
                        nacs[start:stop], energies[g], pairs=calc.layout.nac_pairs
                    )
            if socs is not None:
                atoms.info[info_prefix + "socs"] = socs[g]
            if soc_vectors is not None:
                atoms.info[info_prefix + "soc_vectors"] = soc_vectors[g]
                atoms.info[info_prefix + "socs"] = soc_upper_triangle(
                    soc_matrix_from_vectors(soc_vectors[g], calc.layout.states)
                )
            if dipoles is not None:
                atoms.info[info_prefix + "dipoles"] = dipoles[g]
            index += 1
    return atoms_list
