###########################################################################################
# Minimal ASE Calculator for a single MACE-like model (energies, forces, SOCs, NACs)
# Authors: You + trimmed by ChatGPT
# License: MIT
###########################################################################################

from pathlib import Path
from typing import Union

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from mace import data
from mace.tools import torch_geometric, torch_tools, utils


def get_model_dtype(model: torch.nn.Module) -> str:
    """Return 'float64' or 'float32' for the model's parameter dtype."""
    mode_dtype = next(model.parameters()).dtype
    if mode_dtype == torch.float64:
        return "float64"
    if mode_dtype == torch.float32:
        return "float32"
    raise ValueError(f"Unknown dtype {mode_dtype}")


class MACECalculator(Calculator):
    """Minimal ASE Calculator wrapper for a single model producing:
       energies (per state), forces, SOCs, NACs.
    """

    implemented_properties = ["energy", "free_energy", "forces", "socs", "nacs"]

    def __init__(
        self,
        model_paths: Union[str, Path],
        device: str,
        n_energies: int = 1,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype: str = "",
        charges_key: str = "Qs",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.results = {}
        self.n_energies = int(n_energies)
        # ---- Load single model ----
        model_paths = Path(model_paths)
        if not model_paths.exists():
            raise ValueError(f"Couldn't find model file: {model_paths}")
        self.model = torch.load(f=model_paths, map_location=device, weights_only=False)

        # Device & dtype
        self.device = torch_tools.init_device(device)
        self.model.to(self.device)

        model_dtype = get_model_dtype(self.model)
        if default_dtype == "":
            default_dtype = model_dtype
        if model_dtype != default_dtype:
            if default_dtype == "float64":
                self.model = self.model.double()
            elif default_dtype == "float32":
                self.model = self.model.float()
            else:
                raise ValueError(f"Unsupported default_dtype {default_dtype}")
        torch_tools.set_default_dtype(default_dtype)

        # Freeze
        for p in self.model.parameters():
            p.requires_grad = False

        # Units and topology helpers
        self.energy_units_to_eV = float(energy_units_to_eV)
        self.length_units_to_A = float(length_units_to_A)
        self.z_table = utils.AtomicNumberTable([int(z) for z in self.model.atomic_numbers])
        self.r_max = float(self.model.r_max.cpu())

        # Where to find atomic charges in Atoms for building graphs
        self.charges_key = charges_key

    def _atoms_to_batch(self, atoms):
        cfg = data.config_from_atoms(atoms, charges_key=self.charges_key)
        loader = torch_geometric.dataloader.DataLoader(
            dataset=[data.AtomicData.from_config(cfg, z_table=self.z_table, cutoff=self.r_max)],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(loader)).to(self.device)
        return batch

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms)

        batch = self._atoms_to_batch(atoms)

        # Forward pass. Expect the model to return these keys:
        # "energy" -> (n_states,)
        # "forces" -> (num_atoms, n_states, 3)
        # "socs"   -> (n_states, n_states)      (assumed)
        # "nacs"   -> (num_atoms, n_states, n_states, 3) (assumed)
        out = self.model(batch.to_dict(), training=False)

        # ---- Gather & scale results ----
        # Energies: convert to eV
        energy = out["energy"].detach().to("cpu").numpy() * self.energy_units_to_eV

        # Forces: (num_atoms, n_states, 3) -> scale by energy/length
        forces = (
            out["forces"].detach().to("cpu").numpy()
            * self.energy_units_to_eV
            / self.length_units_to_A
        )

        # SOCs & NACs: pass through as-is (units model-defined)
        socs = out["socs"].detach().to("cpu").numpy()
        nacs = out["nacs"].detach().to("cpu").numpy()

        self.results = {
            "energy": energy,          # (n_states,)
            "free_energy": energy,     # same as energy (0 K)
            "forces": forces,          # (num_atoms, n_states, 3)
            "socs": socs,              # (n_states, n_states)
            "nacs": nacs,              # (num_atoms, n_states, n_states, 3)
        }

