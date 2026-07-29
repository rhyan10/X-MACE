###########################################################################################
# Minimal ASE Calculator for a single MACE-like model (energies, forces, SOCs, NACs)
# Authors: Rhyan Barrett
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
        nacs_key: str = "REF_nacs",
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
        self.nacs_key = nacs_key

    def _atoms_to_batch(self, atoms):
        cfg = data.config_from_atoms(atoms, charges_key=self.charges_key, nacs_key=self.nacs_key)
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

        # Forward pass. The X-MACE models return these keys:
        # "energy" -> (n_graphs, n_states)
        # "forces" -> (num_atoms, n_states, 3)
        # "socs"   -> (n_state_pairs,)                    (empty tensor if not computed)
        # "nacs"   -> (num_atoms, n_state_pairs, 3)       (empty tensor if not computed)
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

        # --- Physical NACs ---
        # The model predicts SMOOTH NACs (couplings pre-multiplied by the energy
        # gap during training). Undo that here by dividing out the gap to recover
        # the physical couplings, which diverge as 1/(En - Em) near conical
        # intersections. Pair order must match the training convention:
        # (0,1), (0,2), (1,2), ... from np.triu_indices(n_states, k=1).
        smooth_nacs = out["nacs"].detach().to("cpu").numpy()   # (n_atoms, n_pairs, 3), smooth
        if smooth_nacs is not None and energy is not None:
            E = energy.reshape(-1)                          # (n_states,)
            i, j = np.triu_indices(E.size, k=1)             # (n_pairs,)
            gaps = np.abs(E[j] - E[i])                      # (n_pairs,)
            nacs = smooth_nacs / np.maximum(gaps, 1e-8)[None, :, None]
        # ---------------------

        self.results = {
            "energy": energy,          # (n_graphs, n_states)
            "free_energy": energy,     # same as energy (0 K)
            "forces": forces,          # (num_atoms, n_states, 3)
            "socs": socs,              # (n_state_pairs,)
            "nacs": nacs,              # (num_atoms, n_state_pairs, 3)
            "smooth_nacs": smooth_nacs,
        }

