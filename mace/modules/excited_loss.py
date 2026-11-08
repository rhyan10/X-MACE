###########################################################################################
# Loss functions for excited-state models (ExcitedMACE, AutoencoderExcitedMACE)
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Optional

import torch

from mace.tools import TensorDict
from mace.tools.torch_geometric import Batch

from .excited import align_dipoles, align_nacs, align_socs
from .loss import reduce_loss

# sqrt(x + eps) keeps the gradient of the RMSE-type coupling losses finite at zero
_SQRT_EPS = 1e-12


def _graph_weights(ref: Batch, name: str) -> torch.Tensor:
    """weight of every configuration for one property: config weight * property weight"""
    return ref.weight * ref[f"{name}_weight"]  # [n_graphs]


def _atom_weights(ref: Batch, name: str) -> torch.Tensor:
    return torch.repeat_interleave(
        _graph_weights(ref, name), ref.ptr[1:] - ref.ptr[:-1]
    )  # [n_atoms]


def excited_energy_loss(
    ref: Batch, pred: TensorDict, key: str = "energy", ddp: Optional[bool] = None
) -> torch.Tensor:
    """Weighted MSE of the total energies of all states [n_graphs, S]."""
    prediction = pred[key]
    target = ref["energy"].view(prediction.shape)
    weights = _graph_weights(ref, "energy").unsqueeze(-1)
    return reduce_loss(weights * torch.square(target - prediction), ddp)


def excited_forces_loss(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    """Weighted MSE of the forces of all states [n_atoms, S, 3]."""
    prediction = pred["forces"]
    target = ref["forces"].view(prediction.shape)
    weights = _atom_weights(ref, "forces").view(-1, 1, 1)
    return reduce_loss(weights * torch.square(target - prediction), ddp)


def latent_loss(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    """MSE between the encoded reference energies and the network's latent vector."""
    weights = _graph_weights(ref, "energy").unsqueeze(-1)
    return reduce_loss(
        weights * torch.square(pred["encoded_energy"] - pred["invariant_vals"]), ddp
    )


def phase_free_nacs_loss(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    """RMSE of the NACs [n_atoms, P, 3] with the best sign per configuration and pair."""
    target = ref["nacs"]
    prediction = pred["nacs"].view(target.shape)
    aligned = align_nacs(target, prediction, ref.batch, ref.num_graphs)
    err2 = torch.sum(torch.square(target - aligned), dim=-1)  # [n_atoms, P]
    weights = _atom_weights(ref, "nacs").unsqueeze(-1)
    return torch.sqrt(reduce_loss(weights * err2, ddp) + _SQRT_EPS)


def phase_free_socs_loss(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    """RMSE of the SOCs [n_graphs, L] with the best sign per configuration."""
    target = ref["socs"]
    prediction = pred["socs"].view(target.shape)
    aligned = align_socs(target, prediction)
    weights = _graph_weights(ref, "socs").unsqueeze(-1)
    return torch.sqrt(
        reduce_loss(weights * torch.square(target - aligned), ddp) + _SQRT_EPS
    )


def phase_free_dipoles_loss(
    ref: Batch, pred: TensorDict, n_permanent: int, ddp: Optional[bool] = None
) -> torch.Tensor:
    """MSE per atom of the dipoles [n_graphs, D, 3]; transition dipoles (after the
    first n_permanent) with the best sign per configuration."""
    target = ref["dipoles"]
    prediction = pred["dipoles"].view(target.shape)
    aligned = align_dipoles(target, prediction, n_permanent)
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)
    weights = _graph_weights(ref, "dipoles").view(-1, 1, 1)
    return reduce_loss(weights * torch.square((target - aligned) / num_atoms), ddp)


class WeightedExcitedStatesLoss(torch.nn.Module):
    """Loss of ExcitedMACE and AutoencoderExcitedMACE.

    energy_weight  * MSE(E)   [+ MSE(decoded E) + MSE(latent) for the autoencoder]
    forces_weight  * MSE(F)
    nacs_weight    * RMSE(NACs, phase free)
    socs_weight    * RMSE(SOCs, phase free)
    dipoles_weight * 100 * MSE(dipoles / n_atoms, transition dipoles phase free)

    Energies are total energies of all states, as in X-MACE. Per-configuration
    weights (config_weight, config_<property>_weight) are honoured; a missing
    label has weight 0. Terms the model does not predict are skipped.
    """

    def __init__(
        self,
        energy_weight: float = 1.0,
        forces_weight: float = 1.0,
        nacs_weight: float = 1.0,
        socs_weight: float = 1.0,
        dipoles_weight: float = 1.0,
        n_permanent_dipoles: int = 0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("energy_weight", energy_weight),
            ("forces_weight", forces_weight),
            ("nacs_weight", nacs_weight),
            ("socs_weight", socs_weight),
            ("dipoles_weight", dipoles_weight),
        ):
            self.register_buffer(
                name, torch.tensor(value, dtype=torch.get_default_dtype())
            )
        self.n_permanent_dipoles = n_permanent_dipoles

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss = torch.zeros((), dtype=ref.positions.dtype, device=ref.positions.device)
        if pred.get("energy") is not None and self.energy_weight > 0:
            energy_terms = excited_energy_loss(ref, pred, "energy", ddp)
            if pred.get("decoded_energy") is not None:
                energy_terms = (
                    energy_terms
                    + excited_energy_loss(ref, pred, "decoded_energy", ddp)
                    + latent_loss(ref, pred, ddp)
                )
            loss = loss + self.energy_weight * energy_terms
        if pred.get("forces") is not None and self.forces_weight > 0:
            loss = loss + self.forces_weight * excited_forces_loss(ref, pred, ddp)
        if pred.get("nacs") is not None and "nacs" in ref and self.nacs_weight > 0:
            loss = loss + self.nacs_weight * phase_free_nacs_loss(ref, pred, ddp)
        if pred.get("socs") is not None and "socs" in ref and self.socs_weight > 0:
            loss = loss + self.socs_weight * phase_free_socs_loss(ref, pred, ddp)
        if (
            pred.get("dipoles") is not None
            and "dipoles" in ref
            and self.dipoles_weight > 0
        ):
            loss = loss + self.dipoles_weight * 100.0 * phase_free_dipoles_loss(
                ref, pred, self.n_permanent_dipoles, ddp
            )
        return loss

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, nacs_weight={self.nacs_weight:.3f}, "
            f"socs_weight={self.socs_weight:.3f}, dipoles_weight={self.dipoles_weight:.3f})"
        )
