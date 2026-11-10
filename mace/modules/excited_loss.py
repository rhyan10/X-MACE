###########################################################################################
# Loss functions for excited-state models (ExcitedMACE, AutoencoderExcitedMACE)
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist

from mace.tools import TensorDict
from mace.tools.torch_geometric import Batch

from .excited import align_nacs, align_socs, phase_signs_per_graph
from .loss import is_ddp_enabled

# sqrt(x + eps) keeps the gradient of the RMSE-type coupling losses finite at zero
_SQRT_EPS = 1e-12

MASK_NAMES = (
    "energy",
    "nacs",
    "socs",
    "soc_vectors",
    "dipoles",
    "transition_dipoles",
    "oscillator_strengths",
)


def _graph_weights(ref: Batch, name: str) -> torch.Tensor:
    """weight of every configuration for one property: config weight * property weight"""
    return ref.weight * ref[f"{name}_weight"]  # [n_graphs]


def _atom_weights(ref: Batch, name: str) -> torch.Tensor:
    return torch.repeat_interleave(
        _graph_weights(ref, name), ref.ptr[1:] - ref.ptr[:-1]
    )  # [n_atoms]


def _graph_heads(ref: Batch) -> torch.Tensor:
    if "head" in ref and ref["head"] is not None:
        return ref["head"].to(torch.long).view(-1)
    return torch.zeros(ref.num_graphs, dtype=torch.long, device=ref.positions.device)


def head_mask(
    masks: Optional[Dict[str, torch.Tensor]],
    name: str,
    ref: Batch,
    width: int,
    per_atom: bool = False,
) -> torch.Tensor:
    """Mask [n_graphs or n_atoms, width] of the real (unpadded) entries of every
    configuration's head; all ones for a single-head model without masks."""
    n = ref.positions.shape[0] if per_atom else ref.num_graphs
    if masks is None or name not in masks:
        return torch.ones(
            n, width, dtype=ref.positions.dtype, device=ref.positions.device
        )
    heads = _graph_heads(ref)
    if per_atom:
        heads = heads[ref.batch]
    table = masks[name].to(ref.positions.device, ref.positions.dtype)
    return table[heads][:, :width]


def masked_mean(
    values: torch.Tensor, mask: torch.Tensor, ddp: Optional[bool]
) -> torch.Tensor:
    """sum(values) / number of unmasked entries (values already multiplied by mask)."""
    count = mask.sum() * (values.numel() / max(mask.numel(), 1))
    total = values.sum()
    ddp = is_ddp_enabled() if ddp is None else ddp
    if ddp and dist.is_initialized():
        count = count.clone()
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        return total * dist.get_world_size() / count.clamp(min=1.0)
    return total / count.clamp(min=1.0)


def masked_rmse(
    values: torch.Tensor, mask: torch.Tensor, ddp: Optional[bool]
) -> torch.Tensor:
    """sqrt(masked_mean(values) + eps), the RMSE of the coupling losses.

    Under DDP every rank only sees its share of the batch, and DDP averages the
    gradients of the ranks. A square root of each rank's own mean would give the
    gradient of the average of per-rank RMSEs, not of the RMSE of the whole batch.
    So every rank returns R + W (s_r - S / W) / (2 N R): R the RMSE of the whole batch
    (from all-reduced sums, no gradient), s_r its own sum, S the sum over all ranks,
    N the number of entries over all ranks, W the number of ranks. The average over
    the ranks is R, and the averaged gradient is that of R, as on one device."""
    ddp = is_ddp_enabled() if ddp is None else ddp
    if not (ddp and dist.is_initialized()):
        return torch.sqrt(masked_mean(values, mask, False) + _SQRT_EPS)
    count = mask.sum() * (values.numel() / max(mask.numel(), 1))
    local = values.sum()
    totals = torch.stack([local.detach(), count.detach().to(local.dtype)])
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    world = dist.get_world_size()
    total, n = totals[0], totals[1].clamp(min=1.0)
    rmse = torch.sqrt(total / n + _SQRT_EPS)
    return rmse + world * (local - total / world) / (2.0 * n * rmse)


def excited_energy_loss(
    ref: Batch,
    pred: TensorDict,
    key: str = "energy",
    ddp: Optional[bool] = None,
    masks: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Weighted MSE of the total energies of all states [n_graphs, S]."""
    prediction = pred[key]
    target = ref["energy"].view(prediction.shape)
    mask = head_mask(masks, "energy", ref, prediction.shape[1])
    weights = _graph_weights(ref, "energy").unsqueeze(-1) * mask
    return masked_mean(weights * torch.square(target - prediction), mask, ddp)


def excited_forces_loss(
    ref: Batch,
    pred: TensorDict,
    ddp: Optional[bool] = None,
    masks: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Weighted MSE of the forces of all states [n_atoms, S, 3]."""
    prediction = pred["forces"]
    target = ref["forces"].view(prediction.shape)
    mask = head_mask(masks, "energy", ref, prediction.shape[1], per_atom=True)
    weights = _atom_weights(ref, "forces").view(-1, 1, 1) * mask.unsqueeze(-1)
    return masked_mean(
        weights * torch.square(target - prediction), mask.unsqueeze(-1), ddp
    )


def oscillator_strengths_loss(
    ref: Batch,
    pred: TensorDict,
    ddp: Optional[bool] = None,
    masks: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Weighted MSE of the oscillator strengths [n_graphs, O] (no phase: f >= 0), in
    units of the model's scale per head and transition (pred["oscillator_strengths_scale"],
    e.g. the rms of the training labels), so that weak and bright transitions and
    data sets of different magnitude count alike."""
    target = ref["oscillator_strengths"]
    prediction = pred["oscillator_strengths"].view(target.shape)
    mask = head_mask(masks, "oscillator_strengths", ref, target.shape[1])
    weights = _graph_weights(ref, "oscillator_strengths").unsqueeze(-1) * mask
    error = target - prediction
    scale = pred.get("oscillator_strengths_scale")
    if scale is not None:
        error = error / scale.view(target.shape).clamp(min=1e-12)
    return masked_mean(weights * torch.square(error), mask, ddp)


def latent_loss(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    """MSE between the encoded reference energies and the network's latent vector."""
    weights = _graph_weights(ref, "energy").unsqueeze(-1)
    values = weights * torch.square(pred["encoded_energy"] - pred["invariant_vals"])
    return masked_mean(values, torch.ones_like(values), ddp)


def phase_free_nacs_loss(
    ref: Batch,
    pred: TensorDict,
    ddp: Optional[bool] = None,
    masks: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """RMSE of the NACs [n_atoms, P, 3] with the best sign per configuration and pair."""
    target = ref["nacs"]
    prediction = pred["nacs"].view(target.shape)
    aligned = align_nacs(target, prediction, ref.batch, ref.num_graphs)
    err2 = torch.sum(torch.square(target - aligned), dim=-1)  # [n_atoms, P]
    mask = head_mask(masks, "nacs", ref, err2.shape[1], per_atom=True)
    weights = _atom_weights(ref, "nacs").unsqueeze(-1) * mask
    return masked_rmse(weights * err2, mask, ddp)


def phase_free_socs_loss(
    ref: Batch,
    pred: TensorDict,
    ddp: Optional[bool] = None,
    masks: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """RMSE of the SOCs [n_graphs, L] with the best sign per configuration."""
    target = ref["socs"]
    prediction = pred["socs"].view(target.shape)
    mask = head_mask(masks, "socs", ref, target.shape[1])
    aligned = align_socs(target * mask, prediction * mask)
    weights = _graph_weights(ref, "socs").unsqueeze(-1) * mask
    return masked_rmse(weights * torch.square(target - aligned), mask, ddp)


def phase_free_soc_vectors_loss(
    ref: Batch,
    pred: TensorDict,
    pair_weights: Optional[torch.Tensor] = None,
    ddp: Optional[bool] = None,
    masks: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """RMSE of the spin-orbit coupling vectors [n_graphs, Q, 3], best sign per
    configuration and pair. pair_weights ([Q] or per head [H, Q], from
    mace.modules.spin_states.soc_pair_weights, normalised to mean 1) make it
    proportional to the RMSE of the M_S-resolved SOC matrix elements."""
    target = ref["soc_vectors"]
    prediction = pred["soc_vectors"].view(target.shape)
    signs = phase_signs_per_graph(target, prediction, None, target.shape[0])
    aligned = prediction * signs.unsqueeze(-1)
    err2 = torch.sum(torch.square(target - aligned), dim=-1)  # [n_graphs, Q]
    if pair_weights is not None:
        pair_weights = pair_weights.to(err2.device, err2.dtype)
        if pair_weights.dim() == 2:
            err2 = err2 * pair_weights[_graph_heads(ref)][:, : err2.shape[1]]
        else:
            err2 = err2 * pair_weights.unsqueeze(0)
    mask = head_mask(masks, "soc_vectors", ref, err2.shape[1])
    weights = _graph_weights(ref, "soc_vectors").unsqueeze(-1) * mask
    return masked_rmse(weights * err2, mask, ddp)


def align_dipoles_masked(
    ref: torch.Tensor, pred: torch.Tensor, transition: torch.Tensor
) -> torch.Tensor:
    """pred dipoles [n_graphs, D, 3]: entries with transition == 1 get the best sign
    per configuration (transition dipoles), the others are compared as they are."""
    signs = phase_signs_per_graph(ref, pred, None, pred.shape[0])  # [n_graphs, D]
    signs = torch.where(transition > 0, signs, torch.ones_like(signs))
    return pred * signs.unsqueeze(-1)


def transition_table(
    masks: Optional[Dict[str, torch.Tensor]],
    ref: Batch,
    width: int,
    n_permanent: int,
) -> torch.Tensor:
    if masks is not None and "transition_dipoles" in masks:
        return head_mask(masks, "transition_dipoles", ref, width)
    flags = torch.zeros(width, dtype=ref.positions.dtype, device=ref.positions.device)
    flags[n_permanent:] = 1.0
    return flags.unsqueeze(0).expand(ref.num_graphs, -1)


def phase_free_dipoles_loss(
    ref: Batch,
    pred: TensorDict,
    n_permanent: int,
    ddp: Optional[bool] = None,
    masks: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """MSE per atom of the dipoles [n_graphs, D, 3]; transition dipoles (after the
    first n_permanent, or per head from the masks) with the best sign per configuration.
    """
    target = ref["dipoles"]
    prediction = pred["dipoles"].view(target.shape)
    transition = transition_table(masks, ref, target.shape[1], n_permanent)
    aligned = align_dipoles_masked(target, prediction, transition)
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)
    mask = head_mask(masks, "dipoles", ref, target.shape[1]).unsqueeze(-1)
    weights = _graph_weights(ref, "dipoles").view(-1, 1, 1) * mask
    return masked_mean(
        weights * torch.square((target - aligned) / num_atoms), mask, ddp
    )


class WeightedExcitedStatesLoss(torch.nn.Module):
    """Loss of ExcitedMACE and AutoencoderExcitedMACE.

    energy_weight  * MSE(E)   [+ MSE(decoded E) + MSE(latent) for the autoencoder]
    forces_weight  * MSE(F)
    nacs_weight    * RMSE(NACs, phase free)
    socs_weight    * RMSE(SOCs, phase free); with soc_mode="spin_orbit" the RMSE of the
                     spin-orbit coupling vectors, weighted per pair in proportion to the
                     M_S-resolved SOC matrix elements each vector gives
    dipoles_weight * 100 * MSE(dipoles / n_atoms, transition dipoles phase free)
    oscillator_strengths_weight * MSE(oscillator strengths / their scale per transition)

    Energies are total energies of all states, as in X-MACE. Per-configuration
    weights (config_weight, config_<property>_weight) are honoured; a missing
    label has weight 0. Terms the model does not predict are skipped.

    Several heads: masks ([n_heads, X] tables, ExcitedMACE.mask_tables()) mark the
    real entries of every head, so padded states and couplings of a head with fewer
    of them (e.g. a ground-state head) do not count; means run over real entries.
    soc_pair_weights may be given per head ([n_heads, Q]) and soc_states per head.
    """

    def __init__(
        self,
        energy_weight: float = 1.0,
        forces_weight: float = 1.0,
        nacs_weight: float = 1.0,
        socs_weight: float = 1.0,
        dipoles_weight: float = 1.0,
        n_permanent_dipoles: int = 0,
        soc_pair_weights: Optional[Sequence] = None,
        soc_states: Optional[Sequence] = None,
        masks: Optional[Dict[str, torch.Tensor]] = None,
        oscillator_strengths_weight: float = 1.0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("energy_weight", energy_weight),
            ("forces_weight", forces_weight),
            ("nacs_weight", nacs_weight),
            ("socs_weight", socs_weight),
            ("dipoles_weight", dipoles_weight),
            ("oscillator_strengths_weight", oscillator_strengths_weight),
        ):
            self.register_buffer(
                name, torch.tensor(value, dtype=torch.get_default_dtype())
            )
        self.n_permanent_dipoles = n_permanent_dipoles
        # states per multiplicity, used to report SOC errors as matrix elements;
        # one list per head (or one list for a single head)
        self.soc_states: Optional[List[List[int]]] = None
        if soc_states:
            if isinstance(soc_states[0], (list, tuple)):
                self.soc_states = [list(s) for s in soc_states]
            else:
                self.soc_states = [list(soc_states)]
        pair_weights = None
        if soc_pair_weights is not None and len(soc_pair_weights) > 0:
            pair_weights = torch.tensor(
                np.asarray(soc_pair_weights, dtype=float),
                dtype=torch.get_default_dtype(),
            )
        self.register_buffer("soc_pair_weights", pair_weights)
        self.masks: Optional[Dict[str, torch.Tensor]] = None
        if masks is not None:
            self.masks = {}
            for name, table in masks.items():
                self.register_buffer(f"mask_{name}", table.clone())
                self.masks[name] = getattr(self, f"mask_{name}")

    def _current_masks(self) -> Optional[Dict[str, torch.Tensor]]:
        if self.masks is None:
            return None
        return {name: getattr(self, f"mask_{name}") for name in self.masks}

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        masks = self._current_masks()
        loss = torch.zeros((), dtype=ref.positions.dtype, device=ref.positions.device)
        if pred.get("energy") is not None and self.energy_weight > 0:
            energy_terms = excited_energy_loss(ref, pred, "energy", ddp, masks)
            if pred.get("decoded_energy") is not None:
                energy_terms = (
                    energy_terms
                    + excited_energy_loss(ref, pred, "decoded_energy", ddp, masks)
                    + latent_loss(ref, pred, ddp)
                )
            loss = loss + self.energy_weight * energy_terms
        if pred.get("forces") is not None and self.forces_weight > 0:
            loss = loss + self.forces_weight * excited_forces_loss(
                ref, pred, ddp, masks
            )
        if pred.get("nacs") is not None and "nacs" in ref and self.nacs_weight > 0:
            loss = loss + self.nacs_weight * phase_free_nacs_loss(ref, pred, ddp, masks)
        if pred.get("socs") is not None and "socs" in ref and self.socs_weight > 0:
            loss = loss + self.socs_weight * phase_free_socs_loss(ref, pred, ddp, masks)
        if (
            pred.get("soc_vectors") is not None
            and "soc_vectors" in ref
            and self.socs_weight > 0
        ):
            loss = loss + self.socs_weight * phase_free_soc_vectors_loss(
                ref, pred, self.soc_pair_weights, ddp, masks
            )
        if (
            pred.get("dipoles") is not None
            and "dipoles" in ref
            and self.dipoles_weight > 0
        ):
            loss = loss + self.dipoles_weight * 100.0 * phase_free_dipoles_loss(
                ref, pred, self.n_permanent_dipoles, ddp, masks
            )
        if (
            pred.get("oscillator_strengths") is not None
            and "oscillator_strengths" in ref
            and self.oscillator_strengths_weight > 0
        ):
            loss = loss + self.oscillator_strengths_weight * oscillator_strengths_loss(
                ref, pred, ddp, masks
            )
        return loss

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, nacs_weight={self.nacs_weight:.3f}, "
            f"socs_weight={self.socs_weight:.3f}, dipoles_weight={self.dipoles_weight:.3f}, "
            f"oscillator_strengths_weight={self.oscillator_strengths_weight:.3f})"
        )
