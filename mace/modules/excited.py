###########################################################################################
# X-MACE: MACE models for electronically excited states
#
#   ExcitedMACE             energies (+ forces) of several states, NACs, SOCs and dipoles
#   AutoencoderExcitedMACE  energies from the eigenvalues of a learnt matrix (deep-set
#                           autoencoder), which keeps them smooth near conical intersections
#
# Both sit on top of a ground-state MACE model of any class (MACE, ScaleShiftMACE,
# MACELES, PolarMACE, MagneticScaleShiftMACE), built from scratch or from a foundation
# model, and support several heads (excited-state and ground-state data sets at once).
#
# Ported from X-MACE (https://github.com/rhyan10/X-MACE, branch X-MACE_socs).
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from e3nn import nn, o3
from e3nn.util.jit import compile_mode

from mace.tools.scatter import scatter_sum

from .blocks import (
    EquivariantProductBasisBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    RealAgnosticResidualInteractionBlock,
)
from .spin_states import parse_states, same_multiplicity_pairs, spin_orbit_pairs
from .utils import get_edge_vectors_and_lengths

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def num_state_pairs(n_states: int) -> int:
    return n_states * (n_states - 1) // 2


def compute_multistate_forces(
    energy: torch.Tensor,  # [n_graphs, n_states]
    positions: torch.Tensor,  # [n_nodes, 3]
    training: bool = False,
) -> torch.Tensor:  # [n_nodes, n_states, 3]
    """Forces -dE_s/dr for every state s (one backward pass per state)."""
    n_states = energy.shape[1]
    forces: List[torch.Tensor] = []
    for state in range(n_states):
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy[:, state])]
        gradient = torch.autograd.grad(
            outputs=[energy[:, state]],
            inputs=[positions],
            grad_outputs=grad_outputs,
            retain_graph=(training or state < n_states - 1),
            create_graph=training,
            allow_unused=True,
        )[0]
        if gradient is None:
            gradient = torch.zeros_like(positions)
        forces.append(-1.0 * gradient)
    return torch.stack(forces, dim=1)


def phase_signs_per_graph(
    ref: torch.Tensor,  # [n_nodes, n_pairs, 3]  (per atom) or [n_graphs, n_pairs, 3]
    pred: torch.Tensor,
    batch: Optional[torch.Tensor],
    num_graphs: int,
) -> torch.Tensor:  # [n_graphs, n_pairs]
    """Sign (+1/-1) of pred that best matches ref for every graph and coupling.

    The phase of a coupling between two electronic states is arbitrary, so one
    sign is chosen per configuration and coupling (the same for all atoms and
    Cartesian components)."""
    err_minus = torch.sum(torch.square(ref - pred), dim=-1)
    err_plus = torch.sum(torch.square(ref + pred), dim=-1)
    if batch is not None:
        err_minus = scatter_sum(err_minus, batch, dim=0, dim_size=num_graphs)
        err_plus = scatter_sum(err_plus, batch, dim=0, dim_size=num_graphs)
    return torch.where(
        err_plus < err_minus, -torch.ones_like(err_plus), torch.ones_like(err_plus)
    )


def align_nacs(
    ref: torch.Tensor, pred: torch.Tensor, batch: torch.Tensor, num_graphs: int
) -> torch.Tensor:
    """pred NACs [n_nodes, n_pairs, 3] with the phase that best matches ref."""
    signs = phase_signs_per_graph(ref, pred, batch, num_graphs)  # [n_graphs, n_pairs]
    return pred * signs[batch].unsqueeze(-1)


def align_socs(ref: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """pred SOCs [n_graphs, n_socs] with one phase per configuration (X-MACE layout)."""
    err_minus = torch.sum(torch.square(ref - pred), dim=-1, keepdim=True)
    err_plus = torch.sum(torch.square(ref + pred), dim=-1, keepdim=True)
    return torch.where(err_plus < err_minus, -pred, pred)


def align_dipoles(
    ref: torch.Tensor, pred: torch.Tensor, n_permanent: int
) -> torch.Tensor:
    """pred dipoles [n_graphs, n_dipoles, 3]: the first n_permanent are compared as
    they are, every transition dipole after them gets its own best phase."""
    if n_permanent >= pred.shape[1]:
        return pred
    transition = pred[:, n_permanent:]
    signs = phase_signs_per_graph(ref[:, n_permanent:], transition, None, pred.shape[0])
    return torch.cat([pred[:, :n_permanent], transition * signs.unsqueeze(-1)], dim=1)


def align_soc_vectors(ref: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """pred spin-orbit coupling vectors [n_graphs, Q, 3] with the best sign per
    configuration and pair of spatial states (the SOC block of a pair changes sign
    with the phase of either state)."""
    return align_dipoles(ref, pred, 0)


# --------------------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------------------


@compile_mode("script")
class NonLinearVectorReadoutBlock(torch.nn.Module):
    """Gated non-linear readout to vectors, e.g. irreps_out = "P x 1o"."""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        irreps_out: o3.Irreps,
    ):
        super().__init__()
        num_hidden = max(1, o3.Irreps(MLP_irreps).count(o3.Irrep(0, 1)))
        self.irreps_out = o3.Irreps(irreps_out)
        self.hidden_irreps = o3.Irreps(f"{num_hidden}x1o")
        self.equivariant_nonlin = nn.Gate(
            irreps_scalars=o3.Irreps(""),
            act_scalars=[],
            irreps_gates=o3.Irreps(f"{num_hidden}x0e"),
            act_gates=[gate],
            irreps_gated=self.hidden_irreps,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=self.irreps_nonlin)
        self.linear_2 = o3.Linear(
            irreps_in=self.hidden_irreps, irreps_out=self.irreps_out
        )

    def forward(
        self,
        x: torch.Tensor,
        heads: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        return self.linear_2(self.equivariant_nonlin(self.linear_1(x)))


@compile_mode("script")
class PairVectorReadoutBlock(torch.nn.Module):
    """Readout to coupling vectors of either parity: "1e" (axial), "1o" (polar) or
    "1o+1e" (both, summed per vector).

    Axial vectors come from cross products of polar l=1 features (1o x 1o -> 1e), so
    the block works with the usual hidden irreps. The features are first projected
    to K scalars and K vectors (K = number of MLP scalars), then combined by a tensor
    product. Returns Cartesian vectors [n_nodes, num_out * 3]."""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        num_out: int,
        parities: str = "1e",
    ):
        super().__init__()
        if parities not in ("1e", "1o", "1o+1e"):
            raise ValueError(f"parities must be '1e', '1o' or '1o+1e', got {parities}")
        irreps_in = o3.Irreps(irreps_in)
        num_hidden = max(1, o3.Irreps(MLP_irreps).count(o3.Irrep(0, 1)))
        hidden = f"{num_hidden}x0e + {num_hidden}x1o"
        if irreps_in.count(o3.Irrep(1, 1)) > 0:
            hidden += f" + {num_hidden}x1e"
        self.hidden_irreps = o3.Irreps(hidden)
        self.num_out = num_out
        self.n_parities = len(parities.split("+"))
        self.irreps_out = o3.Irreps(
            " + ".join(f"{num_out}x{p}" for p in parities.split("+"))
        )
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=self.hidden_irreps)
        self.tensor_product = o3.FullyConnectedTensorProduct(
            self.hidden_irreps, self.hidden_irreps, self.irreps_out
        )

    def forward(
        self,
        x: torch.Tensor,
        heads: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        hidden = self.linear(x)
        out = self.tensor_product(hidden, hidden)
        # [n, parity, num_out * 3] -> sum over the parities
        return out.view(out.shape[0], self.n_parities, self.num_out * 3).sum(dim=1)


@compile_mode("script")
class ExcitedScaleShiftBlock(torch.nn.Module):
    """scale * x + shift_s for the atomic interaction energies x [n_nodes, n_states]."""

    def __init__(
        self, scale: float, shift: Union[float, Sequence[float]], n_states: int
    ):
        super().__init__()
        shift_tensor = torch.as_tensor(
            np.asarray(shift, dtype=float), dtype=torch.get_default_dtype()
        ).reshape(-1)
        if shift_tensor.numel() == 1:
            shift_tensor = shift_tensor.repeat(n_states)
        if shift_tensor.numel() != n_states:
            raise ValueError(
                f"atomic_inter_shift needs 1 or {n_states} values, got {shift_tensor.numel()}"
            )
        scale_tensor = torch.as_tensor(
            np.asarray(scale, dtype=float), dtype=torch.get_default_dtype()
        ).reshape(-1)
        if scale_tensor.numel() != 1:
            raise ValueError("atomic_inter_scale must be a single number")
        self.register_buffer("scale", scale_tensor[0].clone())
        self.register_buffer("shift", shift_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.shift

    def __repr__(self):
        shifts = ", ".join(f"{s:.4f}" for s in self.shift.tolist())
        return f"{self.__class__.__name__}(scale={self.scale.item():.4f}, shift=[{shifts}])"


@compile_mode("script")
class PermutationInvariantEncoder(torch.nn.Module):
    """Deep set over the electronic states: energies [B, S] -> latent [B, K].

    With several multiplicities (groups: the group of every state), each energy is
    tagged with its multiplicity, so the encoding is invariant to permutations of
    states within a multiplicity only."""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 16,
        groups: Optional[List[int]] = None,
    ):
        super().__init__()
        act = torch.nn.ELU
        groups = list(groups) if groups is not None else []
        n_groups = max(groups) + 1 if groups else 1
        self.n_groups = n_groups
        tags = torch.zeros(len(groups), n_groups)
        for state, group in enumerate(groups):
            tags[state, group] = 1.0
        self.register_buffer("group_tags", tags)
        self.elementwise_nn = torch.nn.Sequential(
            torch.nn.Linear(1 + (n_groups if n_groups > 1 else 0), hidden_dim),
            act(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            act(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            act(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            act(),
        )
        self.post_aggregation_nn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            act(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            act(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            act(),
            torch.nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, energies: torch.Tensor) -> torch.Tensor:
        features = energies.unsqueeze(-1)  # [B, S, 1]
        if self.n_groups > 1:
            tags = self.group_tags.to(energies.dtype).unsqueeze(0)
            features = torch.cat(
                [features, tags.expand(energies.shape[0], -1, -1)], dim=-1
            )
        per_state = self.elementwise_nn(features)  # [B, S, H]
        return self.post_aggregation_nn(per_state.sum(dim=1))  # [B, K]


@compile_mode("script")
class PermutationInvariantDecoder(torch.nn.Module):
    """latent [B, K] -> symmetric tridiagonal matrix -> sorted eigenvalues [B, S].

    With several multiplicities (block_sizes: states per multiplicity, in model
    order) there is one tridiagonal block per multiplicity, so the states are ordered
    within each multiplicity, as SHARC's MCH states are."""

    def __init__(
        self,
        latent_dim: int = 16,
        n_energies: int = 3,
        hidden_dim: int = 128,
        block_sizes: Optional[List[int]] = None,
    ):
        super().__init__()
        self.n_energies = n_energies
        self.block_sizes: List[int] = (
            [int(n) for n in block_sizes if n > 0] if block_sizes else [n_energies]
        )
        if sum(self.block_sizes) != n_energies:
            raise ValueError(f"block sizes {self.block_sizes} != {n_energies} states")
        n_values = sum(2 * n - 1 for n in self.block_sizes)
        act = torch.nn.ELU
        self.decoder_nn = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            act(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            act(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            act(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            act(),
            torch.nn.Linear(hidden_dim, n_values),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        values = self.decoder_nn(latent)
        if len(self.block_sizes) == 1:
            diagonal = values[:, : self.n_energies]
            off_diagonal = values[:, self.n_energies :]
            matrix = (
                torch.diag_embed(diagonal)
                + torch.diag_embed(off_diagonal, offset=1)
                + torch.diag_embed(off_diagonal, offset=-1)
            )
            return torch.linalg.eigvalsh(matrix)  # ascending order
        energies: List[torch.Tensor] = []
        start = 0
        for n in self.block_sizes:
            diagonal = values[:, start : start + n]
            off_diagonal = values[:, start + n : start + 2 * n - 1]
            matrix = (
                torch.diag_embed(diagonal)
                + torch.diag_embed(off_diagonal, offset=1)
                + torch.diag_embed(off_diagonal, offset=-1)
            )
            energies.append(torch.linalg.eigvalsh(matrix))
            start += 2 * n - 1
        return torch.cat(energies, dim=1)


# --------------------------------------------------------------------------------------
# Heads: what every head of an excited-state model predicts
# --------------------------------------------------------------------------------------


def excited_head_layout(
    n_states: int = 1,
    states: Optional[Sequence[int]] = None,
    compute_nacs: bool = False,
    nac_num: Optional[int] = None,
    compute_socs: bool = False,
    soc_num: Optional[int] = None,
    soc_mode: str = "free",
    compute_dipoles: bool = False,
    dipole_num: Optional[int] = None,
) -> Dict[str, Any]:
    """Output layout of one head: number of states, multiplicities, and the numbers of
    NACs, free SOCs, SOC vectors and dipoles (0: not predicted). A ground-state head
    is excited_head_layout(1)."""
    if n_states < 1:
        raise ValueError(f"n_energies must be at least 1, got {n_states}")
    if states is not None:
        states = parse_states(states)
        if sum(states) != n_states:
            raise ValueError(
                f"states {states} describe {sum(states)} states, but n_energies={n_states}"
            )
    layout = list(states) if states is not None else [n_states]
    n_pairs = len(same_multiplicity_pairs(layout))
    if soc_mode not in ("free", "spin_orbit"):
        raise ValueError(f"soc_mode must be 'free' or 'spin_orbit', got {soc_mode!r}")
    soc_vector_num = 0
    if compute_socs and soc_mode == "spin_orbit":
        if states is None:
            raise ValueError("soc_mode='spin_orbit' needs the states per multiplicity")
        soc_vector_num = len(spin_orbit_pairs(layout))
        if soc_vector_num < 1:
            raise ValueError(f"No spin-orbit-coupled pairs of states in {layout}")
        compute_socs = False
    nac_num = (nac_num if nac_num is not None else n_pairs) if compute_nacs else 0
    soc_num = (
        (soc_num if soc_num is not None else num_state_pairs(n_states))
        if compute_socs
        else 0
    )
    dipole_num = (
        (dipole_num if dipole_num is not None else n_states + n_pairs)
        if compute_dipoles
        else 0
    )
    if compute_nacs and nac_num < 1:
        raise ValueError("compute_nacs needs n_energies >= 2 or nac_num >= 1")
    if compute_socs and soc_num < 1:
        raise ValueError("compute_socs needs n_energies >= 2 or soc_num >= 1")
    if compute_dipoles and dipole_num < 1:
        raise ValueError("compute_dipoles needs dipole_num >= 1")
    n_permanent = n_states if dipole_num == n_states + n_pairs else dipole_num
    return {
        "n_states": int(n_states),
        "states": layout,
        "nac_num": int(nac_num),
        "soc_num": int(soc_num),
        "soc_vector_num": int(soc_vector_num),
        "dipole_num": int(dipole_num),
        "n_permanent_dipoles": int(n_permanent),
    }


def _index_table(sizes: List[int]) -> torch.Tensor:
    """[H, max(sizes)] indices into the concatenation of all heads' outputs, shifted
    by one: index 0 is a zero column used for padding."""
    width = max(max(sizes), 1)
    table = torch.zeros(len(sizes), width, dtype=torch.long)
    start = 1
    for h, size in enumerate(sizes):
        table[h, :size] = torch.arange(start, start + size)
        start += size
    return table


def _mask_table(sizes: List[int]) -> torch.Tensor:
    width = max(max(sizes), 1)
    mask = torch.zeros(len(sizes), width)
    for h, size in enumerate(sizes):
        mask[h, :size] = 1.0
    return mask


def gather_heads(
    values: torch.Tensor,  # [n, T] or [n, T, 3] (all heads concatenated)
    table: torch.Tensor,  # [H, X]
    heads: torch.Tensor,  # [n]
) -> torch.Tensor:  # [n, X] or [n, X, 3]
    """Pick every row's own head from outputs concatenated over heads (padded)."""
    index = table[heads]  # [n, X]
    if values.dim() == 3:
        padded = torch.cat([torch.zeros_like(values[:, :1]), values], dim=1)
        return torch.gather(
            padded, 1, index.unsqueeze(-1).expand(-1, -1, values.shape[-1])
        )
    padded = torch.cat([torch.zeros_like(values[:, :1]), values], dim=1)
    return torch.gather(padded, 1, index)


def coulomb_energies(
    charges: torch.Tensor,  # [n_nodes, M] (M charge sets, e.g. one per state)
    positions: torch.Tensor,  # [n_nodes, 3]
    batch: torch.Tensor,
    num_graphs: int,
    sigma: float = 1.0,
) -> torch.Tensor:  # [n_graphs, M] in eV
    """Electrostatic energy of Gaussian charges (width sigma, Å) within each molecule
    (no periodic images): 1/2 sum_ij q_i q_j erf(r_ij / 2 sigma) / r_ij."""
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    distance = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-12)
    same = batch.unsqueeze(1) == batch.unsqueeze(0)
    eye = torch.eye(positions.shape[0], dtype=torch.bool, device=positions.device)
    kernel = torch.erf(distance / (2.0 * sigma)) / distance
    kernel = torch.where(same & ~eye, kernel, torch.zeros_like(kernel))
    per_atom = 0.5 * 14.399645 * charges * (kernel @ charges)  # [n_nodes, M]
    return scatter_sum(per_atom, batch, dim=0, dim_size=num_graphs)


def mask_tables_from_layouts(layouts: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """[H, X] masks of the real (unpadded) entries of every head's outputs."""
    dipoles = [layout["dipole_num"] for layout in layouts]
    permanent = [layout["n_permanent_dipoles"] for layout in layouts]
    transition = torch.zeros(len(layouts), max(max(dipoles), 1))
    for h, (p, d) in enumerate(zip(permanent, dipoles)):
        transition[h, p:d] = 1.0
    return {
        "energy": _mask_table([layout["n_states"] for layout in layouts]),
        "nacs": _mask_table([layout["nac_num"] for layout in layouts]),
        "socs": _mask_table([layout["soc_num"] for layout in layouts]),
        "soc_vectors": _mask_table([layout["soc_vector_num"] for layout in layouts]),
        "dipoles": _mask_table(dipoles),
        "transition_dipoles": transition,
    }


@compile_mode("script")
class BackboneCall(torch.nn.Module):
    """Runs the ground-state MACE model (any class) inside an excited-state model.

    head_map[h] is the backbone head used for frames of excited-model head h (by
    default the same index), e.g. several heads sharing a foundation model's head."""

    def __init__(self, model: torch.nn.Module, head_map: Optional[List[int]] = None):
        super().__init__()
        self.model = model
        num_heads = len(getattr(model, "heads", ["Default"]))
        if head_map is None:
            head_map = list(range(num_heads))
        if any(h < 0 or h >= num_heads for h in head_map):
            raise ValueError(f"head_map {head_map} for a backbone with {num_heads} heads")
        self.register_buffer("head_map", torch.tensor(head_map, dtype=torch.long))
        self.identity_heads = list(head_map) == list(range(num_heads))

    def _map_heads(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.identity_heads or "head" not in data:
            return data
        mapped = data.copy()
        mapped["head"] = self.head_map[data["head"].to(torch.long)]
        return mapped

    def forward(
        self, data: Dict[str, torch.Tensor], training: bool
    ) -> Dict[str, Optional[torch.Tensor]]:
        return self.model(self._map_heads(data), training=training, compute_force=False)


@compile_mode("script")
class MagneticBackboneCall(BackboneCall):
    def forward(
        self, data: Dict[str, torch.Tensor], training: bool
    ) -> Dict[str, Optional[torch.Tensor]]:
        return self.model(
            self._map_heads(data),
            training=training,
            compute_force=False,
            compute_magforces=False,
        )


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------


@compile_mode("script")
class ExcitedMACE(torch.nn.Module):
    """Excited states on top of any ground-state MACE model.

    The backbone is a ground-state model of any class (MACE, ScaleShiftMACE, MACELES,
    PolarMACE, MagneticScaleShiftMACE, ...), built from scratch or taken from a
    foundation model, with one head per data set. For a configuration of head h,

        E_{h,s}(R) = E_backbone,h(R) + scale_h * sum_layers readout_{h,s}(R) + n_atoms * shift_{h,s}
                     [+ the state-specific electrostatic correction, long_range_mode="state"]

    so every state starts from the backbone's surface (a foundation model's ground
    state), and a ground-state head (one state, no couplings) is the backbone's own
    head plus a correction. Heads have their own number of states and couplings; the
    outputs of a batch are padded to the largest head:

      energy  [B, S_max]   forces [N, S_max, 3]   nacs [N, P_max, 3]
      socs    [B, L_max]   soc_vectors [B, Q_max, 3]   dipoles [B, D_max, 3]

    model.head_layouts says what each head predicts; mask_tables() which entries are
    real. An equivariant adapter layer (one MACE interaction and product on top of the
    backbone's features) provides l=1 features for the couplings, also for backbones
    without them (e.g. scalar-only foundation models).

    long_range_mode
      "shared"  the backbone's long-range/electrostatic energy (PolarMACE, MACELES) is
                part of E_backbone and the same for every state;
      "state"   every state additionally has its own charges q_s = q_backbone + dq_s
                (dq_s from the readouts, summing to zero per molecule) and the
                electrostatic energy changes by C(q_s) - C(q_backbone), with C the
                Coulomb energy of Gaussian charges; molecules (no periodic cells) only.

    coupling_parity: a coupling between two states changes sign with the phase of
    either state, and an improper operation can flip those phases, so a coupling vector
    may behave as a polar or an axial vector. "both" (default) gives every coupling
    vector a polar and an axial part (equivariant under rotations); "fixed" keeps NACs
    and transition dipoles polar and SOC vectors axial. Permanent dipoles are polar.

    Without a backbone the constructor builds one (ScaleShiftMACE by default) from the
    usual MACE arguments, and the single head is described by n_energies, states,
    compute_nacs, ... as before. With a backbone, heads (names) and backbone_heads (the
    backbone head of each) let the model have other heads than the backbone, e.g. a
    foundation model used as it is, all heads on its default head.
    """

    def __init__(
        self,
        backbone: Optional[torch.nn.Module] = None,
        head_layouts: Optional[List[Dict[str, Any]]] = None,
        MLP_irreps: Optional[o3.Irreps] = None,
        gate: Optional[Callable] = None,
        adapter_irreps: Optional[o3.Irreps] = None,
        adapter_max_ell: int = 3,
        adapter_correlation: int = 2,
        adapter_num_bessel: int = 8,
        adapter_num_polynomial_cutoff: int = 5,
        long_range_mode: str = "shared",
        charge_smearing: float = 1.0,
        coupling_parity: str = "both",
        delta_scale: Optional[Sequence[float]] = None,
        delta_shift: Optional[Sequence[Sequence[float]]] = None,
        backbone_cls: Optional[Type[torch.nn.Module]] = None,
        # single-head arguments (models built without head_layouts)
        n_energies: int = 1,
        compute_nacs: bool = False,
        nac_num: Optional[int] = None,
        compute_socs: bool = False,
        soc_num: Optional[int] = None,
        compute_dipoles: bool = False,
        dipole_num: Optional[int] = None,
        states: Optional[List[int]] = None,
        soc_mode: str = "free",
        atomic_inter_scale: Union[float, Sequence[float]] = 1.0,
        atomic_inter_shift: Union[float, Sequence[float]] = 0.0,
        heads: Optional[List[str]] = None,
        keep_last_layer_irreps: Optional[bool] = None,
        backbone_heads: Optional[List[int]] = None,
        **backbone_kwargs: Any,
    ):
        super().__init__()
        if coupling_parity not in ("both", "fixed"):
            raise ValueError(
                f"coupling_parity must be 'both' or 'fixed', got {coupling_parity!r}"
            )
        if long_range_mode not in ("shared", "state"):
            raise ValueError(
                f"long_range_mode must be 'shared' or 'state', got {long_range_mode!r}"
            )
        if head_layouts is None:
            head_layouts = [
                excited_head_layout(
                    n_states=n_energies,
                    states=states,
                    compute_nacs=compute_nacs,
                    nac_num=nac_num,
                    compute_socs=compute_socs,
                    soc_num=soc_num,
                    soc_mode=soc_mode,
                    compute_dipoles=compute_dipoles,
                    dipole_num=dipole_num,
                )
            ]
        head_layouts = [dict(layout) for layout in head_layouts]
        if MLP_irreps is None:
            MLP_irreps = backbone_kwargs.get("MLP_irreps", o3.Irreps("16x0e"))
        MLP_irreps = o3.Irreps(str(MLP_irreps))
        if gate is None:
            gate = backbone_kwargs.get("gate", torch.nn.functional.silu)

        if backbone is None:
            backbone, head_names, delta_shift_default = self._build_backbone(
                backbone_cls,
                backbone_kwargs,
                head_layouts,
                heads,
                atomic_inter_scale,
                atomic_inter_shift,
                keep_last_layer_irreps,
                MLP_irreps,
                gate,
            )
            if delta_shift is None:
                delta_shift = delta_shift_default
        elif heads is not None:
            # own head names; backbone_heads maps them to the backbone's heads
            head_names = list(heads)
            if backbone_heads is None:
                if len(head_names) != len(getattr(backbone, "heads", ["Default"])):
                    backbone_heads = [0] * len(head_names)
        else:
            head_names = list(getattr(backbone, "heads", ["Default"]))
        if len(head_layouts) == 1 and len(head_names) > 1:
            head_layouts = head_layouts * len(head_names)
        if len(head_layouts) != len(head_names):
            raise ValueError(
                f"{len(head_layouts)} head layouts for the backbone heads {head_names}"
            )

        magnetic = backbone.__class__.__name__.startswith("Magnetic")
        self.backbone = (
            MagneticBackboneCall(backbone, backbone_heads)
            if magnetic
            else BackboneCall(backbone, backbone_heads)
        )
        self.backbone_class = backbone.__class__.__name__
        self.register_buffer("atomic_numbers", backbone.atomic_numbers.clone())
        self.register_buffer("r_max", backbone.r_max.clone())
        self.heads = head_names
        self.head_layouts = head_layouts
        self.coupling_parity = coupling_parity
        self.long_range_mode = long_range_mode
        self.charge_smearing = float(charge_smearing)
        self.backbone_layer_dims: List[int] = [
            o3.Irreps(str(product.linear.irreps_out)).dim
            for product in backbone.products
        ]
        backbone_irreps = [
            o3.Irreps(str(product.linear.irreps_out)) for product in backbone.products
        ]

        # per-head layout tables
        n_states = [layout["n_states"] for layout in head_layouts]
        nacs = [layout["nac_num"] for layout in head_layouts]
        socs = [layout["soc_num"] for layout in head_layouts]
        soc_vectors = [layout["soc_vector_num"] for layout in head_layouts]
        dipoles = [layout["dipole_num"] for layout in head_layouts]
        permanent = [layout["n_permanent_dipoles"] for layout in head_layouts]
        transitions = [d - p for d, p in zip(dipoles, permanent)]
        self.head_n_states: List[int] = n_states
        self.head_states: List[List[int]] = [
            list(layout["states"]) for layout in head_layouts
        ]
        self.n_energies = max(n_states)
        self.nac_num = max(nacs)
        self.soc_num = max(socs)
        self.soc_vector_num = max(soc_vectors)
        self.dipole_num = max(dipoles)
        self.compute_nacs = self.nac_num > 0
        self.compute_socs = self.soc_num > 0
        self.compute_soc_vectors = self.soc_vector_num > 0
        self.compute_dipoles = self.dipole_num > 0
        default = next((h for h, name in enumerate(head_names) if name != "pt_head"), 0)
        self.default_head = default
        # single-head view (calculators, SHARC): the default head
        self.multiplicity_states: List[int] = list(head_layouts[default]["states"])
        self.soc_mode = (
            "spin_orbit" if head_layouts[default]["soc_vector_num"] > 0 else "free"
        )
        self.n_permanent_dipoles = permanent[default]
        self.register_buffer("energy_index", _index_table(n_states))
        self.register_buffer("nac_index", _index_table(nacs))
        self.register_buffer("soc_index", _index_table(socs))
        self.register_buffer("soc_vector_index", _index_table(soc_vectors))
        self.register_buffer("dipole_index", _index_table(dipoles))
        transition_table = torch.zeros(
            len(head_layouts), max(max(dipoles), 1), dtype=torch.long
        )
        start = 1
        for h, (p, t) in enumerate(zip(permanent, transitions)):
            transition_table[h, p : p + t] = torch.arange(start, start + t)
            start += t
        self.register_buffer("transition_index", transition_table)
        self.n_transition_total = int(sum(transitions))

        scale = [1.0] * len(head_layouts) if delta_scale is None else list(delta_scale)
        if len(scale) == 1 and len(head_layouts) > 1:
            scale = scale * len(head_layouts)
        shift = torch.zeros(len(head_layouts), self.n_energies)
        if delta_shift is not None:
            for h, values in enumerate(delta_shift):
                values = list(np.atleast_1d(np.asarray(values, dtype=float)))
                shift[h, : len(values)] = torch.tensor(values[: self.n_energies])
        self.register_buffer(
            "delta_scale", torch.tensor(scale, dtype=torch.get_default_dtype())
        )
        self.register_buffer("delta_shift", shift.to(torch.get_default_dtype()))

        # adapter: one MACE layer on top of the backbone, with l = 1 features
        num_elements = len(backbone.atomic_numbers)
        num_channels = max(1, MLP_irreps.count(o3.Irrep(0, 1)))
        last = backbone_irreps[-1]
        if adapter_irreps is None:
            adapter_irreps = o3.Irreps(f"{num_channels}x0e + {num_channels}x1o")
        adapter_irreps = o3.Irreps(str(adapter_irreps))
        channels = adapter_irreps.count(o3.Irrep(0, 1))
        adapter_in = f"{channels}x0e"
        if last.count(o3.Irrep(1, -1)) > 0:
            adapter_in += f" + {channels}x1o"
        self.adapter_in_irreps = o3.Irreps(adapter_in)
        self.adapter_in = o3.Linear(last, self.adapter_in_irreps)
        self.adapter_radial = RadialEmbeddingBlock(
            r_max=float(backbone.r_max),
            num_bessel=adapter_num_bessel,
            num_polynomial_cutoff=adapter_num_polynomial_cutoff,
        )
        sh_irreps = o3.Irreps.spherical_harmonics(adapter_max_ell)
        self.adapter_sh = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        target_irreps = (sh_irreps * channels).sort()[0].simplify()
        self.adapter_interaction = RealAgnosticResidualInteractionBlock(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=self.adapter_in_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=o3.Irreps(f"{self.adapter_radial.out_dim}x0e"),
            target_irreps=target_irreps,
            hidden_irreps=adapter_irreps,
            avg_num_neighbors=float(
                backbone.interactions[0].avg_num_neighbors
                if hasattr(backbone.interactions[0], "avg_num_neighbors")
                else 10.0
            ),
        )
        self.adapter_product = EquivariantProductBasisBlock(
            node_feats_irreps=target_irreps,
            target_irreps=adapter_irreps,
            correlation=adapter_correlation,
            num_elements=num_elements,
            use_sc=True,
        )
        feature_irreps = backbone_irreps + [adapter_irreps]

        # readouts (outputs of all heads concatenated)
        self.readouts = self._scalar_readouts(
            feature_irreps, MLP_irreps, gate, self._energy_readout_total()
        )
        total = int(sum(socs))
        self.socs_readouts = torch.nn.ModuleList()
        if total > 0:
            self.socs_readouts = self._scalar_readouts(
                feature_irreps, MLP_irreps, gate, total
            )
        both = coupling_parity == "both"
        total = int(sum(nacs))
        self.nacs_readouts = torch.nn.ModuleList()
        self.nacs_axial_readouts = torch.nn.ModuleList()
        if total > 0:
            self.nacs_readouts = self._vector_readouts(
                feature_irreps, MLP_irreps, gate, total
            )
            if both:
                self.nacs_axial_readouts = self._pair_readouts(
                    feature_irreps, MLP_irreps, total, "1e"
                )
        total = int(sum(dipoles))
        self.dipoles_readouts = torch.nn.ModuleList()
        self.dipoles_axial_readouts = torch.nn.ModuleList()
        if total > 0:
            self.dipoles_readouts = self._vector_readouts(
                feature_irreps, MLP_irreps, gate, total
            )
            if both and self.n_transition_total > 0:
                self.dipoles_axial_readouts = self._pair_readouts(
                    feature_irreps, MLP_irreps, self.n_transition_total, "1e"
                )
        total = int(sum(soc_vectors))
        self.soc_vectors_readouts = torch.nn.ModuleList()
        if total > 0:
            self.soc_vectors_readouts = self._pair_readouts(
                feature_irreps, MLP_irreps, total, "1o+1e" if both else "1e"
            )
        self.charge_readouts = torch.nn.ModuleList()
        if long_range_mode == "state":
            self.charge_readouts = self._scalar_readouts(
                feature_irreps, MLP_irreps, gate, int(sum(n_states))
            )
            self._zero_last_layers(self.charge_readouts)
        if self._zero_init_energy_readouts():
            # every state starts from the backbone's energy (plus its shift)
            self._zero_last_layers(self.readouts)

    def _zero_init_energy_readouts(self) -> bool:
        return True

    @staticmethod
    def _zero_last_layers(readouts: torch.nn.ModuleList) -> None:
        with torch.no_grad():
            for readout in readouts:
                last = getattr(readout, "linear_2", None)
                if last is None:
                    last = getattr(readout, "linear", None)
                if last is not None and hasattr(last, "weight"):
                    last.weight.zero_()

    # ----------------------------------------------------------------------------------
    @staticmethod
    def _build_backbone(
        backbone_cls,
        kwargs,
        head_layouts,
        heads,
        atomic_inter_scale,
        atomic_inter_shift,
        keep_last_layer_irreps,
        MLP_irreps,
        gate,
    ):
        """A backbone from the usual MACE arguments (ScaleShiftMACE by default). Per-state
        shifts given as atomic_inter_shift are split into the backbone's shift (their
        mean) and the excited heads' shifts (the differences)."""
        from .models import (  # pylint: disable=import-outside-toplevel
            MACE,
            ScaleShiftMACE,
        )

        if backbone_cls is None:
            backbone_cls = ScaleShiftMACE
        heads = list(heads) if heads else ["Default"]
        if len(head_layouts) not in (1, len(heads)):
            raise ValueError(f"{len(head_layouts)} head layouts for heads {heads}")
        kwargs = dict(kwargs)
        for unused in ("embedding_specs", "use_embedding_readout", "readout_cls"):
            kwargs.pop(unused, None)
        kwargs["MLP_irreps"] = MLP_irreps
        kwargs["gate"] = gate
        hidden = o3.Irreps(str(kwargs["hidden_irreps"]))
        if keep_last_layer_irreps is None:
            keep_last_layer_irreps = hidden.count(o3.Irrep(1, -1)) > 0
        kwargs["keep_last_layer_irreps"] = keep_last_layer_irreps
        shifts = np.asarray(atomic_inter_shift, dtype=float)
        if shifts.ndim == 0:
            per_head = [[float(shifts)] for _ in heads]
        elif shifts.ndim == 1 and len(heads) == 1:
            per_head = [list(shifts)]
        elif shifts.ndim == 1:
            per_head = [[float(x)] for x in shifts]
        else:
            per_head = [list(row) for row in shifts]
        backbone_shift = [float(np.mean(values)) for values in per_head]
        delta_shift = [
            [float(v) - mean for v in values]
            for values, mean in zip(per_head, backbone_shift)
        ]
        scale = np.atleast_1d(np.asarray(atomic_inter_scale, dtype=float))
        if scale.size == 1:
            scale = np.repeat(scale, len(heads))
        if issubclass(backbone_cls, ScaleShiftMACE) or backbone_cls.__name__.endswith(
            "ScaleShiftMACE"
        ):
            backbone = backbone_cls(
                heads=heads,
                atomic_inter_scale=scale.tolist(),
                atomic_inter_shift=backbone_shift,
                **kwargs,
            )
        elif issubclass(backbone_cls, MACE):
            backbone = backbone_cls(heads=heads, **kwargs)
        else:
            backbone = backbone_cls(heads=heads, **kwargs)
        return backbone, heads, delta_shift

    def _energy_readout_total(self) -> int:
        return int(sum(self.head_n_states))

    @staticmethod
    def _scalar_readouts(
        feature_irreps: List[o3.Irreps],
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        num_out: int,
    ) -> torch.nn.ModuleList:
        readouts = torch.nn.ModuleList()
        irreps_out = o3.Irreps(f"{num_out}x0e")
        for i, irreps_in in enumerate(feature_irreps):
            if i == len(feature_irreps) - 1:
                readouts.append(
                    NonLinearReadoutBlock(irreps_in, MLP_irreps, gate, irreps_out)
                )
            else:
                readouts.append(LinearReadoutBlock(irreps_in, irreps_out))
        return readouts

    @staticmethod
    def _vector_readouts(
        feature_irreps: List[o3.Irreps],
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        num_out: int,
    ) -> torch.nn.ModuleList:
        """One readout per feature set that has l = 1 features; the adapter's is
        non-linear. Feature sets without l = 1 contribute nothing."""
        readouts = torch.nn.ModuleList()
        irreps_out = o3.Irreps(f"{num_out}x1o")
        for i, irreps_in in enumerate(feature_irreps):
            if irreps_in.count(o3.Irrep(1, -1)) == 0:
                readouts.append(ZeroVectorReadout(num_out))
            elif i == len(feature_irreps) - 1:
                readouts.append(
                    NonLinearVectorReadoutBlock(irreps_in, MLP_irreps, gate, irreps_out)
                )
            else:
                readouts.append(LinearReadoutBlock(irreps_in, irreps_out))
        return readouts

    @staticmethod
    def _pair_readouts(
        feature_irreps: List[o3.Irreps],
        MLP_irreps: o3.Irreps,
        num_out: int,
        parities: str,
    ) -> torch.nn.ModuleList:
        readouts = torch.nn.ModuleList()
        for irreps_in in feature_irreps:
            if irreps_in.count(o3.Irrep(1, -1)) == 0:
                readouts.append(ZeroVectorReadout(num_out))
            else:
                readouts.append(
                    PairVectorReadoutBlock(irreps_in, MLP_irreps, num_out, parities)
                )
        return readouts

    # ----------------------------------------------------------------------------------
    def mask_tables(self) -> Dict[str, torch.Tensor]:
        """Which padded entries are real for every head ([H, X] masks)."""
        return mask_tables_from_layouts(self.head_layouts)

    def _features(
        self, data: Dict[str, torch.Tensor], node_feats: torch.Tensor
    ) -> List[torch.Tensor]:
        features: List[torch.Tensor] = []
        start = 0
        for dim in self.backbone_layer_dims:
            features.append(node_feats[:, start : start + dim])
            start += dim
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.adapter_sh(vectors)
        edge_feats, cutoff = self.adapter_radial(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        adapter_feats, sc = self.adapter_interaction(
            node_attrs=data["node_attrs"],
            node_feats=self.adapter_in(features[-1]),
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=data["edge_index"],
            cutoff=cutoff,
            first_layer=False,
        )
        adapter_feats = self.adapter_product(
            node_feats=adapter_feats, sc=sc, node_attrs=data["node_attrs"]
        )
        features.append(adapter_feats)
        return features

    def _backbone_charges(
        self, out: Dict[str, Optional[torch.Tensor]], reference: torch.Tensor
    ) -> torch.Tensor:  # [n_nodes]
        charges = out.get("charges")
        if charges is None:
            charges = out.get("latent_charges")
        if charges is None:
            return torch.zeros_like(reference)
        return charges.reshape(reference.shape[0], -1)[:, 0].to(reference.dtype)

    def _couplings(
        self,
        data: Dict[str, torch.Tensor],
        features: List[torch.Tensor],
        node_heads: torch.Tensor,
        graph_heads: torch.Tensor,
        num_graphs: int,
    ) -> Dict[str, Optional[torch.Tensor]]:
        num_nodes = features[0].shape[0]
        dtype, device = features[0].dtype, features[0].device
        nacs: Optional[torch.Tensor] = None
        socs: Optional[torch.Tensor] = None
        dipoles: Optional[torch.Tensor] = None
        soc_vectors: Optional[torch.Tensor] = None
        if len(self.soc_vectors_readouts) > 0:
            node_soc = torch.zeros(num_nodes, 0, dtype=dtype, device=device)
            for i, soc_readout in enumerate(self.soc_vectors_readouts):
                value = soc_readout(features[i])
                node_soc = value if node_soc.shape[1] == 0 else node_soc + value
            graph_soc = scatter_sum(node_soc, data["batch"], dim=0, dim_size=num_graphs)
            soc_vectors = gather_heads(
                graph_soc.view(num_graphs, -1, 3), self.soc_vector_index, graph_heads
            )
        if len(self.nacs_readouts) > 0:
            node_nacs = torch.zeros(num_nodes, 0, dtype=dtype, device=device)
            for i, nac_readout in enumerate(self.nacs_readouts):
                value = nac_readout(features[i])
                node_nacs = value if node_nacs.shape[1] == 0 else node_nacs + value
            for i, nac_axial_readout in enumerate(self.nacs_axial_readouts):
                node_nacs = node_nacs + nac_axial_readout(features[i])
            nacs = gather_heads(
                node_nacs.view(num_nodes, -1, 3), self.nac_index, node_heads
            )
        if len(self.socs_readouts) > 0:
            node_socs = torch.zeros(num_nodes, 0, dtype=dtype, device=device)
            for i, free_soc_readout in enumerate(self.socs_readouts):
                value = free_soc_readout(features[i])
                node_socs = value if node_socs.shape[1] == 0 else node_socs + value
            graph_socs = scatter_sum(
                node_socs, data["batch"], dim=0, dim_size=num_graphs
            )
            socs = gather_heads(graph_socs, self.soc_index, graph_heads)
        if len(self.dipoles_readouts) > 0:
            node_dip = torch.zeros(num_nodes, 0, dtype=dtype, device=device)
            for i, dipole_readout in enumerate(self.dipoles_readouts):
                value = dipole_readout(features[i])
                node_dip = value if node_dip.shape[1] == 0 else node_dip + value
            graph_dip = scatter_sum(node_dip, data["batch"], dim=0, dim_size=num_graphs)
            dipoles = gather_heads(
                graph_dip.view(num_graphs, -1, 3), self.dipole_index, graph_heads
            )
            if len(self.dipoles_axial_readouts) > 0:
                node_tr = torch.zeros(num_nodes, 0, dtype=dtype, device=device)
                for i, dipole_axial_readout in enumerate(self.dipoles_axial_readouts):
                    value = dipole_axial_readout(features[i])
                    node_tr = value if node_tr.shape[1] == 0 else node_tr + value
                graph_tr = scatter_sum(
                    node_tr, data["batch"], dim=0, dim_size=num_graphs
                )
                dipoles = dipoles + gather_heads(
                    graph_tr.view(num_graphs, -1, 3), self.transition_index, graph_heads
                )
        return {
            "nacs": nacs,
            "socs": socs,
            "dipoles": dipoles,
            "soc_vectors": soc_vectors,
        }

    def _state_electrostatics(
        self,
        data: Dict[str, torch.Tensor],
        features: List[torch.Tensor],
        backbone_out: Dict[str, Optional[torch.Tensor]],
        node_heads: torch.Tensor,
        num_graphs: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.long_range_mode != "state":
            return None, None
        if "pbc" in data and bool(torch.any(data["pbc"])):
            raise RuntimeError(
                "long_range_mode='state' supports molecules only (no periodic cells)"
            )
        node_q = torch.zeros(features[0].shape[0], 0, dtype=features[0].dtype)
        node_q = node_q.to(features[0].device)
        for i, charge_readout in enumerate(self.charge_readouts):
            value = charge_readout(features[i])
            node_q = value if node_q.shape[1] == 0 else node_q + value
        dq = gather_heads(node_q, self.energy_index, node_heads)  # [N, S_max]
        num_atoms = (data["ptr"][1:] - data["ptr"][:-1]).to(dq.dtype)
        mean = scatter_sum(dq, data["batch"], dim=0, dim_size=num_graphs)
        dq = dq - (mean / num_atoms.unsqueeze(-1))[data["batch"]]
        q0 = self._backbone_charges(backbone_out, dq[:, 0])
        q_states = q0.unsqueeze(-1) + dq
        energies = coulomb_energies(
            torch.cat([q0.unsqueeze(-1), q_states], dim=1),
            data["positions"],
            data["batch"],
            num_graphs,
            self.charge_smearing,
        )
        return energies[:, 1:] - energies[:, :1], q_states

    def _heads(
        self, data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if "head" in data:
            graph_heads = data["head"].to(torch.long).view(-1)
        else:
            graph_heads = torch.zeros(
                data["ptr"].numel() - 1, dtype=torch.long, device=data["ptr"].device
            )
        return graph_heads, graph_heads[data["batch"]]

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,  # pylint: disable=unused-argument
        compute_stress: bool = False,  # pylint: disable=unused-argument
        compute_displacement: bool = False,  # pylint: disable=unused-argument
        compute_hessian: bool = False,  # pylint: disable=unused-argument
        compute_edge_forces: bool = False,  # pylint: disable=unused-argument
        compute_atomic_stresses: bool = False,  # pylint: disable=unused-argument
        lammps_mliap: bool = False,  # pylint: disable=unused-argument
    ) -> Dict[str, Optional[torch.Tensor]]:
        backbone_out = self.backbone(data, training)
        ground = backbone_out["energy"]
        node_feats = backbone_out["node_feats"]
        assert ground is not None
        assert node_feats is not None
        num_graphs = ground.shape[0]
        graph_heads, node_heads = self._heads(data)
        features = self._features(data, node_feats)

        node_delta = torch.zeros(node_feats.shape[0], 0, dtype=ground.dtype)
        node_delta = node_delta.to(ground.device)
        for i, readout in enumerate(self.readouts):
            value = readout(features[i])
            node_delta = value if node_delta.shape[1] == 0 else node_delta + value
        node_delta = gather_heads(node_delta, self.energy_index, node_heads)
        node_delta = node_delta * self.delta_scale[node_heads].unsqueeze(
            -1
        ) + self.delta_shift[node_heads].to(node_delta.dtype)
        delta = scatter_sum(node_delta, data["batch"], dim=0, dim_size=num_graphs)
        energy = ground.unsqueeze(-1) + delta
        electrostatic, state_charges = self._state_electrostatics(
            data, features, backbone_out, node_heads, num_graphs
        )
        if electrostatic is not None:
            energy = energy + electrostatic

        couplings = self._couplings(data, features, node_heads, graph_heads, num_graphs)
        forces: Optional[torch.Tensor] = None
        if compute_force:
            forces = compute_multistate_forces(energy, data["positions"], training)
        return {
            "energy": energy,
            "ground_energy": ground,
            "node_energy": None,
            "contributions": None,
            "forces": forces,
            "nacs": couplings["nacs"],
            "socs": couplings["socs"],
            "dipoles": couplings["dipoles"],
            "soc_vectors": couplings["soc_vectors"],
            "state_charges": state_charges,
            "state_electrostatic_energy": electrostatic,
            "edge_forces": None,
            "virials": None,
            "stress": None,
            "atomic_virials": None,
            "atomic_stresses": None,
            "displacement": backbone_out.get("displacement"),
            "hessian": None,
            "node_feats": torch.cat(features, dim=-1),
        }


@compile_mode("script")
class ZeroVectorReadout(torch.nn.Module):
    """Placeholder readout for feature sets without l = 1 features."""

    def __init__(self, num_out: int):
        super().__init__()
        self.num_out = num_out

    def forward(
        self,
        x: torch.Tensor,
        heads: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        return torch.zeros(x.shape[0], self.num_out * 3, dtype=x.dtype, device=x.device)


@compile_mode("script")
class AutoencoderExcitedMACE(ExcitedMACE):
    """X-MACE with the deep-set autoencoder, on any backbone and with several heads.

    For each head the readouts predict a permutation-invariant latent vector (K
    values). A decoder maps it to a symmetric tridiagonal matrix whose eigenvalues
    are the state energies relative to the backbone, so states stay ordered and their
    crossings smooth. With several multiplicities the matrix has one block per
    multiplicity (states ordered within each multiplicity, as SHARC's MCH states)
    and the shift is shared within a multiplicity. During training an encoder maps the
    reference energies to the latent space; the decoder must reconstruct them
    (decoded_energy) and the network must reproduce the encoded vector
    (encoded_energy). Ground-state heads (one state) work as in ExcitedMACE.
    """

    def __init__(
        self, *args: Any, num_permutational_invariant: int = 16, **kwargs: Any
    ):
        self.num_permutational_invariant = num_permutational_invariant
        super().__init__(*args, **kwargs)
        latent = num_permutational_invariant
        self.register_buffer(
            "latent_index",
            _index_table([latent] * len(self.head_layouts)),
        )
        self.perm_encoders = torch.nn.ModuleList()
        self.perm_decoders = torch.nn.ModuleList()
        averages: List[torch.Tensor] = []
        for layout in self.head_layouts:
            blocks = [n for n in layout["states"] if n > 0]
            groups = [g for g, n in enumerate(blocks) for _ in range(n)]
            self.perm_encoders.append(
                PermutationInvariantEncoder(
                    latent_dim=latent, groups=groups if len(blocks) > 1 else None
                )
            )
            self.perm_decoders.append(
                PermutationInvariantDecoder(
                    latent_dim=latent, n_energies=layout["n_states"], block_sizes=blocks
                )
            )
            average = torch.zeros(self.n_energies, self.n_energies)
            start = 0
            for n in blocks:
                average[start : start + n, start : start + n] = 1.0 / n
                start += n
            averages.append(average)
        # one energy shift per multiplicity (mean of its states' shifts), so the shift
        # keeps the decoded states in order
        self.register_buffer("shift_average", torch.stack(averages))

    def _energy_readout_total(self) -> int:
        return self.num_permutational_invariant * len(self.head_n_states)

    def _zero_init_energy_readouts(self) -> bool:
        return False

    def _decode(self, latent: torch.Tensor, graph_heads: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(
            latent.shape[0], self.n_energies, dtype=latent.dtype, device=latent.device
        )
        for h, decoder in enumerate(self.perm_decoders):
            decoded = decoder(latent)
            selected = (graph_heads == h).unsqueeze(-1)
            padded = torch.zeros_like(out)
            padded[:, : decoded.shape[1]] = decoded
            out = torch.where(selected, padded, out)
        return out

    def _encode(self, values: torch.Tensor, graph_heads: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(
            values.shape[0],
            self.num_permutational_invariant,
            dtype=values.dtype,
            device=values.device,
        )
        for h, encoder in enumerate(self.perm_encoders):
            encoded = encoder(values[:, : self.head_n_states[h]])
            out = torch.where((graph_heads == h).unsqueeze(-1), encoded, out)
        return out

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,  # pylint: disable=unused-argument
        compute_stress: bool = False,  # pylint: disable=unused-argument
        compute_displacement: bool = False,  # pylint: disable=unused-argument
        compute_hessian: bool = False,  # pylint: disable=unused-argument
        compute_edge_forces: bool = False,  # pylint: disable=unused-argument
        compute_atomic_stresses: bool = False,  # pylint: disable=unused-argument
        lammps_mliap: bool = False,  # pylint: disable=unused-argument
    ) -> Dict[str, Optional[torch.Tensor]]:
        backbone_out = self.backbone(data, training)
        ground = backbone_out["energy"]
        node_feats = backbone_out["node_feats"]
        assert ground is not None
        assert node_feats is not None
        num_graphs = ground.shape[0]
        graph_heads, node_heads = self._heads(data)
        features = self._features(data, node_feats)

        node_latent = torch.zeros(node_feats.shape[0], 0, dtype=ground.dtype)
        node_latent = node_latent.to(ground.device)
        for i, readout in enumerate(self.readouts):
            value = readout(features[i])
            node_latent = value if node_latent.shape[1] == 0 else node_latent + value
        node_latent = gather_heads(node_latent, self.latent_index, node_heads)
        invariant_vals = scatter_sum(
            node_latent, data["batch"], dim=0, dim_size=num_graphs
        )

        num_atoms = (data["ptr"][1:] - data["ptr"][:-1]).to(ground.dtype)
        shift = torch.bmm(
            self.shift_average[graph_heads].to(ground.dtype),
            self.delta_shift[graph_heads].to(ground.dtype).unsqueeze(-1),
        ).squeeze(-1)
        baseline = ground.unsqueeze(-1) + num_atoms.unsqueeze(-1) * shift
        scale = self.delta_scale[graph_heads].unsqueeze(-1)
        energy = baseline + scale * self._decode(invariant_vals, graph_heads)
        electrostatic, state_charges = self._state_electrostatics(
            data, features, backbone_out, node_heads, num_graphs
        )
        if electrostatic is not None:
            energy = energy + electrostatic

        encoded_energy: Optional[torch.Tensor] = None
        decoded_energy: Optional[torch.Tensor] = None
        if "energy" in data:
            reference = data["energy"]
            if reference.dim() == 2 and reference.shape[1] == self.n_energies:
                target = (reference - baseline) / scale
                if electrostatic is not None:
                    target = target - electrostatic / scale
                encoded_energy = self._encode(target, graph_heads)
                decoded_energy = baseline + scale * self._decode(
                    encoded_energy, graph_heads
                )
                if electrostatic is not None:
                    decoded_energy = decoded_energy + electrostatic

        couplings = self._couplings(data, features, node_heads, graph_heads, num_graphs)
        forces: Optional[torch.Tensor] = None
        if compute_force:
            forces = compute_multistate_forces(energy, data["positions"], training)
        return {
            "energy": energy,
            "ground_energy": ground,
            "node_energy": None,
            "contributions": None,
            "invariant_vals": invariant_vals,
            "encoded_energy": encoded_energy,
            "decoded_energy": decoded_energy,
            "forces": forces,
            "nacs": couplings["nacs"],
            "socs": couplings["socs"],
            "dipoles": couplings["dipoles"],
            "soc_vectors": couplings["soc_vectors"],
            "state_charges": state_charges,
            "state_electrostatic_energy": electrostatic,
            "edge_forces": None,
            "virials": None,
            "stress": None,
            "atomic_virials": None,
            "atomic_stresses": None,
            "displacement": backbone_out.get("displacement"),
            "hessian": None,
            "node_feats": torch.cat(features, dim=-1),
        }


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------


def compute_excited_scaling(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
    scaling: str,
    n_states: int,
):
    """(scale, shift[S]) of the atomic interaction energies of one head.

    shift_s  mean over labelled configurations of (E_s - sum E0) / n_atoms
    scale    rms of the forces of all states ("rms_forces_scaling"), std of the
             atomic interaction energies ("std_scaling") or 1 ("no_scaling")
    """
    if scaling == "no_scaling":
        return 1.0, [0.0] * n_states
    from .blocks import AtomicEnergiesBlock  # pylint: disable=import-outside-toplevel

    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)
    inter_energies, forces = [], []
    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        if node_e0.dim() > 1:
            node_e0 = node_e0[:, 0]
        graph_e0 = scatter_sum(node_e0, batch.batch, dim=0, dim_size=batch.num_graphs)
        num_atoms = (batch.ptr[1:] - batch.ptr[:-1]).to(graph_e0.dtype)
        energy = batch.energy.view(batch.num_graphs, -1)[:, :n_states]
        labelled = (batch.weight * batch.energy_weight) > 0
        inter = (energy - graph_e0.unsqueeze(-1)) / num_atoms.unsqueeze(-1)
        inter_energies.append(inter[labelled])
        forces_labelled = torch.repeat_interleave(
            (batch.weight * batch.forces_weight) > 0, batch.ptr[1:] - batch.ptr[:-1]
        )
        forces.append(
            batch.forces.view(batch.forces.shape[0], -1, 3)[forces_labelled][
                :, :n_states
            ]
        )
    inter_energies = torch.cat(inter_energies, dim=0)
    if inter_energies.shape[0] == 0:
        shift = [0.0] * n_states
        std = 1.0
    else:
        shift = inter_energies.mean(dim=0).tolist()
        std = float(torch.std(inter_energies)) if inter_energies.numel() > 1 else 1.0
    forces = torch.cat(forces, dim=0)
    rms_forces = (
        float(torch.sqrt(torch.mean(torch.square(forces)))) if forces.numel() else 1.0
    )
    if scaling == "std_scaling":
        scale = std
    elif scaling == "rms_forces_scaling":
        scale = rms_forces
    else:
        raise ValueError(f"Unknown scaling {scaling!r} for excited-state models")
    if not np.isfinite(scale) or scale < 1e-8:
        logging.warning(f"Scale {scale} is not usable, using 1.0 instead")
        scale = 1.0
    return scale, shift


@torch.no_grad()
def fit_excited_shifts(
    model: ExcitedMACE,
    data_loader: torch.utils.data.DataLoader,
    device: Union[str, torch.device] = "cpu",
) -> None:
    """Set every head's per-state shifts so that, on the training data, the mean
    energy per atom of every state is right relative to the backbone's prediction:
    shift_{h,s} = mean over labelled frames of head h of (E_s - E_backbone) / n_atoms.
    Used when the backbone comes from a foundation model."""
    sums = torch.zeros_like(model.delta_shift, dtype=torch.float64)
    counts = torch.zeros_like(model.delta_shift, dtype=torch.float64)
    masks = model.mask_tables()["energy"].to(torch.float64)
    for batch in data_loader:
        batch = batch.to(device)
        data = batch.to_dict()
        out = model.backbone(data, False)
        ground = out["energy"].detach()
        heads = data["head"].to(torch.long).view(-1)
        num_atoms = (batch.ptr[1:] - batch.ptr[:-1]).to(ground.dtype)
        energy = batch.energy.view(batch.num_graphs, -1)
        labelled = ((batch.weight * batch.energy_weight) > 0).to(torch.float64)
        values = ((energy - ground.unsqueeze(-1)) / num_atoms.unsqueeze(-1)).to(
            torch.float64
        )
        weight = labelled.unsqueeze(-1) * masks[heads]
        sums.index_add_(0, heads, values * weight)
        counts.index_add_(0, heads, weight)
    shift = torch.where(
        counts > 0, sums / counts.clamp(min=1.0), torch.zeros_like(sums)
    )
    model.delta_shift.copy_(shift.to(model.delta_shift.dtype))
