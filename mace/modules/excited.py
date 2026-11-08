###########################################################################################
# X-MACE: MACE models for electronically excited states
#
#   ExcitedMACE             S energies (+ forces), NACs, SOCs and dipoles from one network
#   AutoencoderExcitedMACE  energies from the eigenvalues of a learnt matrix (deep-set
#                           autoencoder), which keeps them smooth near conical intersections
#
# Both models add a ZBL core repulsion (--pair_repulsion) to every state and use a
# per-state scale/shift of the atomic interaction energies.
#
# Ported from X-MACE (https://github.com/rhyan10/X-MACE, branch X-MACE_socs).
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

import numpy as np
import torch
from e3nn import nn, o3
from e3nn.util.jit import compile_mode

from mace.tools.scatter import scatter_sum

from .blocks import InteractionBlock, LinearReadoutBlock, NonLinearReadoutBlock
from .models import MACE
from .spin_states import parse_states, same_multiplicity_pairs, spin_orbit_pairs
from .utils import prepare_graph

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
class AxialVectorReadoutBlock(torch.nn.Module):
    """Readout to axial vectors ("P x 1e"), e.g. spin-orbit coupling vectors.

    Polar l=1 features (1o) give axial vectors through their cross products
    (1o x 1o -> 1e), so the block works with the usual hidden irreps. The features
    are first projected to K scalars and K vectors (K = number of MLP scalars)."""

    def __init__(self, irreps_in: o3.Irreps, MLP_irreps: o3.Irreps, num_out: int):
        super().__init__()
        irreps_in = o3.Irreps(irreps_in)
        num_hidden = max(1, o3.Irreps(MLP_irreps).count(o3.Irrep(0, 1)))
        hidden = f"{num_hidden}x0e + {num_hidden}x1o"
        if irreps_in.count(o3.Irrep(1, 1)) > 0:
            hidden += f" + {num_hidden}x1e"
        self.hidden_irreps = o3.Irreps(hidden)
        self.irreps_out = o3.Irreps(f"{num_out}x1e")
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
        return self.tensor_product(hidden, hidden)


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
    """Deep set over the electronic states: energies [B, S] -> latent [B, K]."""

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 16):
        super().__init__()
        act = torch.nn.ELU
        self.elementwise_nn = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
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
        per_state = self.elementwise_nn(energies.unsqueeze(-1))  # [B, S, H]
        return self.post_aggregation_nn(per_state.sum(dim=1))  # [B, K]


@compile_mode("script")
class PermutationInvariantDecoder(torch.nn.Module):
    """latent [B, K] -> symmetric tridiagonal S x S matrix -> sorted eigenvalues [B, S]."""

    def __init__(
        self, latent_dim: int = 16, n_energies: int = 3, hidden_dim: int = 128
    ):
        super().__init__()
        self.n_energies = n_energies
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
            torch.nn.Linear(hidden_dim, 2 * n_energies - 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        values = self.decoder_nn(latent)
        diagonal = values[:, : self.n_energies]
        off_diagonal = values[:, self.n_energies :]
        matrix = (
            torch.diag_embed(diagonal)
            + torch.diag_embed(off_diagonal, offset=1)
            + torch.diag_embed(off_diagonal, offset=-1)
        )
        return torch.linalg.eigvalsh(matrix)  # ascending order


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------


@compile_mode("script")
class ExcitedMACE(MACE):
    """MACE for several electronic states.

    Outputs (B graphs, N atoms, S states, P NAC vectors, L SOCs, D dipoles):
      energy  [B, S]       forces  [N, S, 3]
      nacs    [N, P, 3]    smooth NACs (NAC * dE), per-atom equivariant vectors
      socs    [B, L]       sum of per-atom invariant scalars
      dipoles [B, D, 3]    sum of per-atom equivariant vectors
      soc_vectors [B, Q, 3]  spin-orbit coupling vectors (soc_mode="spin_orbit"): sums
                           of per-atom axial vectors, one per spin-orbit-coupled pair of
                           spatial states; mace.modules.spin_states turns them into the
                           M_S-resolved SOC matrix (Wigner-Eckart)

    states gives the states per multiplicity as in SHARC ([n_singlets, n_doublets,
    n_triplets, ...]); NACs and transition dipoles then couple states of the same
    multiplicity only.

    The ZBL core repulsion (pair_repulsion=True) is the same for every state. The
    atomic interaction energies are scaled by atomic_inter_scale and shifted by
    atomic_inter_shift (one value per state). When NACs or dipoles are predicted,
    the last layer keeps its l=1 features so it contributes to them as well.
    """

    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        n_energies: int = 1,
        compute_nacs: bool = False,
        nac_num: Optional[int] = None,
        compute_socs: bool = False,
        soc_num: Optional[int] = None,
        compute_dipoles: bool = False,
        dipole_num: Optional[int] = None,
        atomic_inter_scale: float = 1.0,
        atomic_inter_shift: Union[float, Sequence[float]] = 0.0,
        keep_last_layer_irreps: Optional[bool] = None,
        heads: Optional[List[str]] = None,
        states: Optional[List[int]] = None,
        soc_mode: str = "free",
        **kwargs: Any,
    ):
        if heads is not None and len(heads) > 1:
            raise NotImplementedError("ExcitedMACE supports a single head only")
        if n_energies < 1:
            raise ValueError(f"n_energies must be at least 1, got {n_energies}")
        if states is not None:
            states = parse_states(states)
            if sum(states) != n_energies:
                raise ValueError(
                    f"states {states} describe {sum(states)} states, but n_energies={n_energies}"
                )
        layout = list(states) if states is not None else [n_energies]
        # NACs and transition dipoles couple states of the same multiplicity only
        n_pairs = len(same_multiplicity_pairs(layout))
        if soc_mode not in ("free", "spin_orbit"):
            raise ValueError(
                f"soc_mode must be 'free' or 'spin_orbit', got {soc_mode!r}"
            )
        soc_vector_num = 0
        if compute_socs and soc_mode == "spin_orbit":
            if states is None:
                raise ValueError(
                    "soc_mode='spin_orbit' needs the states per multiplicity"
                )
            soc_vector_num = len(spin_orbit_pairs(layout))
            if soc_vector_num < 1:
                raise ValueError(f"No spin-orbit-coupled pairs of states in {layout}")
            compute_socs = False
        nac_num = (nac_num if nac_num is not None else n_pairs) if compute_nacs else 0
        soc_num = (
            (soc_num if soc_num is not None else num_state_pairs(n_energies))
            if compute_socs
            else 0
        )
        dipole_num = (
            (dipole_num if dipole_num is not None else n_energies + n_pairs)
            if compute_dipoles
            else 0
        )
        if compute_nacs and nac_num < 1:
            raise ValueError("compute_nacs needs n_energies >= 2 or nac_num >= 1")
        if compute_socs and soc_num < 1:
            raise ValueError("compute_socs needs n_energies >= 2 or soc_num >= 1")
        if compute_dipoles and dipole_num < 1:
            raise ValueError("compute_dipoles needs dipole_num >= 1")
        needs_vectors = nac_num > 0 or dipole_num > 0 or soc_vector_num > 0
        if keep_last_layer_irreps is None:
            keep_last_layer_irreps = needs_vectors
        if needs_vectors and o3.Irreps(hidden_irreps).count(o3.Irrep(1, -1)) == 0:
            raise ValueError(
                "NACs, dipoles and SOC vectors need hidden irreps with l=1 features "
                "(e.g. --hidden_irreps='64x0e + 64x1o' or --max_L=1)"
            )
        if needs_vectors and not keep_last_layer_irreps and num_interactions == 1:
            raise ValueError(
                "Vector outputs with one layer need keep_last_layer_irreps"
            )

        # Arguments of other model types that ExcitedMACE does not use.
        for unused in ("embedding_specs", "use_embedding_readout", "readout_cls"):
            kwargs.pop(unused, None)
        super().__init__(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            max_ell=max_ell,
            interaction_cls=interaction_cls,
            interaction_cls_first=interaction_cls_first,
            num_interactions=num_interactions,
            num_elements=num_elements,
            hidden_irreps=hidden_irreps,
            MLP_irreps=MLP_irreps,
            atomic_energies=atomic_energies,
            avg_num_neighbors=avg_num_neighbors,
            atomic_numbers=atomic_numbers,
            correlation=correlation,
            gate=gate,
            heads=None,
            keep_last_layer_irreps=keep_last_layer_irreps,
            use_last_readout_only=False,
            **kwargs,
        )
        self.n_energies = n_energies
        self.compute_nacs = nac_num > 0
        self.compute_socs = soc_num > 0
        self.compute_dipoles = dipole_num > 0
        self.nac_num = nac_num
        self.soc_num = soc_num
        self.dipole_num = dipole_num
        self.compute_soc_vectors = soc_vector_num > 0
        self.soc_vector_num = soc_vector_num
        self.soc_mode = "spin_orbit" if soc_vector_num > 0 else "free"
        self.multiplicity_states: List[int] = layout
        self.keep_last_layer_irreps = keep_last_layer_irreps
        self.scale_shift = ExcitedScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift, n_states=n_energies
        )

        layer_irreps = [
            o3.Irreps(str(product.linear.irreps_out)) for product in self.products
        ]
        self.readouts = self._scalar_readouts(
            layer_irreps, MLP_irreps, gate, self._energy_readout_dim()
        )
        self.socs_readouts = torch.nn.ModuleList()
        if self.compute_socs:
            self.socs_readouts = self._scalar_readouts(
                layer_irreps, MLP_irreps, gate, soc_num
            )
        self.nacs_readouts = torch.nn.ModuleList()
        if self.compute_nacs:
            self.nacs_readouts = self._vector_readouts(
                layer_irreps, MLP_irreps, gate, nac_num
            )
        self.dipoles_readouts = torch.nn.ModuleList()
        if self.compute_dipoles:
            self.dipoles_readouts = self._vector_readouts(
                layer_irreps, MLP_irreps, gate, dipole_num
            )
        self.soc_vectors_readouts = torch.nn.ModuleList()
        if self.compute_soc_vectors:
            for irreps_in in layer_irreps:
                if irreps_in.count(o3.Irrep(1, -1)) > 0:
                    self.soc_vectors_readouts.append(
                        AxialVectorReadoutBlock(irreps_in, MLP_irreps, soc_vector_num)
                    )

    def _energy_readout_dim(self) -> int:
        return self.n_energies

    @staticmethod
    def _scalar_readouts(
        layer_irreps: List[o3.Irreps],
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        num_out: int,
    ) -> torch.nn.ModuleList:
        readouts = torch.nn.ModuleList()
        irreps_out = o3.Irreps(f"{num_out}x0e")
        for i, irreps_in in enumerate(layer_irreps):
            if i == len(layer_irreps) - 1:
                readouts.append(
                    NonLinearReadoutBlock(irreps_in, MLP_irreps, gate, irreps_out)
                )
            else:
                readouts.append(LinearReadoutBlock(irreps_in, irreps_out))
        return readouts

    @staticmethod
    def _vector_readouts(
        layer_irreps: List[o3.Irreps],
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        num_out: int,
    ) -> torch.nn.ModuleList:
        readouts = torch.nn.ModuleList()
        irreps_out = o3.Irreps(f"{num_out}x1o")
        for i, irreps_in in enumerate(layer_irreps):
            if irreps_in.count(o3.Irrep(1, -1)) == 0:
                continue  # scalar-only last layer: no vector contribution
            if i == len(layer_irreps) - 1:
                readouts.append(
                    NonLinearVectorReadoutBlock(irreps_in, MLP_irreps, gate, irreps_out)
                )
            else:
                readouts.append(LinearReadoutBlock(irreps_in, irreps_out))
        return readouts

    def _node_features(
        self,
        data: Dict[str, torch.Tensor],
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        cutoff: Optional[torch.Tensor],
    ) -> List[torch.Tensor]:
        node_feats = self.node_embedding(data["node_attrs"])
        node_feats_list: List[torch.Tensor] = []
        for i, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
                first_layer=(i == 0),
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
            )
            node_feats_list.append(node_feats)
        return node_feats_list

    def _couplings(
        self,
        data: Dict[str, torch.Tensor],
        node_feats_list: List[torch.Tensor],
        num_graphs: int,
    ) -> Dict[str, Optional[torch.Tensor]]:
        num_nodes = data["node_attrs"].shape[0]
        nacs: Optional[torch.Tensor] = None
        socs: Optional[torch.Tensor] = None
        dipoles: Optional[torch.Tensor] = None
        soc_vectors: Optional[torch.Tensor] = None
        if self.compute_soc_vectors:
            node_soc = torch.zeros(
                num_nodes,
                self.soc_vector_num * 3,
                dtype=node_feats_list[0].dtype,
                device=node_feats_list[0].device,
            )
            for i, readout in enumerate(self.soc_vectors_readouts):
                node_soc = node_soc + readout(node_feats_list[i])
            soc_vectors = scatter_sum(
                node_soc, data["batch"], dim=0, dim_size=num_graphs
            ).view(num_graphs, self.soc_vector_num, 3)
        if self.compute_nacs:
            node_nacs = torch.zeros(
                num_nodes,
                self.nac_num * 3,
                dtype=node_feats_list[0].dtype,
                device=node_feats_list[0].device,
            )
            for i, readout in enumerate(self.nacs_readouts):
                node_nacs = node_nacs + readout(node_feats_list[i])
            nacs = node_nacs.view(num_nodes, self.nac_num, 3)
        if self.compute_socs:
            node_socs = torch.zeros(
                num_nodes,
                self.soc_num,
                dtype=node_feats_list[0].dtype,
                device=node_feats_list[0].device,
            )
            for i, readout in enumerate(self.socs_readouts):
                node_socs = node_socs + readout(node_feats_list[i])
            socs = scatter_sum(node_socs, data["batch"], dim=0, dim_size=num_graphs)
        if self.compute_dipoles:
            node_dipoles = torch.zeros(
                num_nodes,
                self.dipole_num * 3,
                dtype=node_feats_list[0].dtype,
                device=node_feats_list[0].device,
            )
            for i, readout in enumerate(self.dipoles_readouts):
                node_dipoles = node_dipoles + readout(node_feats_list[i])
            dipoles = scatter_sum(
                node_dipoles, data["batch"], dim=0, dim_size=num_graphs
            ).view(num_graphs, self.dipole_num, 3)
        return {
            "nacs": nacs,
            "socs": socs,
            "dipoles": dipoles,
            "soc_vectors": soc_vectors,
        }

    def _graph_setup(self, data: Dict[str, torch.Tensor]):
        ctx = prepare_graph(data)
        num_atoms_arange = ctx.num_atoms_arange.to(torch.int64)
        node_heads = ctx.node_heads.to(torch.int64)
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]  # [n_nodes]
        edge_attrs = self.spherical_harmonics(ctx.vectors)
        edge_feats, cutoff = self.radial_embedding(
            ctx.lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                ctx.lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
        else:
            pair_node_energy = torch.zeros_like(node_e0)
        return ctx, node_e0, pair_node_energy, edge_attrs, edge_feats, cutoff

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
        ctx, node_e0, pair_node_energy, edge_attrs, edge_feats, cutoff = (
            self._graph_setup(data)
        )
        num_graphs = ctx.num_graphs
        batch = data["batch"]
        node_feats_list = self._node_features(data, edge_attrs, edge_feats, cutoff)

        node_inter_list: List[torch.Tensor] = []
        for i, readout in enumerate(self.readouts):
            node_inter_list.append(readout(node_feats_list[i]))  # [n_nodes, S]
        node_inter = self.scale_shift(torch.sum(torch.stack(node_inter_list), dim=0))
        node_base = (node_e0 + pair_node_energy).unsqueeze(-1)  # [n_nodes, 1]
        node_energy = node_base + node_inter  # [n_nodes, S]
        energy = scatter_sum(node_energy, batch, dim=0, dim_size=num_graphs)

        e0 = scatter_sum(node_e0, batch, dim=0, dim_size=num_graphs)
        pair_energy = scatter_sum(pair_node_energy, batch, dim=0, dim_size=num_graphs)
        contributions = torch.stack(
            [
                e0.unsqueeze(-1).expand(-1, self.n_energies),
                pair_energy.unsqueeze(-1).expand(-1, self.n_energies),
            ]
            + [
                scatter_sum(
                    self.scale_shift.scale * node_es, batch, dim=0, dim_size=num_graphs
                )
                for node_es in node_inter_list
            ],
            dim=-1,
        )  # [n_graphs, S, n_layers + 2]

        couplings = self._couplings(data, node_feats_list, num_graphs)
        forces: Optional[torch.Tensor] = None
        if compute_force:
            forces = compute_multistate_forces(energy, ctx.positions, training)

        return {
            "energy": energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "pair_energy": pair_energy,
            "forces": forces,
            "nacs": couplings["nacs"],
            "socs": couplings["socs"],
            "dipoles": couplings["dipoles"],
            "soc_vectors": couplings["soc_vectors"],
            "edge_forces": None,
            "virials": None,
            "stress": None,
            "atomic_virials": None,
            "atomic_stresses": None,
            "displacement": ctx.displacement,
            "hessian": None,
            "node_feats": torch.cat(node_feats_list, dim=-1),
        }


@compile_mode("script")
class AutoencoderExcitedMACE(ExcitedMACE):
    """X-MACE with the deep-set autoencoder.

    The network predicts a permutation-invariant latent vector (readouts, K values
    per configuration). A decoder maps it to a symmetric tridiagonal S x S matrix
    whose eigenvalues are the interaction energies, so the states stay ordered and
    their crossings smooth. During training an encoder maps the reference energies
    to the latent space; the decoder must reconstruct them (decoded_energy) and the
    network must reproduce the encoded latent vector (encoded_energy).
    """

    def __init__(
        self, *args: Any, num_permutational_invariant: int = 16, **kwargs: Any
    ):
        self.num_permutational_invariant = num_permutational_invariant
        super().__init__(*args, **kwargs)
        self.perm_encoder = PermutationInvariantEncoder(
            latent_dim=num_permutational_invariant
        )
        self.perm_decoder = PermutationInvariantDecoder(
            latent_dim=num_permutational_invariant, n_energies=self.n_energies
        )

    def _energy_readout_dim(self) -> int:
        return self.num_permutational_invariant

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
        ctx, node_e0, pair_node_energy, edge_attrs, edge_feats, cutoff = (
            self._graph_setup(data)
        )
        num_graphs = ctx.num_graphs
        batch = data["batch"]
        node_feats_list = self._node_features(data, edge_attrs, edge_feats, cutoff)

        node_latent = torch.zeros(
            data["node_attrs"].shape[0],
            self.num_permutational_invariant,
            dtype=node_feats_list[0].dtype,
            device=node_feats_list[0].device,
        )
        for i, readout in enumerate(self.readouts):
            node_latent = node_latent + readout(node_feats_list[i])
        invariant_vals = scatter_sum(node_latent, batch, dim=0, dim_size=num_graphs)

        num_atoms = (data["ptr"][1:] - data["ptr"][:-1]).to(invariant_vals.dtype)
        e0 = scatter_sum(node_e0, batch, dim=0, dim_size=num_graphs)
        pair_energy = scatter_sum(pair_node_energy, batch, dim=0, dim_size=num_graphs)
        baseline = (e0 + pair_energy).unsqueeze(-1) + num_atoms.unsqueeze(
            -1
        ) * self.scale_shift.shift.unsqueeze(
            0
        )  # [n_graphs, S]
        scale = self.scale_shift.scale
        energy = baseline + scale * self.perm_decoder(invariant_vals)

        encoded_energy: Optional[torch.Tensor] = None
        decoded_energy: Optional[torch.Tensor] = None
        if "energy" in data:
            ref_energy = data["energy"]
            if ref_energy.dim() == 2 and ref_energy.shape[1] == self.n_energies:
                encoded = self.perm_encoder((ref_energy - baseline) / scale)
                encoded_energy = encoded
                decoded_energy = baseline + scale * self.perm_decoder(encoded)

        couplings = self._couplings(data, node_feats_list, num_graphs)
        forces: Optional[torch.Tensor] = None
        if compute_force:
            forces = compute_multistate_forces(energy, ctx.positions, training)

        return {
            "energy": energy,
            "node_energy": None,
            "contributions": None,
            "pair_energy": pair_energy,
            "invariant_vals": invariant_vals,
            "encoded_energy": encoded_energy,
            "decoded_energy": decoded_energy,
            "forces": forces,
            "nacs": couplings["nacs"],
            "socs": couplings["socs"],
            "dipoles": couplings["dipoles"],
            "soc_vectors": couplings["soc_vectors"],
            "edge_forces": None,
            "virials": None,
            "stress": None,
            "atomic_virials": None,
            "atomic_stresses": None,
            "displacement": ctx.displacement,
            "hessian": None,
            "node_feats": torch.cat(node_feats_list, dim=-1),
        }


# --------------------------------------------------------------------------------------
# Statistics and transfer learning
# --------------------------------------------------------------------------------------


def compute_excited_scaling(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
    scaling: str,
    n_states: int,
):
    """(scale, shift[S]) of the atomic interaction energies for ExcitedMACE.

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
        energy = batch.energy.view(batch.num_graphs, -1)
        labelled = (batch.weight * batch.energy_weight) > 0
        inter = (energy - graph_e0.unsqueeze(-1)) / num_atoms.unsqueeze(-1)
        inter_energies.append(inter[labelled])
        forces_labelled = torch.repeat_interleave(
            (batch.weight * batch.forces_weight) > 0, batch.ptr[1:] - batch.ptr[:-1]
        )
        forces.append(batch.forces.view(batch.forces.shape[0], -1, 3)[forces_labelled])
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


def load_foundations_excited(
    model: ExcitedMACE,
    model_foundation: torch.nn.Module,
    table,
    load_readout: bool = False,
    max_L: int = 1,
    default_dtype: Optional[torch.dtype] = None,
) -> ExcitedMACE:
    """Initialise an ExcitedMACE from a ground-state MACE foundation model.

    Embedding, radial basis, interactions and products are taken from the
    foundation. Blocks whose shape differs because ExcitedMACE keeps l=1
    features in the last layer get the foundation weights for their scalar part
    (which e3nn stores first). With load_readout, every state starts from the
    foundation energy readout (and its scale/shift)."""
    # pylint: disable=import-outside-toplevel
    from mace.tools.finetuning_utils import load_foundations_elements_default

    original = {
        name: param.detach().clone() for name, param in model.named_parameters()
    }
    load_foundations_elements_default(
        model,
        model_foundation,
        table,
        load_readout=False,
        max_L=max_L,
        default_dtype=default_dtype,
    )
    for name, param in list(model.named_parameters()):
        if name not in original or param.shape == original[name].shape:
            continue
        target = original[name].clone()
        source = param.detach().reshape(-1)
        if source.numel() > target.numel():
            raise RuntimeError(
                f"Foundation parameter {name} is larger than the ExcitedMACE one "
                f"({source.numel()} > {target.numel()}); the architectures do not match"
            )
        target.view(-1)[: source.numel()] = source.to(target.dtype)
        module_name, _, attr = name.rpartition(".")
        setattr(model.get_submodule(module_name), attr, torch.nn.Parameter(target))
        logging.info(
            f"Foundation weights copied into the scalar part of {name} "
            f"({source.numel()} of {target.numel()} values)"
        )

    if hasattr(model_foundation, "pair_repulsion_fn") and hasattr(
        model, "pair_repulsion_fn"
    ):
        model.pair_repulsion_fn.load_state_dict(
            model_foundation.pair_repulsion_fn.state_dict()
        )

    if load_readout and not isinstance(model, AutoencoderExcitedMACE):
        _copy_foundation_readouts(model, model_foundation)
    return model


def _copy_foundation_readouts(model: ExcitedMACE, model_foundation: torch.nn.Module):
    """Every state starts from the foundation's energy readout."""
    if len(model.readouts) != len(model_foundation.readouts):
        logging.warning("Foundation readouts not loaded: different number of layers")
        return
    n_states = model.n_energies
    with torch.no_grad():
        for readout, source in zip(model.readouts, model_foundation.readouts):
            if hasattr(readout, "linear_1"):
                if readout.linear_1.weight.shape != source.linear_1.weight.shape:
                    logging.warning(
                        "Foundation readouts not loaded: the MLP of the last readout "
                        "has a different size (use the foundation's --MLP_irreps)"
                    )
                    return
        for readout, source in zip(model.readouts, model_foundation.readouts):
            if hasattr(readout, "linear_1"):
                readout.linear_1.weight.copy_(source.linear_1.weight)
                hidden = source.linear_2.weight.numel()
                readout.linear_2.weight.copy_(
                    source.linear_2.weight.view(hidden, 1).repeat(1, n_states).view(-1)
                )
            else:
                num_in = source.linear.weight.numel()
                if readout.linear.weight.numel() != num_in * n_states:
                    logging.warning("Foundation readouts not loaded: shapes differ")
                    return
                readout.linear.weight.copy_(
                    source.linear.weight.view(num_in, 1).repeat(1, n_states).view(-1)
                )
        if hasattr(model_foundation, "scale_shift"):
            scale = torch.atleast_1d(model_foundation.scale_shift.scale)[0]
            shift = torch.atleast_1d(model_foundation.scale_shift.shift)[0]
            model.scale_shift.scale.copy_(scale)
            model.scale_shift.shift.fill_(float(shift))
    logging.info("Every state starts from the foundation energy readout")
