import ast
import logging

import numpy as np
import torch
from e3nn import o3

from mace import modules
from mace.data.excited import add_excited_state_keys, spec_layout
from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.tools.deprecation import warn
from mace.tools.finetuning_utils import load_foundations_elements, load_foundations_mdp
from mace.tools.scripts_utils import extract_config_mace_model, resolve_m_max
from mace.tools.torch_tools import dtype_dict
from mace.tools.utils import AtomicNumberTable

EXCITED_ERROR_TABLES = ("EnergyNacsDipoleMAE", "EnergyNacsDipoleRMSE")


def configure_excited_state_args(args, excited_spec) -> None:
    """Settings shared by ExcitedMACE and AutoencoderExcitedMACE runs."""
    if args.n_energies is None:
        if excited_spec.states is None:
            logging.warning("--n_energies not set, training a single electronic state")
        args.n_energies = excited_spec.n_states
    add_excited_state_keys(args.key_specification, args)
    if args.loss != "excited":
        if args.loss not in ("weighted", "ef"):
            logging.warning(
                f"--loss={args.loss} is not available for {args.model}, using --loss=excited"
            )
        args.loss = "excited"
    if args.error_table not in EXCITED_ERROR_TABLES:
        args.error_table = "EnergyNacsDipoleMAE"
    args.compute_stress = False
    args.compute_virials = False
    args.compute_forces = True
    logging.info(
        f"{args.model}: {excited_spec.describe()}, "
        f"coupling parity={getattr(args, 'coupling_parity', 'both')}, "
        f"pair repulsion={'on' if args.pair_repulsion else 'off'}, "
        f"backbone={getattr(args, 'excited_backbone', 'ScaleShiftMACE')}, "
        f"long range={getattr(args, 'excited_long_range', 'shared')}"
    )
    args.excited_spec = excited_spec


def configure_model(
    args,
    train_loader,
    atomic_energies,
    model_foundation=None,
    heads=None,
    z_table=None,
    head_configs=None,
):
    if args.model in ("ExcitedMACE", "AutoencoderExcitedMACE"):
        return configure_excited_model(
            args, train_loader, atomic_energies, model_foundation, heads, z_table
        )
    # Selecting outputs
    compute_virials = args.loss == "virials"
    compute_stress = args.loss in ("stress", "huber", "universal")

    if compute_virials:
        args.compute_virials = True
    elif compute_stress:
        args.compute_stress = True

    if compute_virials or compute_stress:
        if args.error_table in ["PerAtomRMSE", "PerAtomMAE", "TotalRMSE", "TotalMAE"]:
            args.error_table = (
                "PerAtomRMSEstressvirials"
                if "RMSE" in args.error_table
                else "PerAtomMAEstressvirials"
            )

    output_args = {
        "energy": args.compute_energy,
        "forces": args.compute_forces,
        "virials": compute_virials,
        "stress": compute_stress,
        "dipoles": args.compute_dipole,
        "polarizabilities": args.compute_polarizability,
        "magforces": args.compute_magforces,
    }
    logging.info(
        f"During training the following quantities will be reported: {', '.join([f'{report}' for report, value in output_args.items() if value])}"
    )
    logging.info("===========MODEL DETAILS===========")

    if args.scaling == "no_scaling":
        args.std = 1.0
        if head_configs is not None:
            for head_config in head_configs:
                head_config.std = 1.0
        logging.info("No scaling selected")

    if (
        head_configs is not None
        and args.std is not None
        and not isinstance(args.std, list)
    ):
        atomic_inter_scale = []
        for head_config in head_configs:
            if hasattr(head_config, "std") and head_config.std is not None:
                atomic_inter_scale.append(head_config.std)
            elif args.std is not None:
                atomic_inter_scale.append(
                    args.std if isinstance(args.std, float) else 1.0
                )
        args.std = atomic_inter_scale

    elif (args.mean is None or args.std is None) and (
        args.model not in ("AtomicDipolesMACE", "AtomicDielectricMACE")
    ):
        args.mean, args.std = modules.scaling_classes[args.scaling](
            train_loader, atomic_energies
        )
    if args.embedding_specs is not None:
        logging.info("Using embedding specifications from command line arguments")
        logging.info(f"Embedding specifications: {args.embedding_specs}")
    # Build model
    if model_foundation is not None and args.model in [
        "MACE",
        "ScaleShiftMACE",
        "MACELES",
        "PolarMACE",
    ]:
        logging.info("Loading FOUNDATION model")
        model_config_foundation = extract_config_mace_model(model_foundation)
        model_config_foundation["atomic_energies"] = atomic_energies

        if args.embedding_specs:
            model_config_foundation["embedding_specs"] = args.embedding_specs

        if args.foundation_model_elements:
            foundation_z_table = AtomicNumberTable(
                [int(z) for z in model_foundation.atomic_numbers]
            )
            model_config_foundation["atomic_numbers"] = foundation_z_table.zs
            model_config_foundation["num_elements"] = len(foundation_z_table)
            z_table = foundation_z_table
            logging.info(
                f"Using all elements from foundation model: {foundation_z_table.zs}"
            )
        else:
            model_config_foundation["atomic_numbers"] = z_table.zs
            model_config_foundation["num_elements"] = len(z_table)
            logging.info(f"Using filtered elements: {z_table.zs}")

        args.max_L = model_config_foundation["hidden_irreps"].lmax

        if args.model in (
            "ScaleShiftMACE",
            "PolarMACE",
            "MagneticScaleShiftMACE",
        ) or model_foundation.__class__.__name__ in (
            "ScaleShiftMACE",
            "PolarMACE",
            "MagneticScaleShiftMACE",
        ):
            model_config_foundation["atomic_inter_shift"] = (
                _determine_atomic_inter_shift(args.mean, heads)
            )
        else:
            model_config_foundation["atomic_inter_shift"] = [0.0] * len(heads)
        model_config_foundation["atomic_inter_scale"] = [1.0] * len(heads)
        args.avg_num_neighbors = model_config_foundation["avg_num_neighbors"]
        if args.model == "MACELES":
            args.model = "FoundationMACELES"
        elif args.model in ("MACE", "ScaleShiftMACE"):
            args.model = "FoundationMACE"
        model_config_foundation["heads"] = heads
        model_config = model_config_foundation

        logging.info("Model configuration extracted from foundation model")
        logging.info(f"Using {args.loss} loss function for fine-tuning")
        logging.info(
            f"Message passing with hidden irreps {model_config_foundation['hidden_irreps']})"
        )
        logging.info(
            f"{model_config_foundation['num_interactions']} layers, each with correlation order: {model_config_foundation['correlation']} (body order: {model_config_foundation['correlation']+1}) and spherical harmonics up to: l={model_config_foundation['max_ell']}"
        )
        logging.info(
            f"Radial cutoff: {model_config_foundation['r_max']} A (total receptive field for each atom: {model_config_foundation['r_max'] * model_config_foundation['num_interactions']} A)"
        )
        logging.info(
            f"Distance transform for radial basis functions: {model_config_foundation['distance_transform']}"
        )
    else:
        logging.info("Building model")
        logging.info(
            f"Message passing with {args.num_channels} channels and max_L={args.max_L} ({args.hidden_irreps})"
        )
        logging.info(
            f"{args.num_interactions} layers, each with correlation order: {args.correlation} (body order: {args.correlation+1}) and spherical harmonics up to: l={args.max_ell}"
        )
        logging.info(
            f"{args.num_radial_basis} radial and {args.num_cutoff_basis} basis functions"
        )
        logging.info(
            f"Radial cutoff: {args.r_max} A (total receptive field for each atom: {args.r_max * args.num_interactions} A)"
        )
        logging.info(
            f"Distance transform for radial basis functions: {args.distance_transform}"
        )

        assert (
            len({irrep.mul for irrep in o3.Irreps(args.hidden_irreps)}) == 1
        ), "All channels must have the same dimension, use the num_channels and max_L keywords to specify the number of channels and the maximum L"

        logging.info(f"Hidden irreps: {args.hidden_irreps}")

        cueq_config = None
        if args.only_cueq:
            logging.info("Using only the backend of the model")
            cueq_config = CuEquivarianceConfig(
                enabled=True,
                layout="ir_mul",
                group="O3_e3nn",
                optimize_all=True,
                conv_fusion=args.cueq_conv_fusion
                and torch.device(args.device).type == "cuda",
            )

        model_config = dict(
            r_max=args.r_max,
            num_bessel=args.num_radial_basis,
            num_polynomial_cutoff=args.num_cutoff_basis,
            max_ell=args.max_ell,
            interaction_cls=modules.interaction_classes[args.interaction],
            num_interactions=args.num_interactions,
            num_elements=len(z_table),
            hidden_irreps=o3.Irreps(args.hidden_irreps),
            edge_irreps=o3.Irreps(args.edge_irreps) if args.edge_irreps else None,
            atomic_energies=atomic_energies,
            apply_cutoff=args.apply_cutoff,
            avg_num_neighbors=args.avg_num_neighbors,
            atomic_numbers=z_table.zs,
            use_reduced_cg=args.use_reduced_cg,
            use_so3=args.use_so3,
            use_edge_irreps_first=args.use_edge_irreps_first,
            cueq_config=cueq_config,
        )
        model_config_foundation = None

    model = _build_model(args, model_config, model_config_foundation, heads)

    if model_foundation is not None:
        if getattr(args, "finetune_dipoles_polarizabilities", False):
            # MDP fine-tuning: dedicated loader that handles higher-order irreps
            load_foundations_mdp(model, model_foundation, z_table, max_L=args.max_L)
        else:
            model = load_foundations_elements(
                model,
                model_foundation,
                z_table,
                load_readout=args.foundation_model_readout,
                max_L=args.max_L,
                default_dtype=dtype_dict.get(args.default_dtype, torch.float64),
            )

    return model, output_args


def _determine_atomic_inter_shift(mean, heads):
    if isinstance(mean, np.ndarray):
        if mean.size == 1:
            return mean.item()
        if mean.size == len(heads):
            return mean.tolist()
        logging.info("Mean not in correct format, using default value of 0.0")
        return [0.0] * len(heads)
    if isinstance(mean, list) and len(mean) == len(heads):
        return mean
    if isinstance(mean, float):
        return [mean] * len(heads)
    logging.info("Mean not in correct format, using default value of 0.0")
    return [0.0] * len(heads)


def _parse_literal_or_none(value):
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in ("none", ""):
            return None
        return ast.literal_eval(stripped)
    return value


FOUNDATION_BACKBONES = {
    "ScaleShiftMACE": "ScaleShiftMACE",
    "PolarMACE": "PolarMACE",
    "MACELES": "MACELES",
}


def _foundation_as_backbone(args, model_foundation, heads, z_table):
    """A foundation model whose architecture cannot be re-built for the training
    elements, used unchanged as the backbone (its own E0s, scale and shift)."""
    import copy  # pylint: disable=import-outside-toplevel

    foundation_cls = model_foundation.__class__.__name__
    zs = [int(z) for z in model_foundation.atomic_numbers]
    if list(z_table.zs) != zs:
        raise ValueError(
            f"A {foundation_cls} foundation model is used as it is, so the training "
            f"data must use its elements {zs} (got {list(z_table.zs)}); add "
            "--foundation_model_elements=True"
        )
    foundation_heads = [str(h) for h in getattr(model_foundation, "heads", ["Default"])]
    head = getattr(args, "foundation_head", None) or foundation_heads[0]
    if head not in foundation_heads:
        raise ValueError(f"--foundation_head {head} not in {foundation_heads}")
    backbone_heads = [foundation_heads.index(head)] * len(heads)
    args.r_max = float(model_foundation.r_max)
    args.avg_num_neighbors = float(
        getattr(model_foundation.interactions[0], "avg_num_neighbors", 1.0)
    )
    logging.info(
        f"Ground-state backbone: the {foundation_cls} foundation model as it is "
        f"(head {head} for every head)"
    )
    output_args = {
        "energy": True,
        "forces": True,
        "virials": False,
        "stress": False,
        "dipoles": False,
        "polarizabilities": False,
        "magforces": False,
    }
    return copy.deepcopy(model_foundation), backbone_heads, output_args


def excited_head_statistics(train_loader, atomic_energies, heads, specs, scaling):
    """(scale, shifts[S]) of every head from its own training frames: shifts are the
    mean (E_s - sum E0)/n_atoms per state, scale the rms of the forces ("std_scaling":
    the std of the interaction energies per atom; "no_scaling": 1 and 0)."""
    from mace.tools.scatter import (  # pylint: disable=import-outside-toplevel
        scatter_sum,
    )

    energies = torch.as_tensor(np.atleast_2d(atomic_energies))
    inter = {h: [] for h in range(len(heads))}
    forces = {h: [] for h in range(len(heads))}
    for batch in train_loader:
        graph_heads = batch.head.view(-1).to(torch.long)
        num_atoms = (batch.ptr[1:] - batch.ptr[:-1]).to(batch.positions.dtype)
        element = torch.argmax(batch.node_attrs, dim=1)
        node_heads = graph_heads[batch.batch]
        table = energies.to(batch.positions.dtype)
        node_e0 = table[torch.clamp(node_heads, max=table.shape[0] - 1), element]
        graph_e0 = scatter_sum(node_e0, batch.batch, dim=0, dim_size=batch.num_graphs)
        energy = batch.energy.view(batch.num_graphs, -1)
        per_atom = (energy - graph_e0.unsqueeze(-1)) / num_atoms.unsqueeze(-1)
        labelled = (batch.weight * batch.energy_weight) > 0
        atom_labelled = torch.repeat_interleave(
            (batch.weight * batch.forces_weight) > 0, batch.ptr[1:] - batch.ptr[:-1]
        )
        batch_forces = batch.forces.view(batch.forces.shape[0], -1, 3)
        for h, name in enumerate(heads):
            n_states = specs[name].n_states
            rows = (graph_heads == h) & labelled
            inter[h].append(per_atom[rows, :n_states])
            atoms = (node_heads == h) & atom_labelled
            forces[h].append(batch_forces[atoms, :n_states])
    stats = []
    for h, name in enumerate(heads):
        n_states = specs[name].n_states
        values = torch.cat(inter[h], dim=0)
        head_forces = torch.cat(forces[h], dim=0)
        if scaling == "no_scaling":
            stats.append((1.0, [0.0] * n_states))
            continue
        shift = values.mean(dim=0).tolist() if values.shape[0] else [0.0] * n_states
        if scaling == "std_scaling":
            scale = float(torch.std(values)) if values.numel() > 1 else 1.0
        else:
            scale = (
                float(torch.sqrt(torch.mean(torch.square(head_forces))))
                if head_forces.numel()
                else 1.0
            )
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        stats.append((scale, shift))
    return stats


def configure_excited_model(
    args, train_loader, atomic_energies, model_foundation, heads, z_table
):
    """An excited-state model: a ground-state backbone of any MACE class (built from
    scratch or from a foundation model, with all heads) and excited-state heads."""
    import copy  # pylint: disable=import-outside-toplevel

    heads = list(heads)
    specs = args.excited_specs
    layouts = [spec_layout(specs[name]) for name in heads]
    stats = excited_head_statistics(
        train_loader, atomic_energies, heads, specs, args.scaling
    )
    for name, (scale, shift) in zip(heads, stats):
        logging.info(
            f"Head {name}: {specs[name].describe()}; interaction energy scale "
            f"{scale:.4f}, shift per state "
            + ", ".join(f"{x:.4f}" for x in np.atleast_1d(shift))
        )

    backbone_args = copy.copy(args)
    foundation_cls = None
    backbone_heads = None
    if model_foundation is not None:
        foundation_cls = model_foundation.__class__.__name__
    if foundation_cls is not None and foundation_cls not in FOUNDATION_BACKBONES:
        # architectures MACE cannot re-build for new elements (plain MACE, magnetic
        # models, ...): the foundation model is the backbone as it is, every head on
        # its (selected) head
        backbone, backbone_heads, output_args = _foundation_as_backbone(
            args, model_foundation, heads, z_table
        )
    else:
        if foundation_cls is not None:
            backbone_args.model = FOUNDATION_BACKBONES[foundation_cls]
        else:
            backbone_args.model = getattr(args, "excited_backbone", "ScaleShiftMACE")
        backbone_args.loss = "weighted"
        backbone_args.error_table = "PerAtomMAE"
        backbone_args.std = [float(scale) for scale, _ in stats]
        backbone_args.mean = np.array(
            [float(np.mean(shift)) for _, shift in stats], dtype=float
        )
        logging.info(f"Ground-state backbone: {backbone_args.model}")
        backbone, output_args = configure_model(
            backbone_args,
            train_loader,
            atomic_energies,
            model_foundation,
            heads,
            z_table,
        )
        for key in ("max_L", "avg_num_neighbors", "MLP_irreps", "r_max"):
            if hasattr(backbone_args, key):
                setattr(args, key, getattr(backbone_args, key))

    backbone_channels = o3.Irreps(str(backbone.products[-1].linear.irreps_out)).count(
        o3.Irrep(0, 1)
    )
    channels = max(8, min(64, backbone_channels))
    adapter_irreps = getattr(args, "excited_adapter_irreps", None) or (
        f"{channels}x0e + {channels}x1o"
    )
    kwargs = dict(
        backbone=backbone,
        head_layouts=layouts,
        MLP_irreps=o3.Irreps(
            getattr(args, "excited_MLP_irreps", None) or args.MLP_irreps
        ),
        gate=modules.gate_dict[args.gate],
        adapter_irreps=o3.Irreps(adapter_irreps),
        adapter_max_ell=args.max_ell,
        long_range_mode=getattr(args, "excited_long_range", "shared"),
        charge_smearing=getattr(args, "excited_charge_smearing", 1.0),
        coupling_parity=getattr(args, "coupling_parity", "both"),
        delta_scale=[float(scale) for scale, _ in stats],
        delta_shift=[
            [float(x) - float(np.mean(shift)) for x in np.atleast_1d(shift)]
            for _, shift in stats
        ],
    )
    if backbone_heads is not None:
        kwargs.update(heads=list(heads), backbone_heads=backbone_heads)
    if args.model == "AutoencoderExcitedMACE":
        model = modules.AutoencoderExcitedMACE(
            **kwargs, num_permutational_invariant=args.num_permutational_invariant
        )
    else:
        model = modules.ExcitedMACE(**kwargs)
    if model_foundation is not None:
        # states start from the foundation's energy: fit their shifts to the data
        logging.info(
            f"Backbone architecture taken from the foundation model ({foundation_cls}); "
            "every state starts from the foundation energy"
        )
        device = torch.device(args.device)
        model.to(device)
        modules.fit_excited_shifts(model, train_loader, device=device)
        logging.info(
            "Per-state shifts relative to the foundation backbone: "
            + "; ".join(
                f"{name}: "
                + ", ".join(
                    f"{x:.4f}"
                    for x in model.delta_shift[h, : specs[name].n_states].tolist()
                )
                for h, name in enumerate(heads)
            )
        )
    logging.info(
        f"Excited-state model on a {backbone.__class__.__name__} backbone with heads "
        + ", ".join(f"{name} ({specs[name].describe()})" for name in heads)
    )
    return model, output_args


def _build_model(
    args, model_config, model_config_foundation, heads
):  # pylint: disable=too-many-return-statements
    if args.model == "MagneticScaleShiftMACE":
        m_max = resolve_m_max(args.m_max, list(model_config["atomic_numbers"]))
        return modules.MagneticScaleShiftMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=_determine_atomic_inter_shift(args.mean, heads),
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
            m_max=m_max,
            max_m_ell=args.max_m_ell,
            num_mag_radial_basis=args.num_mag_radial_basis,
            num_mag_radial_basis_one_body=args.num_mag_radial_basis_one_body,
            use_magmom_one_body=args.use_magmom_one_body,
        )
    if args.model == "MACE":
        if args.interaction_first not in [
            "RealAgnosticInteractionBlock",
            "RealAgnosticDensityInteractionBlock",
            "RealAgnosticResidualNonLinearInteractionBlock",
        ]:
            warn(
                "pkg.first_block_coercion",
                context=f"--interaction_first {args.interaction_first} is being "
                "replaced by RealAgnosticInteractionBlock for --model MACE",
            )
            args.interaction_first = "RealAgnosticInteractionBlock"
        return modules.ScaleShiftMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=[0.0] * len(heads),
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
            embedding_specs=args.embedding_specs,
            use_embedding_readout=args.use_embedding_readout,
            use_last_readout_only=args.use_last_readout_only,
            use_agnostic_product=args.use_agnostic_product,
        )
    if args.model == "ScaleShiftMACE":
        return modules.ScaleShiftMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=args.mean,
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
            embedding_specs=args.embedding_specs,
            use_embedding_readout=args.use_embedding_readout,
            use_last_readout_only=args.use_last_readout_only,
            use_agnostic_product=args.use_agnostic_product,
        )
    if args.model == "PolarMACE" and model_config_foundation is not None:
        return modules.PolarMACE(**model_config_foundation)
    if args.model == "PolarMACE":
        field_feature_widths = _parse_literal_or_none(args.field_feature_widths)
        field_feature_norms = _parse_literal_or_none(args.field_feature_norms)
        fixedpoint_update_config = _parse_literal_or_none(args.fixedpoint_update_config)
        field_readout_config = _parse_literal_or_none(args.field_readout_config)
        return modules.PolarMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=_determine_atomic_inter_shift(args.mean, heads),
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
            embedding_specs=args.embedding_specs,
            use_embedding_readout=args.use_embedding_readout,
            use_last_readout_only=args.use_last_readout_only,
            use_agnostic_product=args.use_agnostic_product,
            kspace_cutoff_factor=args.kspace_cutoff_factor,
            atomic_multipoles_max_l=args.atomic_multipoles_max_l,
            atomic_multipoles_smearing_width=args.atomic_multipoles_smearing_width,
            field_feature_max_l=args.field_feature_max_l,
            field_feature_widths=(
                field_feature_widths if field_feature_widths is not None else [1.0]
            ),
            num_recursion_steps=args.num_recursion_steps,
            field_si=args.field_si,
            include_electrostatic_self_interaction=args.include_electrostatic_self_interaction,
            add_local_electron_energy=args.add_local_electron_energy,
            quadrupole_feature_corrections=args.quadrupole_feature_corrections,
            return_electrostatic_potentials=args.return_electrostatic_potentials,
            field_feature_norms=field_feature_norms,
            field_norm_factor=args.field_norm_factor,
            fixedpoint_update_config=fixedpoint_update_config,
            field_readout_config=field_readout_config,
        )
    if args.model == "FoundationMACE":
        return modules.ScaleShiftMACE(**model_config_foundation)
    if args.model == "FoundationMACELES":
        from mace.modules.extensions import MACELES

        return MACELES(
            les_arguments=args.les_arguments,
            **model_config_foundation,
        )
    if args.model == "ScaleShiftBOTNet":
        # say it is deprecated
        raise RuntimeError("ScaleShiftBOTNet is deprecated, use MACE instead")
    if args.model == "BOTNet":
        raise RuntimeError("BOTNet is deprecated, use MACE instead")
    if args.model == "AtomicDipolesMACE":
        assert args.loss == "dipole", "Use dipole loss with AtomicDipolesMACE model"
        assert (
            args.error_table == "DipoleRMSE"
        ), "Use error_table DipoleRMSE with AtomicDipolesMACE model"
        return modules.AtomicDipolesMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
        )

    if args.model == "AtomicDielectricMACE":
        args.error_table = "DipolePolarRMSE"
        # std_df = modules.scaling_classes["rms_dipoles_scaling"](train_loader)
        assert (
            args.loss == "dipole_polar"
        ), "Use dipole_polar loss with AtomicDielectricMACE model"
        assert args.error_table in (
            "DipoleRMSE",
            "DipolePolarRMSE",
        ), "Use error_table DipoleRMSE with AtomicDielectricMACE model"
        return modules.AtomicDielectricMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            use_polarizability=True,
        )

    if args.model == "EnergyDipolesMACE":
        assert (
            args.loss == "energy_forces_dipole"
        ), "Use energy_forces_dipole loss with EnergyDipolesMACE model"
        assert (
            args.error_table == "EnergyDipoleRMSE"
        ), "Use error_table EnergyDipoleRMSE with AtomicDipolesMACE model"
        return modules.EnergyDipolesMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
        )
    if args.model == "MACELES":
        from mace.modules.extensions import MACELES

        return MACELES(
            les_arguments=args.les_arguments,
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=[0.0] * len(heads),
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
            embedding_specs=args.embedding_specs,
            use_embedding_readout=args.use_embedding_readout,
            use_last_readout_only=args.use_last_readout_only,
            use_agnostic_product=args.use_agnostic_product,
        )
    raise RuntimeError(f"Unknown model: '{args.model}'")
