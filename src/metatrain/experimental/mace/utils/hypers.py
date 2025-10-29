import argparse

from mace.tools import build_default_arg_parser


# Keys from the MACE argparser that correspond to model hyperparameters
MACE_MODEL_ARG_KEYS = [
    "num_radial_basis",
    "num_cutoff_basis",
    "max_ell",
    "interaction",
    "num_interactions",
    "hidden_irreps",
    "edge_irreps",
    "apply_cutoff",
    "avg_num_neighbors",
    "pair_repulsion",
    "distance_transform",
    "correlation",
    "gate",
    "interaction_first",
    "MLP_irreps",
    "radial_MLP",
    "radial_type",
    "use_embedding_readout",
    "use_last_readout_only",
    "use_agnostic_product",
]


def get_mace_defaults():
    parser = build_default_arg_parser()

    # Extract defaults without triggering required arguments
    mace_defaults = {
        action.dest: action.default
        for action in parser._actions
        if action.default is not argparse.SUPPRESS and action.dest != "help"
    }

    return mace_defaults
