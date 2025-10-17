"""
Fills the default_hypers_template.yaml with the default arguments
from MACE.
"""
import argparse
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

from mace.tools import build_default_arg_parser

# Keys from the MACE argparser that correspond to model hyperparameters
mace_model_keys = [
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
        if action.default is not argparse.SUPPRESS and action.dest != 'help'
    }

    # Currently the default for gate is 'silu', but this makes the model
    # no torchscriptable. We set it to None.
    mace_defaults["gate"] = None

    return mace_defaults

def regenerate_default_hypers():

    # Extract defaults without triggering required arguments
    mace_defaults = get_mace_defaults()

    # Render jinja2 template
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("default-hypers-template.yaml")

    mace_model_defaults = [(k, mace_defaults[k]) for k in mace_model_keys]

    with open(Path(__file__).parent / "default-hypers.yaml", "w") as f:
        f.write(template.render(mace_defaults=mace_defaults, mace_model_defaults=mace_model_defaults))

if __name__ == "__main__":
    regenerate_default_hypers()
