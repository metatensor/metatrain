"""
Tools to generate hyperparameter specifications for MACE models.

Simply run this file with python to generate a documentation.py file
with the MACE hyperparameter specification extracted from the MACE
argparser. The generated file will not be ready to be used as the
architecture's documentation file, it is just meant to help writing
the main documentation file by copy-pasting bits from the generated
file, instead of having to look manually at MACE's argparser.
"""

import argparse
import copy
from pathlib import Path

from mace.tools import build_default_arg_parser


_overwrite_defaults = {
    "hidden_irreps": "128x0e + 128x1o + 128x2e",
}


# Keys from the MACE argparser that correspond to model hyperparameters
MACE_MODEL_ARG_KEYS = [
    "r_max",
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


def get_mace_defaults() -> dict:
    parser = build_default_arg_parser()

    # Extract defaults without triggering required arguments
    mace_defaults = {
        action.dest: action.default
        for action in parser._actions
        if action.default is not argparse.SUPPRESS and action.dest != "help"
    }

    return mace_defaults


def get_mace_hypers_spec() -> dict:
    """Get the MACE hyperparameter specification.

    :return: A dictionary with the MACE hyperparameter specification.
    """

    def _get_type(action: argparse.Action) -> str:
        if action.dest == "radial_MLP":
            assert action.default == "[64, 64, 64]"
            return "list[int]"
        elif action.choices is not None:
            choices = copy.copy(list(action.choices))
            optional = False
            if "None" in choices:
                optional = True
                choices.remove("None")

            literal_str = (
                "Literal[" + ", ".join(repr(choice) for choice in choices) + "]"
            )
            if optional:
                return f"Optional[{literal_str}]"

            return literal_str

        optional = False
        if action.default is None:
            optional = True

        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            annotation = "bool"
        elif action.type is None:
            annotation = "str"
        elif hasattr(action.type, "__name__") and action.type.__name__ == "str2bool":
            annotation = "bool"
        else:
            annotation = getattr(action.type, "__name__", "str")

        if optional:
            annotation = f"Optional[{annotation}]"

        return annotation

    parser = build_default_arg_parser()
    hypers_spec = {
        action.dest: {
            "type": _get_type(action),
            "help": action.help,
            "default": action.default,
        }
        for action in parser._actions
        if action.dest != "help"
    }

    return hypers_spec


def write_mace_hypers_spec():
    """Writes the MACE hyperparameter specification to a documentation.py file.

    It extracts the hyperparameter specification from the MACE argparser
    and writes it to a documentation.py file in the same directory as this script.
    """

    with open(Path(__file__).parent / "documentation.py_template", "r") as f:
        template_content = f.read()

    def _get_default(key, spec):
        default = _overwrite_defaults.get(key, spec["default"])
        if "list" in spec["type"]:
            return default
        elif default == "None":
            return "None"
        return repr(default)

    hypers_spec = get_mace_hypers_spec()
    mace_hypers_str = ""
    for key in MACE_MODEL_ARG_KEYS:
        if key == "r_max":
            # r_max is documented separately
            continue
        spec = hypers_spec[key]
        mace_hypers_str += f"    {key}: {spec['type']} = {_get_default(key, spec)}\n"
        mace_hypers_str += f'    r"""{spec["help"]}"""\n'

    template_content = template_content.format(
        mace_hypers=mace_hypers_str,
        **{
            f"mace_param_{key}": (
                f"{hypers_spec[key]['type']} = {_get_default(key, hypers_spec[key])}"
            )
            for key in hypers_spec
        },
        **{f"mace_help_{key}": hypers_spec[key]["help"] for key in hypers_spec},
    )

    with open(Path(__file__).parent / "documentation.py", "w") as f:
        f.write(template_content)


if __name__ == "__main__":
    write_mace_hypers_spec()
