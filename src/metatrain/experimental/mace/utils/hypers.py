import argparse
import copy

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


def get_mace_hypers_spec():
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
