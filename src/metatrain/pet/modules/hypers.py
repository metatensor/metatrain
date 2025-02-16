import re
import warnings

import yaml  # type: ignore


def propagate_duplicated_params(provided_hypers, default_hypers, first_key, second_key):
    if (first_key in provided_hypers.keys()) and (second_key in provided_hypers.keys()):
        raise ValueError(f"only one of {first_key} and {second_key} should be provided")

    if (first_key in default_hypers.keys()) and (second_key in default_hypers.keys()):
        raise ValueError(
            f"only one of {first_key} and {second_key} should be in default hypers"
        )

    output_key, output_value = None, None
    for key in [first_key, second_key]:
        if key in provided_hypers.keys():
            output_key = key
            output_value = provided_hypers[key]

    if output_key is None:
        for key in [first_key, second_key]:
            if key in default_hypers.keys():
                output_key = key
                output_value = default_hypers[key]

    if output_key is None:
        raise ValueError(f"{first_key} or {second_key} must be provided somewhere")

    return output_key, output_value


def check_is_shallow(hypers):
    for key in hypers.keys():
        if isinstance(hypers[key], dict):
            raise ValueError("Nesting of more than two is not supported")


def combine_hypers(provided_hypers, default_hypers):
    group_keys = [
        "ARCHITECTURAL_HYPERS",
        "FITTING_SCHEME",
        "MLIP_SETTINGS",
        "GENERAL_TARGET_SETTINGS",
        "UTILITY_FLAGS",
    ]

    for key in provided_hypers.keys():
        if key not in group_keys:
            raise ValueError(f"unknown hyper parameter {key}")

    for key in default_hypers.keys():
        if key not in group_keys:
            raise ValueError(f"unknown hyper parameter {key}")

    result = {}
    for key in group_keys:
        default_now = default_hypers[key]
        if key in provided_hypers.keys():
            provided_now = provided_hypers[key]
        else:
            provided_now = {}
        if key == "FITTING_SCHEME":
            duplicated_params = [
                ["ATOMIC_BATCH_SIZE", "STRUCTURAL_BATCH_SIZE"],
                ["EPOCH_NUM", "EPOCH_NUM_ATOMIC"],
                ["SCHEDULER_STEP_SIZE", "SCHEDULER_STEP_SIZE_ATOMIC"],
                ["EPOCHS_WARMUP", "EPOCHS_WARMUP_ATOMIC"],
            ]
        else:
            duplicated_params = []
        result[key] = combine_hypers_shallow(
            provided_now, default_now, duplicated_params
        )

    if (not result["MLIP_SETTINGS"]["USE_ENERGIES"]) and (
        not result["MLIP_SETTINGS"]["USE_FORCES"]
    ):
        raise ValueError(
            "At least one of the energies and forces should be used for fitting"
        )

    if (not result["MLIP_SETTINGS"]["USE_ENERGIES"]) or (
        not result["MLIP_SETTINGS"]["USE_FORCES"]
    ):
        if result["FITTING_SCHEME"]["ENERGY_WEIGHT"] is not None:
            warnings.warn(
                "ENERGY_WEIGHT was provided, but in the current calculation, "
                + "it doesn't ffect anything since only one target of energies and "
                + "forces is used",
                stacklevel=2,
            )

    if result["ARCHITECTURAL_HYPERS"]["USE_ADDITIONAL_SCALAR_ATTRIBUTES"]:
        if result["ARCHITECTURAL_HYPERS"]["SCALAR_ATTRIBUTES_SIZE"] is None:
            raise ValueError(
                "scalar attributes size must be provided if "
                "use_additional_scalar_attributes == True"
            )

    if result["FITTING_SCHEME"]["DO_GRADIENT_CLIPPING"]:
        if result["FITTING_SCHEME"]["GRADIENT_CLIPPING_MAX_NORM"] is None:
            raise ValueError(
                "gradient clipping max_norm must be provided if "
                "do_gradient_clipping == True"
            )

    if result["FITTING_SCHEME"]["BALANCED_DATA_LOADER"]:
        if "STRUCTURAL_BATCH_SIZE" in result["FITTING_SCHEME"].keys():
            if result["FITTING_SCHEME"]["STRUCTURAL_BATCH_SIZE"] is not None:
                raise ValueError(
                    "if using balanced_data_loader only atomic batch size "
                    "can be provided"
                )

    if result["FITTING_SCHEME"]["MODEL_TO_START_WITH"] is not None:
        if result["FITTING_SCHEME"]["ALL_SPECIES_PATH"] is None:
            raise ValueError(
                "When fine-tuning the model starting from a checkpoint "
                "[MODEL_TO_START_WITH, the path to the all_species.npy "
                " file [ALL_SPECIES_PATH] must be provided."
            )

    return result


def combine_hypers_shallow(provided_hypers, default_hypers, duplicated_params):
    check_is_shallow(provided_hypers)
    check_is_shallow(default_hypers)

    duplicated_params_unrolled = []
    for el in duplicated_params:
        duplicated_params_unrolled.append(el[0])
        duplicated_params_unrolled.append(el[1])

    for key in provided_hypers.keys():
        if key not in default_hypers.keys():
            if key not in duplicated_params_unrolled:
                raise ValueError(f"unknown hyper parameter {key}")

    result = {}

    for key in default_hypers.keys():
        if key in provided_hypers.keys():
            if key not in duplicated_params_unrolled:
                result[key] = provided_hypers[key]
        else:
            if key not in duplicated_params_unrolled:
                result[key] = default_hypers[key]

    for el in duplicated_params:
        dupl_key, dupl_value = propagate_duplicated_params(
            provided_hypers, default_hypers, el[0], el[1]
        )
        result[dupl_key] = dupl_value

    return result


def fix_Nones_in_yaml(hypers_dict):
    for key in hypers_dict.keys():
        if (hypers_dict[key] == "None") or (hypers_dict[key] == "none"):
            hypers_dict[key] = None
        if isinstance(hypers_dict[key], dict):
            fix_Nones_in_yaml(hypers_dict[key])


class Hypers:
    def __init__(self, hypers_dict):
        for key, value in hypers_dict.items():
            if isinstance(value, dict):
                self.__dict__[key] = Hypers(value)
            else:
                self.__dict__[key] = value


def load_hypers_from_file(path_to_hypers):
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )

    with open(path_to_hypers, "r") as f:
        hypers = yaml.load(f, Loader=loader)
        fix_Nones_in_yaml(hypers)

    return Hypers(hypers)


def set_hypers_from_files(path_to_provided_hypers, path_to_default_hypers):
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )

    with open(path_to_provided_hypers, "r") as f:
        provided_hypers = yaml.load(f, Loader=loader)
        fix_Nones_in_yaml(provided_hypers)

    with open(path_to_default_hypers, "r") as f:
        default_hypers = yaml.load(f, Loader=loader)
        fix_Nones_in_yaml(default_hypers)

    combined_hypers = combine_hypers(provided_hypers, default_hypers)
    return Hypers(combined_hypers)


def hypers_to_dict(obj):
    if isinstance(obj, Hypers):
        return {key: hypers_to_dict(value) for key, value in obj.__dict__.items()}
    else:
        return obj


def save_hypers(hypers, path_save):
    hypers_dict = hypers_to_dict(hypers)
    with open(path_save, "w") as f:
        yaml.dump(hypers_dict, f)
