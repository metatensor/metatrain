from typing import Any, Dict


def update_hypers(
    hypers: Dict[str, Any], model_hypers: Dict[str, Any], do_forces: bool = True
):
    """
    Updates the hypers dictionary with the model hypers, the
    MLIP_SETTINGS and UTILITY_FLAGS keys of the PET model.
    """

    # set model hypers
    hypers = hypers.copy()
    hypers["ARCHITECTURAL_HYPERS"] = model_hypers
    hypers["ARCHITECTURAL_HYPERS"]["DTYPE"] = "float32"
    hypers["ARCHITECTURAL_HYPERS"]["D_OUTPUT"] = 1
    hypers["ARCHITECTURAL_HYPERS"]["TARGET_TYPE"] = "structural"
    hypers["ARCHITECTURAL_HYPERS"]["TARGET_AGGREGATION"] = "sum"

    # set MLIP_SETTINGS
    hypers["MLIP_SETTINGS"] = {
        "ENERGY_KEY": "energy",
        "FORCES_KEY": "forces",
        "USE_ENERGIES": True,
        "USE_FORCES": do_forces,
    }

    # set PET utility flags
    hypers["UTILITY_FLAGS"] = {
        "CALCULATION_TYPE": None,
    }
    return hypers
