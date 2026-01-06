def model_update_v1_v2(checkpoint: dict) -> None:
    """
    Update model checkpoint from version 1 to version 2.

    The only change is to fix a typo in the unit of momentum targets.

    :param checkpoint: The checkpoint to update.
    """
    for _, target in checkpoint["model_data"]["dataset_info"].targets.items():
        if target.unit == "(eV*u)^1/2":
            target.unit = "(eV*u)^(1/2)"


def model_update_v2_v3(checkpoint: dict) -> None:
    """
    Update model checkpoint from version 2 to version 3.

    :param checkpoint: The checkpoint to update.
    """
    # Add missing activation_temperature hyperparameter with default value
    if "activation_temperature" not in checkpoint["model_hypers"]:
        checkpoint["model_hypers"]["activation_temperature"] = 1.0


def trainer_update_v1_v2(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 1 to version 2.

    :param checkpoint: The checkpoint to update.
    """
    # remove all entries in the loss `sliding_factor`
    old_loss_hypers = checkpoint["train_hypers"]["loss"].copy()
    dataset_info = checkpoint["model_data"]["dataset_info"]
    new_loss_hypers = {}

    for target_name in dataset_info.targets.keys():
        # retain everything except sliding_factor for each target
        new_loss_hypers[target_name] = {
            k: v
            for k, v in old_loss_hypers[target_name].items()
            if k != "sliding_factor"
        }

    checkpoint["train_hypers"]["loss"] = new_loss_hypers


def trainer_update_v2_v3(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 2 to version 3.

    :param checkpoint: The checkpoint to update.
    """
    # - Rename ``fixed_composition_weights`` to ``atomic_baseline``.
    atomic_baseline = checkpoint["train_hypers"].pop("fixed_composition_weights")
    checkpoint["train_hypers"]["atomic_baseline"] = atomic_baseline
