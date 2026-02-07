import re

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


###########################
# MODEL ###################
###########################


def model_update_v1_v2(checkpoint: dict) -> None:
    """
    Update model checkpoint from version 1 to version 2.

    We moved the ``type_to_index`` mapping inside the composition model instead of
    the BaseCompositionModel.

    :param checkpoint: The checkpoint to update.
    """
    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is not None:
            state_dict["additive_models.0.model.type_to_index"] = state_dict.pop(
                "additive_models.0.type_to_index"
            )


def model_update_v2_v3(checkpoint: dict) -> None:
    """
    Update model checkpoint from version 2 to version 3.

    This update makes sure all optional fields are present in the checkpoint, setting
    them as ``None`` if they were not present before.

    :param checkpoint: The checkpoint to update.
    """
    # explicitly set epoch and best_epoch to `None` if they do not exist
    checkpoint["epoch"] = checkpoint.get("epoch")
    checkpoint["best_epoch"] = checkpoint.get("best_epoch")

    if checkpoint["best_model_state_dict"] is not None:
        checkpoint["best_model_state_dict"] = checkpoint.get("best_model_state_dict")


def model_update_v3_v4(checkpoint: dict) -> None:
    """
    Update model checkpoint from version 3 to version 4.

    This update changes the way target scaling factors are stored. Previously, a
    single tensor of scales was stored, now each target has its own scale TensorMap.

    :param checkpoint: The checkpoint to update.
    """
    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is not None:
            if (
                "scaler.scales" not in state_dict
                and "scaler.dummy_buffer" in state_dict
                and "scaler.model.type_to_index" in state_dict
            ):
                continue  # already updated
            old_scales_tensor = state_dict.pop("scaler.scales")
            old_output_name_to_output_index = {}
            for target_index, target_name in enumerate(
                checkpoint["model_data"]["dataset_info"].targets.keys()
            ):
                old_output_name_to_output_index[target_name] = target_index
            state_dict["scaler.dummy_buffer"] = torch.tensor(
                [0.0], dtype=old_scales_tensor.dtype
            )
            state_dict["scaler.model.type_to_index"] = state_dict[
                "additive_models.0.model.type_to_index"
            ]

            all_targets = checkpoint["model_data"]["dataset_info"].targets
            for target_name, target_info in all_targets.items():
                layout = target_info.layout
                if layout.sample_names == ["system"]:
                    samples = Labels(["atomic_type"], torch.tensor([[-1]]))

                elif layout.sample_names == ["system", "atom"]:
                    samples = Labels(
                        ["atomic_type"],
                        torch.arange(
                            len(checkpoint["model_data"]["dataset_info"].atomic_types)
                        ).reshape(-1, 1),
                    )
                else:
                    raise ValueError(  # will never happen
                        "Unknown sample kind. Please contact the developers."
                    )
                scales_tensormap = TensorMap(
                    keys=layout.keys,
                    blocks=[
                        TensorBlock(
                            values=torch.full(  # important when scale_targets=False
                                (len(samples), len(block.properties)),
                                old_scales_tensor[
                                    old_output_name_to_output_index[target_name]
                                ],
                                dtype=torch.float64,
                            ),
                            samples=samples,
                            components=[],
                            properties=block.properties,
                        )
                        for block in layout.blocks()
                    ],
                )
                state_dict[f"scaler.{target_name}_scaler_buffer"] = mts.save_buffer(
                    mts.make_contiguous(scales_tensormap)
                )


def model_update_v4_v5(checkpoint: dict) -> None:
    """
    Update model checkpoint from version 4 to version 5.

    The main change is the use of mts.torch.nn.Module as a base class for
    mts.torch.nn.Linear and other layers, which now has a `module_list` sub-module.

    :param checkpoint: The checkpoint to update.
    """
    LAST_LAYER_REGEX = re.compile(r"last_layers\.(.*)\.module_map\.(\d+)\.weight")

    for key in ["model_state_dict", "best_model_state_dict"]:
        state_dict = checkpoint.get(key)
        if state_dict is not None:
            new_state_dict = {}
            last_layer_entries = set()
            for name, value in state_dict.items():
                if name.startswith("layernorm."):
                    new_name = name.replace("layernorm.", "layernorm.module_list.")
                    new_state_dict[new_name] = value
                elif name.startswith("bpnn."):
                    new_name = name.replace("bpnn.", "bpnn.module_list.")
                    new_state_dict[new_name] = value
                else:
                    match = re.match(LAST_LAYER_REGEX, name)
                    if match is not None:
                        last_layer_entries.add(match.group(1))
                        new_name = (
                            f"last_layers.{match.group(1)}.module_map."
                            + f"module_list.{match.group(2)}.weight"
                        )
                        new_state_dict[new_name] = value
                    else:
                        new_state_dict[name] = value

            dtype = state_dict["layernorm.0.weight"].dtype
            device = state_dict["layernorm.1.weight"].device
            mts_helper = torch.zeros(0, dtype=dtype, device=device)

            new_state_dict["layernorm._mts_helper"] = mts_helper
            # This should contain the serialized _in_keys and _out_properties, but we
            # can not recover them here, so we set them to empty dicts and hope they
            # where properly set when creating the model instance.
            new_state_dict["layernorm._extra_state"] = {}

            new_state_dict["bpnn._mts_helper"] = mts_helper
            new_state_dict["bpnn._extra_state"] = {}

            for target in last_layer_entries:
                new_state_dict[f"last_layers.{target}._mts_helper"] = mts_helper
                new_state_dict[f"last_layers.{target}._extra_state"] = {}

                new_state_dict[f"last_layers.{target}.module_map._mts_helper"] = (
                    mts_helper
                )
                new_state_dict[f"last_layers.{target}.module_map._extra_state"] = {}

            checkpoint[key] = new_state_dict


def model_update_v5_v6(checkpoint: dict) -> None:
    """
    Update model checkpoint from version 5 to version 6.

    :param checkpoint: The checkpoint to update.
    """
    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is not None:
            for key in list(state_dict.keys()):
                if "soap_calculator.calculator." in key:
                    new_key = key.replace(
                        "soap_calculator.calculator.", "soap_calculator."
                    )
                    state_dict[new_key] = state_dict.pop(key)


def model_update_v6_v7(checkpoint: dict) -> None:
    """
    Update model checkpoint from version 6 to version 7.

    :param checkpoint: The checkpoint to be updated.
    """
    checkpoint["model_data"]["model_hypers"]["legacy"] = True


def model_update_v7_v8(checkpoint: dict) -> None:
    """
    Update model checkpoint from version 7 to version 8.

    The lambda-basis parameters (``spex_calculator``, ``spex_contraction``,
    ``spex_contraction_for_tensors``, ``center_encoding``) were moved from
    ``TensorBasis`` into a nested ``LambdaBasis`` sub-module called
    ``lambda_basis_module``.

    :param checkpoint: The checkpoint to update.
    """
    lambda_basis_attrs = (
        "spex_calculator.",
        "spex_contraction.",
        "spex_contraction_for_tensors.",
        "center_encoding.",
    )

    for sd_key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(sd_key)) is not None:
            new_state_dict = {}
            for name, value in state_dict.items():
                if "basis_calculators." in name:
                    # Find the part after basis_calculators.target.dict_key.
                    # Format: basis_calculators.{target}.{dict_key}.{attr}...
                    prefix, _, rest = name.partition("basis_calculators.")
                    parts = rest.split(".", 2)  # [target, dict_key, attr...]
                    if len(parts) == 3:
                        attr_rest = parts[2]
                        if any(attr_rest.startswith(a) for a in lambda_basis_attrs):
                            new_name = (
                                f"{prefix}basis_calculators.{parts[0]}.{parts[1]}"
                                f".lambda_basis_module.{attr_rest}"
                            )
                            new_state_dict[new_name] = value
                            continue
                new_state_dict[name] = value
            checkpoint[sd_key] = new_state_dict


###########################
# TRAINER #################
###########################


def trainer_update_v1_v2(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 1 to version 2.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["best_epoch"] = checkpoint.get("best_epoch")


def trainer_update_v2_v3(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 2 to version 3.

    :param checkpoint: The checkpoint to update.
    """
    old_loss_hypers = checkpoint["train_hypers"]["loss"].copy()
    dataset_info = checkpoint["model_data"]["dataset_info"]
    new_loss_hypers = {}

    for target_name in dataset_info.targets.keys():
        new_loss_hypers[target_name] = {
            "type": old_loss_hypers["type"],
            "weight": old_loss_hypers["weights"].get(target_name, 1.0),
            "reduction": old_loss_hypers["reduction"],
            "sliding_factor": old_loss_hypers.get("sliding_factor", None),
        }
    checkpoint["train_hypers"]["loss"] = new_loss_hypers


def trainer_update_v3_v4(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 3 to version 4.

    :param checkpoint: The checkpoint to update.
    """
    # num_workers=0 means that the main process will do the data loading, which is
    # equivalent to not setting it (this was the behavior before v4)
    checkpoint["train_hypers"]["num_workers"] = 0


def trainer_update_v4_v5(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 4 to version 5.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["train_hypers"]["fixed_scaling_weights"] = {}


def trainer_update_v5_v6(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 5 to version 6.

    :param checkpoint: The checkpoint to update.
    """
    raise ValueError(
        "In order to use this checkpoint, you need metatrain 2025.10 or earlier. "
        "You can install it with `pip install metatrain==2025.10`."
    )


def trainer_update_v6_v7(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 7 to version 8.

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


def trainer_update_v7_v8(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 7 to version 8.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["train_hypers"]["remove_composition_contribution"] = True


def trainer_update_v8_v9(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 8 to version 9.

    :param checkpoint: The checkpoint to update.
    """
    # - Remove the ``remove_composition_contribution`` hyper.
    # - Rename ``fixed_composition_weights`` to ``atomic_baseline``.
    # - If ``remove_composition_contribution`` is False, set all atomic baselines
    #   to 0.0 for all targets.
    use_atomic_baseline = checkpoint["train_hypers"].pop(
        "remove_composition_contribution"
    )
    atomic_baseline = checkpoint["train_hypers"].pop("fixed_composition_weights")

    if not use_atomic_baseline:
        # Just set
        dataset_info = checkpoint["model_data"]["dataset_info"]
        atomic_baseline = {target_name: 0.0 for target_name in dataset_info.targets}

    checkpoint["train_hypers"]["atomic_baseline"] = atomic_baseline


def trainer_update_v9_v10(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 9 to version 10.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["train_hypers"]["batch_atom_bounds"] = [None, None]
