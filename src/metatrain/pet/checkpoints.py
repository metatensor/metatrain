import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


###########################
# MODEL ###################
###########################


def model_update_v1_v2(checkpoint):
    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is not None:
            if "additive_models.0.model.type_to_index" not in state_dict:
                state_dict["additive_models.0.model.type_to_index"] = state_dict.pop(
                    "additive_models.0.type_to_index"
                )


def model_update_v2_v3(checkpoint):
    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is not None:
            if "train_hypers" in state_dict:
                finetune_config = state_dict["train_hypers"].get("finetune", {})
            else:
                finetune_config = {}
            state_dict["finetune_config"] = finetune_config


def model_update_v3_v4(checkpoint):
    checkpoint["epoch"] = checkpoint.get("epoch")
    checkpoint["best_epoch"] = checkpoint.get("best_epoch")

    if checkpoint["best_model_state_dict"] is not None:
        checkpoint["best_model_state_dict"] = checkpoint.get("best_model_state_dict")


def model_update_v4_v5(checkpoint):
    checkpoint["model_data"]["dataset_info"]._atomic_types = list(
        checkpoint["model_data"]["dataset_info"]._atomic_types
    )


def model_update_v5_v6(checkpoint):
    if not checkpoint["best_model_state_dict"]:
        checkpoint["best_model_state_dict"] = checkpoint["model_state_dict"]


def model_update_v6_v7(checkpoint):
    # this update consists in changes in the scaler
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
            for target_name, target_info in checkpoint["model_data"][
                "dataset_info"
            ].targets.items():
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
    return checkpoint


###########################
# TRAINER #################
###########################


def trainer_update_v1_v2(checkpoint):
    checkpoint["train_hypers"]["scheduler_factor"] = checkpoint["train_hypers"].get(
        "scheduler_factor", 0.5
    )


def trainer_update_v2_v3(checkpoint):
    checkpoint["best_epoch"] = checkpoint.get("best_epoch")


def trainer_update_v3_v4(checkpoint):
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


def trainer_update_v4_v5(checkpoint):
    raise ValueError(
        "In order to use this checkpoint, you need metatrain 2025.10 or earlier. "
        "You can install it with `pip install metatrain==2025.10`."
    )
