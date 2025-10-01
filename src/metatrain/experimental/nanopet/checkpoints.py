import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


# ===== Model checkpoint updates =====


def model_update_v1_v2(checkpoint):
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
                "additive_models.0.type_to_index"
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


# ===== Trainer checkpoint updates =====


def trainer_update_v1_v2(checkpoint):
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


def trainer_update_v2_v3(checkpoint):
    checkpoint["train_hypers"]["fixed_scaling_weights"] = {}
