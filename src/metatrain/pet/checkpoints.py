import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


###########################
# MODEL ###################
###########################


def model_update_v1_v2(checkpoint: dict) -> None:
    """
    Update a v1 checkpoint to v2.

    :param checkpoint: The checkpoint to update.
    """
    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is not None:
            if "additive_models.0.model.type_to_index" not in state_dict:
                state_dict["additive_models.0.model.type_to_index"] = state_dict.pop(
                    "additive_models.0.type_to_index"
                )


def model_update_v2_v3(checkpoint: dict) -> None:
    """
    Update a v2 checkpoint to v3.

    :param checkpoint: The checkpoint to update.
    """
    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is not None:
            if "train_hypers" in checkpoint:
                finetune_config = checkpoint["train_hypers"].get("finetune", {})
            else:
                finetune_config = {}
            state_dict["finetune_config"] = finetune_config


def model_update_v3_v4(checkpoint: dict) -> None:
    """
    Update a v3 checkpoint to v4.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["epoch"] = checkpoint.get("epoch")
    checkpoint["best_epoch"] = checkpoint.get("best_epoch")

    if checkpoint["best_model_state_dict"] is not None:
        checkpoint["best_model_state_dict"] = checkpoint.get("best_model_state_dict")


def model_update_v4_v5(checkpoint: dict) -> None:
    """
    Update a v4 checkpoint to v5.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["model_data"]["dataset_info"]._atomic_types = list(
        checkpoint["model_data"]["dataset_info"]._atomic_types
    )


def model_update_v5_v6(checkpoint: dict) -> None:
    """
    Update a v5 checkpoint to v6.

    :param checkpoint: The checkpoint to update.
    """
    if not checkpoint["best_model_state_dict"]:
        checkpoint["best_model_state_dict"] = checkpoint["model_state_dict"]


def model_update_v6_v7(checkpoint: dict) -> None:
    """
    Update model checkpoint from version 6 to version 7.

    :param checkpoint: The checkpoint to be updated.
    """
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


def model_update_v7_v8(checkpoint: dict) -> None:
    """
    Update a v7 checkpoint to v8.

    :param checkpoint: The checkpoint to update.
    """
    if "node_embedding.weight" in checkpoint["model_state_dict"]:
        ##############################################
        # **Updating the large-scale PET checkpoints**
        ##############################################

        checkpoint["model_data"]["model_hypers"]["normalization"] = "RMSNorm"
        checkpoint["model_data"]["model_hypers"]["activation"] = "SwiGLU"
        checkpoint["model_data"]["model_hypers"]["d_node"] = (
            4 * checkpoint["model_data"]["model_hypers"]["d_pet"]
        )
        checkpoint["model_data"]["model_hypers"]["transformer_type"] = "PreLN"
        checkpoint["model_data"]["model_hypers"]["featurizer_type"] = "feedforward"
        checkpoint["model_data"]["model_hypers"]["d_feedforward"] = (
            2 * checkpoint["model_data"]["model_hypers"]["d_pet"]
        )
        if (state_dict := checkpoint.get("model_state_dict")) is not None:
            new_state_dict = {}
            for k, v in state_dict.items():
                if "embedding." in k and "node" not in k:
                    k = k.replace("embedding.", "edge_embedder.")
                    v = v[:-1, :]  # removing the embedding for padding +1 type
                if "node_embedding." in k:
                    k = k.replace("node_embedding.", "node_embedders.0.")
                    v = v[:-1, :]  # removing the embedding for padding +1 type
                if ".neighbor_embedder." in k:
                    v = v[:-1, :]  # removing the embedding for padding +1 type
                if "combination_rmsnorms" in k:
                    k = k.replace("combination_rmsnorms", "combination_norms")
                new_state_dict[k] = v
            checkpoint["model_state_dict"] = new_state_dict

        # for the large-scale checkpoints, the model for evaluation is always
        # taken to be the last
        checkpoint["best_model_state_dict"] = checkpoint["model_state_dict"]

    else:
        ###############################################
        # **Updating the standard old PET checkpoints**
        ###############################################

        # Adding the option for choosing the normalization type
        if "normalization" not in checkpoint["model_data"]["model_hypers"]:
            checkpoint["model_data"]["model_hypers"]["normalization"] = "LayerNorm"
        # Adding the option for choosing the activation function
        if "activation" not in checkpoint["model_data"]["model_hypers"]:
            checkpoint["model_data"]["model_hypers"]["activation"] = "SiLU"
        # Setting the node features dimension to be the same as d_pet if not specified
        if "d_node" not in checkpoint["model_data"]["model_hypers"]:
            checkpoint["model_data"]["model_hypers"]["d_node"] = checkpoint[
                "model_data"
            ]["model_hypers"]["d_pet"]
        # Setting the default transformer type to PostLN if not specified
        if "transformer_type" not in checkpoint["model_data"]["model_hypers"]:
            checkpoint["model_data"]["model_hypers"]["transformer_type"] = "PostLN"
        if "featurizer_type" not in checkpoint["model_data"]["model_hypers"]:
            checkpoint["model_data"]["model_hypers"]["featurizer_type"] = "residual"
        for key in ["model_state_dict", "best_model_state_dict"]:
            if (state_dict := checkpoint.get(key)) is not None:
                new_state_dict = {}
                for k, v in state_dict.items():
                    # Replacing the nn.Sequential MLP with a custom FeedForward module
                    if ".mlp.0" in k:
                        k = k.replace(".mlp.0", ".mlp.w_in")
                    if ".mlp.3" in k:
                        k = k.replace(".mlp.3", ".mlp.w_out")
                    # Moving the node embedder to a top-level PET model attribute
                    if "embedding." in k:
                        k = k.replace("embedding.", "edge_embedder.")
                        v = v[:-1, :]  # removing the embedding for padding +1 type
                    if ".node_embedder." in k:
                        key_content = k.split(".")
                        k = ".".join(
                            ["node_embedders", key_content[1], key_content[-1]]
                        )
                        v = v[:-1, :]  # removing the embedding for padding +1 type
                    if ".neighbor_embedder." in k:
                        v = v[:-1, :]  # removing the embedding for padding +1 type
                    new_state_dict[k] = v
                checkpoint[key] = new_state_dict


###########################
# TRAINER #################
###########################


def trainer_update_v1_v2(checkpoint: dict) -> None:
    """
    Update a v1 Trainer checkpoint to v2.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["train_hypers"]["scheduler_factor"] = checkpoint["train_hypers"].get(
        "scheduler_factor", 0.5
    )


def trainer_update_v2_v3(checkpoint: dict) -> None:
    """
    Update a v2 Trainer checkpoint to v3.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["best_epoch"] = checkpoint.get("best_epoch")


def trainer_update_v3_v4(checkpoint: dict) -> None:
    """
    Update a v3 Trainer checkpoint to v4.

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


def trainer_update_v4_v5(checkpoint: dict) -> None:
    """
    Update a v4 Trainer checkpoint to v5.

    :param checkpoint: The checkpoint to update.
    """
    raise ValueError(
        "In order to use this checkpoint, you need metatrain 2025.10 or earlier. "
        "You can install it with `pip install metatrain==2025.10`."
    )


def trainer_update_v5_v6(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 5 to version 6.

    :param checkpoint: The checkpoint to update.
    """
    # num_workers=0 means that the main process will do the data loading, which is
    # equivalent to not setting it (this was the behavior before v6)
    checkpoint["train_hypers"]["num_workers"] = 0


def trainer_update_v6_v7(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 6 to version 7.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["train_hypers"]["fixed_scaling_weights"] = {}


def trainer_update_v7_v8(checkpoint: dict) -> None:
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


def trainer_update_v8_v9(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 7 to version 8.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["train_hypers"]["remove_composition_contribution"] = True
