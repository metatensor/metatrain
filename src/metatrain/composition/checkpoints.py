import torch


def model_update_v1_v2(checkpoint: dict, prefix: str = "") -> None:
    """
    Update model checkpoint from version 1 to version 2.

    The model now uses metatensor's ``nn.Module`` and ``register_buffer`` instead of
    manually tracking the data.

    :param checkpoint: The checkpoint to update.
    :param prefix: Prefix prepended to the state_dict keys of the composition model
        (e.g. ``"additive_models.0."`` when the composition model is nested inside
        another architecture's ``additive_models[0]``).
    """
    scaler_key = f"{prefix}model"

    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is None:
            continue

        # If both model_state_dict and best_model_state_dict point to the same
        # dict, the upgrade was already applied in the first iteration.
        if f"{scaler_key}._mts_helper" in state_dict:
            continue

        dummy_buffer = state_dict[f"{prefix}dummy_buffer"]
        empty_tensor = torch.zeros(
            0, dtype=dummy_buffer.dtype, device=dummy_buffer.device
        )

        extra_state: dict[str, dict] = {"weights": {}}

        for target_name in checkpoint["model_data"]["dataset_info"].targets:
            buffer_key = f"{prefix}{target_name}_composition_buffer"
            if buffer_key not in state_dict:
                continue

            extra_state["weights"][target_name] = (
                "metatensor.TensorMap",
                state_dict.pop(buffer_key),
                empty_tensor,
            )

        state_dict[f"{scaler_key}._mts_helper"] = empty_tensor
        state_dict[f"{scaler_key}._extra_state"] = extra_state


def trainer_update_v1_v2(checkpoint: dict) -> None:
    """
    Update a v1 Trainer checkpoint to v2.

    v2 added the ``distributed``, ``distributed_port`` and ``num_workers``
    training hypers. A missing ``distributed`` or ``num_workers`` means
    automatic behavior, so only the port needs a value.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["train_hypers"]["distributed_port"] = checkpoint["train_hypers"].get(
        "distributed_port", 39591
    )
