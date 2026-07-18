import metatensor.torch as mts
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.scaler.checkpoints import update_per_property_scales
from metatrain.utils.data.target_info import TargetInfo
from metatrain.utils.omegaconf import DEPRECATED_METATOMIC_TARGET_NAMES


def _rename_column_in_labels(labels: Labels, old_name: str, new_name: str) -> Labels:
    """If labels has a column called ``old_name``, return a copy where it is renamed to
    ``new_name``, otherwise return ``labels`` unchanged"""

    if old_name not in labels.names:
        return labels

    renamed_names = [new_name if name == old_name else name for name in labels.names]
    return Labels(renamed_names, labels.values)


def _rename_target_name_in_tensor_map(
    tensor_map: TensorMap, old_name: str, new_name: str
) -> TensorMap:
    """Return a copy of tensor_map in which every ``Labels`` has its ``old_name`` column
    renamed to ``new_name``"""

    renamed_blocks = []
    for block in tensor_map.blocks():
        # rewrite all Labels on the main block
        renamed_block = TensorBlock(
            values=block.values,
            samples=_rename_column_in_labels(block.samples, old_name, new_name),
            components=[
                _rename_column_in_labels(c, old_name, new_name)
                for c in block.components
            ],
            properties=_rename_column_in_labels(block.properties, old_name, new_name),
        )

        # do the same for each gradient block
        for gradient_name, gradient_block in block.gradients():
            renamed_block.add_gradient(
                parameter=gradient_name,
                gradient=TensorBlock(
                    values=gradient_block.values,
                    samples=_rename_column_in_labels(
                        gradient_block.samples, old_name, new_name
                    ),
                    components=[
                        _rename_column_in_labels(c, old_name, new_name)
                        for c in gradient_block.components
                    ],
                    properties=_rename_column_in_labels(
                        gradient_block.properties, old_name, new_name
                    ),
                ),
            )

        renamed_blocks.append(renamed_block)

    return TensorMap(keys=tensor_map.keys, blocks=renamed_blocks)


###########################
# MODEL ###################
###########################


def model_update_v1_v2(checkpoint: dict) -> None:
    """
    Update a v1 checkpoint to v2.

    :param checkpoint: The checkpoint to update.
    """
    update_per_property_scales(checkpoint)


def model_update_v2_v3(checkpoint: dict) -> None:
    """
    Update a v2 checkpoint to v3 to rename deprecated metatomic target names
    in dataset_info, train_hypers and state_dict

    :param checkpoint: The checkpoint to update.
    """
    targets = checkpoint["model_data"]["dataset_info"].targets
    loss = checkpoint.get("train_hypers", {}).get("loss")
    loss = loss if isinstance(loss, dict) else {}
    model_state_dict = checkpoint.get("model_state_dict") or {}
    best_model_state_dict = checkpoint.get("best_model_state_dict") or {}

    for old, new in DEPRECATED_METATOMIC_TARGET_NAMES.items():
        # 1. rename in dataset_info.targets
        if old in targets:
            target_info = targets.pop(old)
            targets[new] = TargetInfo(
                layout=_rename_target_name_in_tensor_map(target_info.layout, old, new),
                quantity=target_info.quantity,
                unit=target_info.unit,
                description=target_info.description,
            )

        # 2. rename in trainer loss hypers (keyed by target name)
        if old in loss:
            loss[new] = loss.pop(old)

        # 3. rename in state_dict paths
        for state_dict in (model_state_dict, best_model_state_dict):
            for key in list(state_dict.keys()):
                new_key = (
                    # heads
                    key.replace(f"node_heads.{old}.", f"node_heads.{new}.")
                    .replace(f"edge_heads.{old}.", f"edge_heads.{new}.")
                    # last layers
                    .replace(f"node_last_layers.{old}.", f"node_last_layers.{new}.")
                    .replace(f"edge_last_layers.{old}.", f"edge_last_layers.{new}.")
                    # scaler
                    .replace(f"scaler.{old}_", f"scaler.{new}_")
                    # <target>___.weight / bias
                    .replace(f".{old}___", f".{new}___")
                )
                if new_key != key:
                    value = state_dict.pop(key)
                    if key.startswith("scaler.") and key.endswith("_buffer"):
                        # buffer = serialized TensorMap; deserialize, rebuild its
                        # Labels, re-serialize
                        value = mts.save_buffer(
                            mts.make_contiguous(
                                _rename_target_name_in_tensor_map(
                                    mts.load_buffer(value), old, new
                                )
                            )
                        )
                    state_dict[new_key] = value


###########################
# TRAINER #################
###########################


def trainer_update_v1_v2(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 1 to version 2.

    The ``batch_atom_bounds`` field has been removed from the trainer schema
    (max-atom packing is now done by the sampler, via ``max_atoms_per_batch``
    and ``min_atoms_per_batch``). ``batch_atom_bounds`` bounds are translated
    into the equivalent sampler settings; if it was unset, the new sampler
    settings default to no packing.

    :param checkpoint: The checkpoint to update.
    """
    train_hypers = checkpoint["train_hypers"]
    min_bound, max_bound = train_hypers.pop("batch_atom_bounds", [None, None])
    train_hypers["max_atoms_per_batch"] = max_bound
    train_hypers["min_atoms_per_batch"] = min_bound if min_bound is not None else 0
