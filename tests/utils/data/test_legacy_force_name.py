import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.pet.checkpoints import model_update_v16_v17
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.legacy_names import (
    LEGACY_NON_CONSERVATIVE_FORCE,
    upgrade_legacy_non_conservative_force,
)
from metatrain.utils.data.target_info import TargetInfo


def _force_target(property_name: str) -> TargetInfo:
    block = TensorBlock(
        values=torch.empty((0, 3, 1), dtype=torch.float64),
        samples=Labels(["system", "atom"], torch.empty((0, 2), dtype=torch.int32)),
        components=[Labels(["xyz"], torch.arange(3, dtype=torch.int32).reshape(-1, 1))],
        properties=Labels([property_name], torch.zeros((1, 1), dtype=torch.int32)),
    )
    return TargetInfo(
        layout=TensorMap(keys=Labels.single(), blocks=[block]),
        quantity="force",
        unit="eV/A",
    )


def _scaler_buffer(property_name: str) -> torch.Tensor:
    tensor_map = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.ones((1, 1), dtype=torch.float64),
                samples=Labels(["atomic_type"], torch.tensor([[1]])),
                components=[],
                properties=Labels(
                    [property_name], torch.zeros((1, 1), dtype=torch.int32)
                ),
            )
        ],
    )
    return mts.save_buffer(mts.make_contiguous(tensor_map))


def test_upgrade_renames_property_label_and_scaler_buffer():
    """PET-MAD checkpoints keep the plural property name under the singular key."""
    checkpoint = {
        "model_data": {
            "dataset_info": DatasetInfo(
                length_unit="angstrom",
                atomic_types=[1],
                targets={
                    "non_conservative_force": _force_target(
                        LEGACY_NON_CONSERVATIVE_FORCE
                    ),
                },
            ),
        },
        "model_state_dict": {
            "scaler.non_conservative_force_per_target_scaler_buffer": _scaler_buffer(
                LEGACY_NON_CONSERVATIVE_FORCE
            ),
            "scaler.non_conservative_force_per_property_scaler_buffer": _scaler_buffer(
                LEGACY_NON_CONSERVATIVE_FORCE
            ),
            "scaler.dummy_buffer": torch.zeros(1),
        },
        "best_model_state_dict": {},
    }
    upgrade_legacy_non_conservative_force(checkpoint)

    target = checkpoint["model_data"]["dataset_info"].targets["non_conservative_force"]
    assert target.layout.block().properties.names == ["non_conservative_force"]
    state_dict = checkpoint["model_state_dict"]
    for key in (
        "scaler.non_conservative_force_per_target_scaler_buffer",
        "scaler.non_conservative_force_per_property_scaler_buffer",
    ):
        loaded = mts.load_buffer(state_dict[key])
        assert loaded.block().properties.names == ["non_conservative_force"]
    assert torch.equal(state_dict["scaler.dummy_buffer"], torch.zeros(1))


def test_upgrade_renames_legacy_target_key_and_state_dict_path():
    weight = torch.zeros(1)
    checkpoint = {
        "model_data": {
            "dataset_info": DatasetInfo(
                length_unit="angstrom",
                atomic_types=[1],
                targets={
                    LEGACY_NON_CONSERVATIVE_FORCE: _force_target(
                        LEGACY_NON_CONSERVATIVE_FORCE
                    ),
                },
            ),
        },
        "train_hypers": {
            "loss": {LEGACY_NON_CONSERVATIVE_FORCE: {"weight": 1.0}},
            "per_structure_targets": [],
        },
        "model_state_dict": {
            "backend.node_heads.non_conservative_forces.0.weight": weight,
        },
        "best_model_state_dict": None,
    }
    model_update_v16_v17(checkpoint)

    targets = checkpoint["model_data"]["dataset_info"].targets
    assert LEGACY_NON_CONSERVATIVE_FORCE not in targets
    assert targets["non_conservative_force"].layout.block().properties.names == [
        "non_conservative_force"
    ]
    assert "non_conservative_force" in checkpoint["train_hypers"]["loss"]
    assert (
        "backend.node_heads.non_conservative_force.0.weight"
        in checkpoint["model_state_dict"]
    )


def test_upgrade_leaves_singular_checkpoints_unchanged():
    checkpoint = {
        "model_data": {
            "dataset_info": DatasetInfo(
                length_unit="angstrom",
                atomic_types=[1],
                targets={"energy": _force_target("energy")},
            ),
        },
        "model_state_dict": {"scaler.dummy_buffer": torch.zeros(1)},
        "best_model_state_dict": None,
    }
    upgrade_legacy_non_conservative_force(checkpoint)
    assert checkpoint["model_data"]["dataset_info"].targets[
        "energy"
    ].layout.block().properties.names == ["energy"]
