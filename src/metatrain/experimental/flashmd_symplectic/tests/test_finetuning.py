import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.experimental.flashmd_symplectic import FlashMDSymplectic
from metatrain.pet.modules.finetuning import apply_finetuning_strategy
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import TargetInfo

from . import MODEL_HYPERS


def _get_dataset_info():
    return DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 6],
        targets={
            name: TargetInfo(
                layout=TensorMap(
                    keys=Labels.single(),
                    blocks=[
                        TensorBlock(
                            values=torch.empty((0, 3, 1), dtype=torch.float64),
                            samples=Labels(
                                names=["system", "atom"],
                                values=torch.empty((0, 2), dtype=int),
                            ),
                            components=[
                                Labels.range("xyz", 3),
                            ],
                            properties=Labels.range("length", 1),
                        )
                    ],
                ),
                quantity="length",
                unit="angstrom",
            )
            for name in ["position", "momentum"]
        },
    )


def test_lora_finetuning_functionality():
    model = FlashMDSymplectic(MODEL_HYPERS, _get_dataset_info())

    finetuning_strategy = {
        "read_from": None,
        "method": "lora",
        "config": {
            "target_modules": ["input_linear", "output_linear"],
            "rank": 4,
            "alpha": 8,
        },
        "inherit_heads": {},
    }

    model = apply_finetuning_strategy(model, finetuning_strategy)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in model.parameters())
    assert num_trainable_params < num_params


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_lora_finetuning_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    model = FlashMDSymplectic(MODEL_HYPERS, _get_dataset_info()).to(device)

    finetuning_strategy = {
        "read_from": None,
        "method": "lora",
        "config": {
            "target_modules": ["input_linear", "output_linear"],
            "rank": 4,
            "alpha": 8,
        },
        "inherit_heads": {},
    }

    model = apply_finetuning_strategy(model, finetuning_strategy)
    for param in model.parameters():
        assert param.device.type == device, f"Parameter {param.name} is not on {device}"


def test_heads_finetuning_functionality():
    """Regression test: FlashMDSymplectic's heads/last_layers live directly on
    the model (unlike PET, there is no ``backend`` submodule), so the
    ``head_modules`` and ``last_layer_modules`` prefixes must match
    FlashMDSymplectic's own parameter names."""
    model = FlashMDSymplectic(MODEL_HYPERS, _get_dataset_info())

    finetuning_strategy = {
        "read_from": None,
        "method": "heads",
        "config": {
            "head_modules": ["node_heads", "edge_heads"],
            "last_layer_modules": ["node_last_layers", "edge_last_layers"],
        },
        "inherit_heads": {},
    }

    model = apply_finetuning_strategy(model, finetuning_strategy)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in model.parameters())
    assert 0 < num_trainable_params < num_params


def test_heads_finetuning_unknown_modules():
    """Unknown 'head_modules'/'last_layer_modules' should raise, not silently
    freeze the whole model."""
    model = FlashMDSymplectic(MODEL_HYPERS, _get_dataset_info())

    finetuning_strategy = {
        "read_from": None,
        "method": "heads",
        "config": {
            "head_modules": ["does_not_exist"],
            "last_layer_modules": ["also_does_not_exist"],
        },
        "inherit_heads": {},
    }

    with pytest.raises(ValueError, match="No parameters were found matching"):
        apply_finetuning_strategy(model, finetuning_strategy)


def _get_single_target_dataset_info(target_name):
    return DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 6],
        targets={
            target_name: TargetInfo(
                layout=TensorMap(
                    keys=Labels.single(),
                    blocks=[
                        TensorBlock(
                            values=torch.empty((0, 3, 1), dtype=torch.float64),
                            samples=Labels(
                                names=["system", "atom"],
                                values=torch.empty((0, 2), dtype=int),
                            ),
                            components=[
                                Labels.range("xyz", 3),
                            ],
                            properties=Labels.range("length", 1),
                        )
                    ],
                ),
                quantity="length",
                unit="angstrom",
            )
        },
    )


def _finetuning_strategy(method):
    if method == "lora":
        config = {
            "target_modules": ["input_linear", "output_linear"],
            "rank": 4,
            "alpha": 8,
        }
    elif method == "heads":
        config = {
            "head_modules": ["node_heads", "edge_heads"],
            "last_layer_modules": ["node_last_layers", "edge_last_layers"],
        }
    else:
        config = {}
    return {"read_from": None, "method": method, "config": config, "inherit_heads": {}}


@pytest.mark.parametrize("method", ["full", "lora"])
def test_finetune_full_lora_prunes_stale_targets(method):
    """A target not part of the current full/lora finetuning run's dataset (here
    ``"position"``) is dropped from the model, since its head is no longer
    compatible with the fine-tuned backbone.

    Removal only happens once ``apply_finetuning_strategy`` runs (as it would when
    training actually starts): ``restart`` alone must not remove it yet, since
    ``inherit_heads`` (applied within ``apply_finetuning_strategy``) may still need
    to copy weights from the stale target's head."""
    model = FlashMDSymplectic(MODEL_HYPERS, _get_dataset_info())
    new_dataset_info = _get_single_target_dataset_info("momentum")

    model.restart(new_dataset_info, finetune_method=method)
    assert "position" in model.node_heads

    apply_finetuning_strategy(model, _finetuning_strategy(method))

    assert "position" not in model.dataset_info.targets
    assert "position" not in model.supported_outputs()
    assert "position" not in model.node_heads
    assert "position" not in model.edge_heads
    assert "position" not in model.node_last_layers
    assert "position" not in model.edge_last_layers
    assert "momentum" in model.dataset_info.targets
    assert "momentum" in model.node_heads


def test_finetune_heads_keeps_stale_targets():
    """With heads-only finetuning, the backbone is unchanged, so a target not part
    of the current run's dataset must be kept."""
    model = FlashMDSymplectic(MODEL_HYPERS, _get_dataset_info())
    new_dataset_info = _get_single_target_dataset_info("momentum")

    model.restart(new_dataset_info, finetune_method="heads")
    apply_finetuning_strategy(model, _finetuning_strategy("heads"))

    assert "position" in model.dataset_info.targets
    assert "position" in model.node_heads
    assert "momentum" in model.dataset_info.targets
    assert "momentum" in model.node_heads


def test_restart_without_finetune_method_keeps_stale_targets():
    """A plain restart (not part of a finetuning run) must not prune any target."""
    model = FlashMDSymplectic(MODEL_HYPERS, _get_dataset_info())
    new_dataset_info = _get_single_target_dataset_info("momentum")

    model.restart(new_dataset_info)

    assert "position" in model.dataset_info.targets
    assert "position" in model.node_heads
    assert "momentum" in model.dataset_info.targets
    assert "momentum" in model.node_heads


def _get_three_target_dataset_info():
    return DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 6],
        targets={
            **_get_dataset_info().targets,  # "position", "momentum"
            **_get_single_target_dataset_info("mtt::extra").targets,
        },
    )


def test_restart_with_fewer_targets_does_not_crash():
    """Continuing training (plain restart, no ``finetune_method``) from a model
    that has 3 targets, specifying only 2 of them, must not raise -- the 3rd
    target's head is kept in the model, untouched, rather than being required to
    be part of every subsequent run."""
    model = FlashMDSymplectic(MODEL_HYPERS, _get_three_target_dataset_info())
    reduced_dataset_info = _get_dataset_info()  # only "position", "momentum"

    model.restart(reduced_dataset_info)

    assert "mtt::extra" in model.dataset_info.targets
    assert "mtt::extra" in model.node_heads
    assert "position" in model.dataset_info.targets
    assert "momentum" in model.dataset_info.targets
