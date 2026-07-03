import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.experimental.flashmd_symplectic import FlashMDSymplectic
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import TargetInfo
from metatrain.utils.finetuning import apply_finetuning_strategy

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
