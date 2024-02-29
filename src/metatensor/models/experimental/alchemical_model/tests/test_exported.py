import pytest
import torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from metatensor.models.experimental.alchemical_model import DEFAULT_HYPERS, Model
from metatensor.models.utils.export import export
from metatensor.models.utils.model_io import load_exported_model


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_to(tmp_path, device, dtype):
    """Tests that the `.to()` method of the exported model works."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    with tmp_path.as_cwd():
        capabilities = ModelCapabilities(
            length_unit="Angstrom",
            species=[1, 6, 7, 8],
            outputs={
                "energy": ModelOutput(
                    quantity="energy",
                    unit="eV",
                )
            },
        )
        model = Model(capabilities, DEFAULT_HYPERS["model"])
        export(model, "model.pt")
        exported = load_exported_model("model.pt")

        exported.to(device=device, dtype=dtype)
