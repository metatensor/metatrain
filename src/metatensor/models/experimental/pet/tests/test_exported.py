import pytest
import torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from metatensor.models.experimental.pet import DEFAULT_HYPERS, Model
from metatensor.models.utils.export import export
from metatensor.models.utils.model_io import load_exported_model


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_to(monkeypatch, tmp_path, device, dtype):
    """Tests that the `.to()` method of the exported model works."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    monkeypatch.chdir(tmp_path)

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
    pet = Model(capabilities, DEFAULT_HYPERS["ARCHITECTURAL_HYPERS"])
    export(pet, "pet.pt")
    exported = load_exported_model("pet.pt")

    exported.to(device=device, dtype=dtype)
