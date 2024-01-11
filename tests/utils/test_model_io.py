from pathlib import Path

import metatensor.torch
import rascaline.torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from metatensor.models import soap_bpnn
from metatensor.models.utils.data import read_structures
from metatensor.models.utils.model_io import load_model, save_model


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / "resources"


def test_save_load_model(monkeypatch, tmp_path):
    """Test that saving and loading a model works and preserves its internal state."""
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

    model = soap_bpnn.Model(capabilities)
    structures = read_structures(RESOURCES_PATH / "qm9_reduced_100.xyz")

    output_before_save = model(rascaline.torch.systems_to_torch(structures))

    save_model(model, "test_model.pt")
    loaded_model = load_model("test_model.pt")

    output_after_load = loaded_model(rascaline.torch.systems_to_torch(structures))

    assert metatensor.torch.allclose(
        output_before_save["energy"], output_after_load["energy"]
    )
