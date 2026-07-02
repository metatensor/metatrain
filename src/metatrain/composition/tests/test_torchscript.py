import pytest
import torch

from metatrain.utils.data.readers import read_systems

from . import DATASET_PATH
from .test_regression import _make_synthetic_targets, _train_composition_model


pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


torch.set_default_dtype(torch.float64)


def test_torchscript():
    """Test that the model can be jitted after training."""
    systems = read_systems(DATASET_PATH)
    per_species_energies = {1: -0.5, 6: -10.0, 7: -15.0, 8: -20.0}
    target_values = _make_synthetic_targets(systems, per_species_energies)

    model, _ = _train_composition_model(systems, target_values)
    model.eval()

    ref_output = model(systems[:5], {"energy": model.outputs["energy"]})

    scripted = torch.jit.script(model)
    scripted_output = scripted(systems[:5], {"energy": model.outputs["energy"]})

    assert torch.allclose(
        ref_output["energy"].block().values,
        scripted_output["energy"].block().values,
    )


def test_torchscript_save_load(tmpdir):
    """Test that the jitted model can be saved and loaded."""
    systems = read_systems(DATASET_PATH)
    per_species_energies = {1: -0.5, 6: -10.0, 7: -15.0, 8: -20.0}
    target_values = _make_synthetic_targets(systems, per_species_energies)

    model, _ = _train_composition_model(systems, target_values)
    model.eval()

    ref_output = model(systems[:5], {"energy": model.outputs["energy"]})

    with tmpdir.as_cwd():
        torch.jit.save(torch.jit.script(model), "composition.pt")
        loaded = torch.jit.load("composition.pt")
        loaded_output = loaded(systems[:5], {"energy": model.outputs["energy"]})

    assert torch.allclose(
        ref_output["energy"].block().values,
        loaded_output["energy"].block().values,
    )
