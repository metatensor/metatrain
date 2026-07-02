import torch
from metatomic.torch import ModelMetadata

from metatrain.utils.data.readers import read_systems

from . import DATASET_PATH
from .test_regression import _make_synthetic_targets, _train_composition_model


torch.set_default_dtype(torch.float64)


def test_export():
    """Test that export() produces a valid AtomisticModel."""
    systems = read_systems(DATASET_PATH)
    per_species_energies = {1: -0.5, 6: -10.0, 7: -15.0, 8: -20.0}
    target_values = _make_synthetic_targets(systems, per_species_energies)

    model, dataset_info = _train_composition_model(systems, target_values)

    exported = model.export(metadata=ModelMetadata(name="test_composition"))

    assert exported.capabilities().atomic_types == [1, 6, 7, 8]
    assert "energy" in exported.capabilities().outputs
    assert "test" in str(exported.metadata())
    assert exported.capabilities().dtype == "float64"
