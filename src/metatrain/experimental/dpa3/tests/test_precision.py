import copy

import torch

from metatrain.experimental.dpa3 import DPA3
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info

from . import MODEL_HYPERS


def _make_dataset_info():
    targets = {"mtt::U0": get_energy_target_info({"quantity": "energy", "unit": "eV"})}
    return DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=targets
    )


def test_default_precision_is_float32():
    """Model defaults to float32 descriptor and fitting_net precision."""
    dataset_info = _make_dataset_info()
    model = DPA3(MODEL_HYPERS, dataset_info)
    assert model.dtype == torch.float32


def test_descriptor_precision_float64():
    """Setting descriptor.precision to float64 changes model.dtype."""
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["descriptor"]["precision"] = "float64"
    dataset_info = _make_dataset_info()
    model = DPA3(hypers, dataset_info)
    assert model.dtype == torch.float64


def test_mixed_precision_descriptor_fitting():
    """Descriptor and fitting_net can have different precision strings."""
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["descriptor"]["precision"] = "float64"
    hypers["fitting_net"]["precision"] = "float32"
    dataset_info = _make_dataset_info()
    # Should not raise -- mixed precision is allowed
    model = DPA3(hypers, dataset_info)
    # Model dtype follows the descriptor
    assert model.dtype == torch.float64
