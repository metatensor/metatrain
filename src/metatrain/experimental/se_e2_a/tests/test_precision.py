import copy

import torch

from metatrain.experimental.se_e2_a import SeE2A
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info

from . import MODEL_HYPERS
from .test_basic import _minimal_hypers


def _make_dataset_info():
    targets = {
        "mtt::U0": get_energy_target_info(
            "mtt::U0", {"quantity": "energy", "unit": "eV"}
        )
    }
    return DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=targets
    )


def test_default_precision_is_float32():
    """The documented default builds a float32 descriptor."""
    assert MODEL_HYPERS["descriptor"]["precision"] == 32
    assert MODEL_HYPERS["fitting_net"]["precision"] == 32
    hypers = _minimal_hypers("experimental.se_e2_a")
    model = SeE2A(hypers, _make_dataset_info())
    assert model.dtype == torch.float32


def test_descriptor_precision_float64():
    """``descriptor.precision = 64`` selects the model dtype."""
    hypers = _minimal_hypers("experimental.se_e2_a")
    hypers["descriptor"]["precision"] = 64
    model = SeE2A(hypers, _make_dataset_info())
    assert model.dtype == torch.float64


def test_mixed_precision_descriptor_fitting():
    """The fitting network may use a different precision from the descriptor."""
    hypers = copy.deepcopy(_minimal_hypers("experimental.se_e2_a"))
    hypers["descriptor"]["precision"] = 64
    hypers["fitting_net"]["precision"] = 32
    model = SeE2A(hypers, _make_dataset_info())
    assert model.dtype == torch.float64
