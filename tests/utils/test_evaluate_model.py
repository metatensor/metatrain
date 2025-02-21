import pytest
import torch

from metatrain.soap_bpnn import __model__
from metatrain.utils.data import DatasetInfo, read_systems
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from . import MODEL_HYPERS, RESOURCES_PATH


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("exported", [True, False])
def test_evaluate_model(training, exported):
    """Test that the evaluate_model function works as intended."""

    systems = read_systems(RESOURCES_PATH / "carbon_reduced_100.xyz")[:2]

    atomic_types = set(
        torch.unique(torch.concatenate([system.types for system in systems]))
    )

    targets = {
        "energy": get_energy_target_info(
            {"unit": "eV"},
            add_position_gradients=True,
            add_strain_gradients=True,
        )
    }

    dataset_info = DatasetInfo(
        length_unit="angstrom", atomic_types=atomic_types, targets=targets
    )
    model = __model__(model_hypers=MODEL_HYPERS, dataset_info=dataset_info)

    if exported:
        model = model.export()
        requested_neighbor_lists = get_requested_neighbor_lists(model)
        systems = [
            get_system_with_neighbor_lists(system, requested_neighbor_lists)
            for system in systems
        ]

    systems = [system.to(torch.float32) for system in systems]
    outputs = evaluate_model(
        model, systems, targets, is_training=training, check_consistency=True
    )

    assert isinstance(outputs, dict)
    assert "energy" in outputs
    assert "positions" in outputs["energy"].block().gradients_list()
    assert "strain" in outputs["energy"].block().gradients_list()

    if training:
        assert outputs["energy"].block().gradient("positions").values.requires_grad
        assert outputs["energy"].block().gradient("strain").values.requires_grad
    else:
        assert not outputs["energy"].block().gradient("positions").values.requires_grad
        assert not outputs["energy"].block().gradient("strain").values.requires_grad
