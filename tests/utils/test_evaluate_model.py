import pytest
import torch
from metatensor.torch.atomistic import ModelCapabilities

from metatrain.experimental.soap_bpnn import __model__
from metatrain.utils.data import DatasetInfo, TargetInfo, read_systems
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.export import export
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS, RESOURCES_PATH


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("exported", [True, False])
def test_evaluate_model(training, exported):
    """Test that the evaluate_model function works as intended."""

    systems = read_systems(RESOURCES_PATH / "alchemical_reduced_10.xyz")[:2]

    atomic_types = set(
        torch.unique(torch.concatenate([system.types for system in systems]))
    )

    targets = {
        "energy": TargetInfo(
            quantity="energy", unit="eV", gradients=["positions", "strain"]
        )
    }

    dataset_info = DatasetInfo(
        length_unit="angstrom", atomic_types=atomic_types, targets=targets
    )
    model = __model__(model_hypers=MODEL_HYPERS, dataset_info=dataset_info)

    if exported:

        capabilities = ModelCapabilities(
            length_unit=model.dataset_info.length_unit,
            outputs=model.outputs,
            atomic_types=list(model.dataset_info.atomic_types),
            supported_devices=model.__supported_devices__,
            interaction_range=model.hypers["soap"]["cutoff"],
            dtype="float32",
        )

        model = export(model, capabilities)
        systems = [
            get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
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
