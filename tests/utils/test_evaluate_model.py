from pathlib import Path

import pytest
import torch
from metatensor.torch.atomistic import ModelCapabilities

from metatensor.models.experimental.soap_bpnn import __model__
from metatensor.models.utils.architectures import get_default_hypers
from metatensor.models.utils.data import DatasetInfo, TargetInfo, read_systems
from metatensor.models.utils.evaluate_model import evaluate_model
from metatensor.models.utils.export import export
from metatensor.models.utils.neighbor_lists import get_system_with_neighbor_lists


MODEL_HYPERS = get_default_hypers("experimental.soap_bpnn")["model"]
RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / "resources"


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("exported", [True, False])
def test_evaluate_model(training, exported):
    """Test that the evaluate_model function works as intended."""

    systems = read_systems(
        RESOURCES_PATH / "alchemical_reduced_10.xyz", dtype=torch.get_default_dtype()
    )[:2]

    all_types = list(
        torch.unique(torch.concatenate([system.types for system in systems]))
    )

    targets = {
        "energy": TargetInfo(quantity="energy", gradients=["positions", "strain"])
    }

    dataset_info = DatasetInfo(
        length_unit="angstrom", all_types=all_types, targets=targets
    )
    model = __model__(model_hypers=MODEL_HYPERS, dataset_info=dataset_info)

    if exported:

        capabilities = ModelCapabilities(
            length_unit=model.dataset_info.length_unit,
            outputs=model.outputs,
            atomic_types=model.all_types,
            supported_devices=model.__supported_devices__,
            interaction_range=model.hypers["soap"]["cutoff"],
            dtype="float32",
        )

        model = export(model, capabilities)
        systems = [
            get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
            for system in systems
        ]

    outputs = evaluate_model(model, systems, targets, is_training=training)

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
