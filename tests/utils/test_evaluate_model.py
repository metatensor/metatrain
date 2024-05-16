import os
from pathlib import Path

import pytest
import torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from metatensor.models.experimental import soap_bpnn
from metatensor.models.utils.data import read_systems
from metatensor.models.utils.evaluate_model import evaluate_model
from metatensor.models.utils.io import export
from metatensor.models.utils.neighbor_lists import get_system_with_neighbor_lists


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / "resources"


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("exported", [True, False])
def test_evaluate_model(tmp_path, training, exported):
    """Test that the evaluate_model function works as intended."""

    systems = read_systems(
        RESOURCES_PATH / "alchemical_reduced_10.xyz", dtype=torch.get_default_dtype()
    )[:2]

    atomic_types = list(
        torch.unique(torch.concatenate([system.types for system in systems]))
    )

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=atomic_types,
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
        interaction_range=soap_bpnn.DEFAULT_HYPERS["model"]["soap"]["cutoff"],
        dtype="float32",
    )

    model = soap_bpnn.Model(capabilities)
    if exported:
        os.chdir(tmp_path)
        export(model, "model.pt")
        model = torch.jit.load("model.pt")
        systems = [
            get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
            for system in systems
        ]

    targets = {"energy": ["positions", "strain"]}

    outputs = evaluate_model(
        model,
        systems,
        targets,
        is_training=training,
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
