from pathlib import Path

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from metatensor.models.experimental import soap_bpnn
from metatensor.models.utils.compute_loss import compute_model_loss
from metatensor.models.utils.data import read_structures
from metatensor.models.utils.loss import TensorMapDictLoss


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / "resources"


def test_compute_model_loss():
    """Test that the model loss is computed."""

    loss_fn = TensorMapDictLoss(
        weights={
            "energy": {"values": 1.0, "positions": 10.0},
        }
    )

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        species=[21, 23, 24, 27, 29, 39, 40, 41, 72, 74, 78],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
    )

    model = soap_bpnn.Model(capabilities)
    # model = torch.jit.script(model)  # jit the model for good measure

    structures = read_structures(RESOURCES_PATH / "alchemical_reduced_10.xyz")[:2]

    gradient_samples = Labels(
        names=["sample", "atom"],
        values=torch.stack(
            [
                torch.concatenate(
                    [
                        torch.tensor([i] * len(structure))
                        for i, structure in enumerate(structures)
                    ]
                ),
                torch.concatenate(
                    [torch.arange(len(structure)) for structure in structures]
                ),
            ],
            dim=1,
        ),
    )

    gradient_components = [
        Labels(
            names=["xyz"],
            values=torch.tensor([[0], [1], [2]]),
        )
    ]

    block = TensorBlock(
        values=torch.tensor([[0.0] * len(structures)]).T,
        samples=Labels.range("structure", len(structures)),
        components=[],
        properties=Labels.single(),
    )

    block.add_gradient(
        "positions",
        TensorBlock(
            values=torch.tensor(
                [
                    [[1.0], [1.0], [1.0]]
                    for structure in structures
                    for _ in range(len(structure.positions))
                ]
            ),
            samples=gradient_samples,
            components=gradient_components,
            properties=Labels.single(),
        ),
    )

    targets = {
        "energy": TensorMap(
            keys=Labels(
                names=["o3_lambda", "o3_sigma"],
                values=torch.tensor([[0, 1]]),
            ),
            blocks=[block],
        ),
    }

    compute_model_loss(
        loss_fn,
        model,
        structures,
        targets,
    )
