import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from metatrain.utils.system_data import get_system_data_transform


def _system() -> System:
    return System(
        positions=torch.tensor([[0.0, 0.0, 0.0]]),
        types=torch.tensor([1]),
        cell=torch.zeros((3, 3)),
        pbc=torch.tensor([False, False, False]),
    )


def _extra(key: str) -> TensorMap:
    return TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.tensor([[1.0]]),
                samples=Labels("system", torch.tensor([[0]])),
                components=[],
                properties=Labels.range("ignored", 1),
            )
        ],
    )


def test_system_data_property_label_drops_variant():
    """Attaching extra data uses a property label without ``/<variant>``."""
    key = "mtt::charge/dft"
    transform = get_system_data_transform([key])
    systems, _, _ = transform([_system()], {}, {key: _extra(key)})

    stored = systems[0].get_data(key)
    assert stored.block().properties.names == ["charge"]
