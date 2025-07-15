import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorMap
from metatomic.torch import System

from metatrain.utils.transfer import batch_to


def test_batch_to_dtype():
    system = System(
        positions=torch.tensor([[1.0, 1.0, 1.0]]),
        cell=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        types=torch.tensor([1]),
        pbc=torch.tensor([True, True, True]),
    )
    targets = TensorMap(
        keys=Labels.single(),
        blocks=[mts.block_from_array(torch.tensor([[1.0]]))],
    )

    systems = [system]
    targets = {"energy": targets}

    assert systems[0].positions.dtype == torch.float32
    assert systems[0].cell.dtype == torch.float32
    assert targets["energy"].block().values.dtype == torch.float32

    systems, targets, _ = batch_to(systems, targets, dtype=torch.float64)

    assert systems[0].positions.dtype == torch.float64
    assert systems[0].cell.dtype == torch.float64
    assert targets["energy"].block().values.dtype == torch.float64


def test_batch_to_device():
    system = System(
        positions=torch.tensor([[1.0, 1.0, 1.0]]),
        cell=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        types=torch.tensor([1]),
        pbc=torch.tensor([True, True, True]),
    )
    targets = TensorMap(
        keys=Labels.single(),
        blocks=[mts.block_from_array(torch.tensor([[1.0]]))],
    )

    systems = [system]
    targets = {"energy": targets}

    assert systems[0].positions.device == torch.device("cpu")
    assert systems[0].types.device == torch.device("cpu")
    assert targets["energy"].block().values.device == torch.device("cpu")

    systems, targets, _ = batch_to(systems, targets, device=torch.device("meta"))

    assert systems[0].positions.device == torch.device("meta")
    assert systems[0].types.device == torch.device("meta")
    assert targets["energy"].block().values.device == torch.device("meta")
