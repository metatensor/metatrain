import metatensor.torch as mts
import pytest
import torch
from metatensor.torch import Labels, TensorMap
from metatomic.torch import System

from metatrain.utils.transfer import batch_to


@pytest.fixture
def simple_tensormap():
    return TensorMap(
        keys=Labels.single(),
        blocks=[mts.block_from_array(torch.tensor([[1.0]]))],
    )


@pytest.fixture
def simple_system():
    return System(
        positions=torch.tensor([[1.0, 1.0, 1.0]]),
        cell=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        types=torch.tensor([1]),
        pbc=torch.tensor([True, True, True]),
    )


def test_batch_to_dtype(simple_system, simple_tensormap):
    systems = [simple_system]
    targets = {"energy": simple_tensormap}

    assert systems[0].positions.dtype == torch.float32
    assert systems[0].cell.dtype == torch.float32
    assert targets["energy"].block().values.dtype == torch.float32

    systems, targets, _ = batch_to(systems, targets, dtype=torch.float64)

    assert systems[0].positions.dtype == torch.float64
    assert systems[0].cell.dtype == torch.float64
    assert targets["energy"].block().values.dtype == torch.float64


def test_batch_to_device(simple_system, simple_tensormap):
    systems = [simple_system]
    targets = {"energy": simple_tensormap}

    assert systems[0].positions.device == torch.device("cpu")
    assert systems[0].types.device == torch.device("cpu")
    assert targets["energy"].block().values.device == torch.device("cpu")

    systems, targets, _ = batch_to(systems, targets, device=torch.device("meta"))

    assert systems[0].positions.device == torch.device("meta")
    assert systems[0].types.device == torch.device("meta")
    assert targets["energy"].block().values.device == torch.device("meta")


def test_batch_to_extra_data_mask_branch(simple_system, simple_tensormap):
    system = simple_system
    targets = {"energy": simple_tensormap}

    # extra_data with one normal key and one mask key
    extra_data = {
        "feature": simple_tensormap,
        "feature_mask": TensorMap(
            keys=Labels.single(),
            blocks=[mts.block_from_array(torch.tensor([[1]], dtype=torch.int64))],
        ),
    }

    # Apply batch_to requesting float64
    _, _, extra_out = batch_to(
        [system], targets, extra_data=extra_data, dtype=torch.float64
    )

    # The non-mask TensorMap should be float64
    feat_tm = extra_out["feature"]
    assert feat_tm.block().values.dtype == torch.float64

    # The mask TensorMap should be bool, despite original dtype and requested dtype
    mask_tm = extra_out["feature_mask"]
    assert mask_tm.block().values.dtype == torch.bool
