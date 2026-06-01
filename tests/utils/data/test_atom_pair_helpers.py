import metatensor.torch as mts
import pytest
import torch
from featomic.torch.clebsch_gordan._coefficients import (
    calculate_cg_coefficients,
)
from metatensor.torch import Labels, TensorMap

from metatrain.utils.data import DiskDataset
from metatrain.utils.data.atom_pair_helpers import (
    get_bidirectional_edges,
    get_single_direction_edges,
)
from metatrain.utils.data.spherical_helpers import uncouple_tensor_blocks

from ...conftest import RESOURCES_PATH


@pytest.fixture(params=[1, 2])
def batch_size(request):
    return request.param


@pytest.fixture(params=["coupled", "uncoupled"])
def basis(request):
    return request.param


def test_jit_script_single_direction():
    torch.jit.script(get_single_direction_edges)


def test_jit_script_bidirectional():
    torch.jit.script(get_bidirectional_edges)


def test_edges_single_bidirectional_roundtrip(batch_size, basis):
    """Tests that the functions to get single direction edges
    and to recover the bidirectional edges work correctly
    by doing a round trip and checking that the data is
    the same as the original one.

    We load bidirectional data and we do:
    bidirectional -> single direction -> bidirectional.

    :param batch_size: The number of samples to load from the dataset
      for the test.
    """

    target_name = "mtt::matrix_edges::ham_coup"

    bidirectional_ds = DiskDataset(
        RESOURCES_PATH / "scfbench_2_bidirectional_edges.zip"
    )
    batch_data = mts.join(
        [
            sample[target_name]
            for i, sample in enumerate(bidirectional_ds)
            if i < batch_size
        ],
        axis="samples",
    )

    # Remove blocks with no samples.
    keys = []
    blocks = []
    for key, block in batch_data.items():
        if block.samples.values.shape[0] > 0:
            keys.append(key.values)
            blocks.append(block.copy(deep=False))
    batch_data = TensorMap(
        keys=Labels(batch_data.keys.names, torch.stack(keys)), blocks=blocks
    )

    if basis == "uncoupled":
        cg_coeffs = calculate_cg_coefficients(
            4,
            cg_backend="python-dense",
            arrays_backend="torch",
            dtype=torch.float64,
            device=torch.device("cpu"),
        )

        batch_data = uncouple_tensor_blocks(batch_data, cg_coeffs)

    # Do round trip
    single_dir_data = get_single_direction_edges(batch_data)
    recovered_data = get_bidirectional_edges(single_dir_data)

    # TODO: Don't sort on properties maybe? They should probably come out with
    # the properties sorted.
    mts.allclose_raise(
        mts.sort(recovered_data, ["keys", "samples", "properties"]),
        mts.sort(batch_data, ["keys", "samples", "properties"]),
        atol=1e-7,
    )
