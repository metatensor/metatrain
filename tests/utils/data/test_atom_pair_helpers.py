import metatensor.torch as mts
import pytest

from metatrain.utils.data import DiskDataset
from metatrain.utils.data.atom_pair_helpers import (
    get_bidirectional_edges,
    get_single_direction_edges,
)

from ...conftest import RESOURCES_PATH


@pytest.mark.parametrize("batch_size", [1, 2])
def test_edges_single_bidirectional_roundtrip(batch_size):
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

    single_dir_data = get_single_direction_edges(batch_data)
    recovered_data = get_bidirectional_edges(single_dir_data)

    mts.allclose_raise(
        mts.sort(recovered_data, ["keys", "samples", "properties"]),
        mts.sort(batch_data, ["keys", "samples", "properties"]),
        atol=1e-7,
    )
