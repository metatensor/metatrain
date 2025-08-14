import sys

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.utils.augmentation import RotationalAugmenter
from metatrain.utils.data import TargetInfo


@pytest.fixture
def layout_spherical():
    return TensorMap(
        keys=Labels(
            names=["o3_lambda", "o3_sigma"],
            values=torch.tensor([[0, 1], [2, 1]]),
        ),
        blocks=[
            TensorBlock(
                values=torch.empty(0, 1, 1),
                samples=Labels(
                    names=["system"],
                    values=torch.empty((0, 1), dtype=torch.int32),
                ),
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.arange(0, 1, dtype=torch.int32).reshape(-1, 1),
                    ),
                ],
                properties=Labels.single(),
            ),
            TensorBlock(
                values=torch.empty(0, 5, 1),
                samples=Labels(
                    names=["system"],
                    values=torch.empty((0, 1), dtype=torch.int32),
                ),
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.arange(-2, 3, dtype=torch.int32).reshape(-1, 1),
                    ),
                ],
                properties=Labels.single(),
            ),
        ],
    )


def test_missing_library(monkeypatch, layout_spherical):
    # Pretend 'spherical' is not installed
    monkeypatch.setitem(sys.modules, "spherical", None)

    target_info_dict = {
        "foo": TargetInfo(quantity="energy", unit=None, layout=layout_spherical)
    }

    msg = (
        "To perform data augmentation on spherical targets, please "
        "install the `spherical` package with `pip install spherical`."
    )
    with pytest.raises(ImportError, match=msg):
        RotationalAugmenter(target_info_dict)
