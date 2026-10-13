import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.atomic_basis_helpers import densify_atomic_basis_dataset_info
from metatrain.utils.data.match_tmaps import match_layout
from metatrain.utils.data.target_info import get_generic_target_info


_sparse_target_info = get_generic_target_info(
    "spherical_atomic_basis",
    {
        "quantity": "",
        "unit": "",
        "type": {
            "spherical": {
                "irreps": {
                    1: [{"o3_lambda": 0, "o3_sigma": 1}],
                    8: [
                        {"num": 2, "o3_lambda": 0, "o3_sigma": 1},
                        {"o3_lambda": 1, "o3_sigma": 1},
                    ],
                }
            }
        },
        "num_subtargets": 1,
        "sample_kind": "atom",
    },
)

_dataset_info = DatasetInfo(
    length_unit="angstrom",
    atomic_types=[1, 8],
    targets={"mtt::spherical_atomic_basis": _sparse_target_info},
)

_dense_target_info = densify_atomic_basis_dataset_info(_dataset_info).targets[
    "mtt::spherical_atomic_basis"
]
_sparse_layout = _sparse_target_info.layout
_dense_layout = _dense_target_info.layout


def test_jit_script():
    torch.jit.script(match_layout)


@pytest.mark.parametrize("target_info_layout", ["sparse", "dense", None])
def test_sparse_to_dense(target_info_layout):

    info_layout = {"sparse": _sparse_layout, "dense": _dense_layout, None: None}[
        target_info_layout
    ]

    densified = match_layout(_sparse_layout, _dense_layout, info_layout)

    assert densified.keys.names == _dense_layout.keys.names
    assert densified.keys.values.shape[0] == _dense_layout.keys.values.shape[0]
    assert all(
        (
            densified.keys.values[i].tolist() == _dense_layout.keys.values[i].tolist()
            for i in range(densified.keys.values.shape[0])
        )
    )


def test_dense_to_sparse():

    systems = [
        System(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
            ),
            types=torch.tensor([1, 1, 8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
    ]

    device = _dense_layout.device

    dense_target = TensorMap(
        keys=_dense_layout.keys,
        blocks=[
            TensorBlock(
                values=torch.rand(
                    (3, *block.values.shape[1:]),
                    dtype=block.values.dtype,
                    device=device,
                ),
                samples=Labels(
                    names=block.samples.names,
                    values=torch.tensor(
                        [[0, 0], [0, 1], [0, 2]], dtype=torch.int64, device=device
                    ),
                ),
                components=block.components,
                properties=block.properties,
            )
            for block in _dense_layout.blocks()
        ],
    )

    sparsified = match_layout(
        dense_target,
        _sparse_layout,
        _sparse_layout,
        systems=systems,  # We don't need actual systems for this test
    )

    assert sparsified.keys.names == _sparse_layout.keys.names
    assert sparsified.keys.values.shape[0] == _sparse_layout.keys.values.shape[0]
    assert all(
        (
            sparsified.keys.values[i].tolist() == _sparse_layout.keys.values[i].tolist()
            for i in range(sparsified.keys.values.shape[0])
        )
    )

    for key, block in sparsified.items():
        dense_values = dense_target.block(
            dict(o3_lambda=key["o3_lambda"], o3_sigma=key["o3_sigma"])
        ).values
        sparse_values = block.values
        if key["atom_type"] == 1:
            assert torch.allclose(sparse_values, dense_values[[0, 1], ..., :1])
        elif key["atom_type"] == 8:
            assert torch.allclose(sparse_values, dense_values[[2], ...])
