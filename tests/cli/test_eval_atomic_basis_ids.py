"""
Regression test: the ids of an evaluation batch must stay in batch order.

``prepare_atomic_basis_targets`` pairs its ``system_ids`` with ``systems``
*positionally*, so an id list in any other order attaches an id to the wrong
system. The (system, atom) pairs it then builds do not exist in the target, the
padding matches fewer rows than the block has, and the run dies with a shape
mismatch during the final evaluation:

    RuntimeError: shape mismatch: value tensor of shape [348, 1, 8] cannot be
    broadcast to indexing result of shape [301, 1, 8]

The evaluation used to derive the ids with ``torch.unique``, which sorts, so it
worked only when the split file happened to be in ascending order. It fails as
soon as the systems of a batch are not, which is the normal case for a random
split, and only for the subsets a metric is configured on -- the branch does not
run for the others, which is why a run can train and evaluate the training set
happily and die on validation.
"""

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from metatrain.cli.eval import _native_system_ids
from metatrain.utils.data.atomic_basis_helpers import prepare_atomic_basis_targets


#: Two systems of *different* sizes: the mismatch only shows when swapping the
#: ids changes how many (system, atom) pairs exist.
SIZES = [3, 5]

#: Their ids in the dataset, in the order the batch delivers them. Descending,
#: so sorting is not the identity.
NATIVE_IDS = [7, 2]


def _systems():
    return [
        System(
            types=torch.ones(size, dtype=torch.int32),
            positions=torch.arange(3 * size, dtype=torch.float64).reshape(size, 3),
            cell=torch.zeros(3, 3, dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        )
        for size in SIZES
    ]


def _target_and_layout():
    """A per-atom atomic-basis target carrying the native ids in its samples."""
    samples, values = [], []
    for system_id, size in zip(NATIVE_IDS, SIZES, strict=True):
        for atom in range(size):
            samples.append([system_id, atom])
            values.append([[float(system_id), float(atom)]])

    keys = Labels(
        ["o3_lambda", "o3_sigma", "atom_type"],
        torch.tensor([[0, 1, 1]], dtype=torch.int32),
    )
    block = TensorBlock(
        values=torch.tensor(values, dtype=torch.float64),
        samples=Labels(["system", "atom"], torch.tensor(samples, dtype=torch.int32)),
        components=[Labels("o3_mu", torch.tensor([[0]], dtype=torch.int32))],
        properties=Labels("properties", torch.tensor([[0], [1]], dtype=torch.int32)),
    )
    layout = TensorMap(
        keys,
        [
            TensorBlock(
                values=torch.zeros((0, 1, 2), dtype=torch.float64),
                samples=Labels(
                    ["system", "atom"], torch.zeros((0, 2), dtype=torch.int32)
                ),
                components=[Labels("o3_mu", torch.tensor([[0]], dtype=torch.int32))],
                properties=block.properties,
            )
        ],
    )
    return TensorMap(keys, [block]), layout


def test_batch_order_ids_pad_correctly():
    tensor, layout = _target_and_layout()
    padded = prepare_atomic_basis_targets(
        _systems(), torch.tensor(NATIVE_IDS), tensor, layout, None
    )
    assert padded.block(0).values.shape[0] == sum(SIZES)


def test_sorted_ids_break_the_padding():
    """The old behaviour, kept as the thing the fix must prevent."""
    tensor, layout = _target_and_layout()
    sorted_ids = torch.unique(tensor.block(0).samples.column("system"))
    assert sorted_ids.tolist() != NATIVE_IDS, "the ids must be out of order to test"
    with pytest.raises(RuntimeError, match="shape mismatch"):
        prepare_atomic_basis_targets(_systems(), sorted_ids, tensor, layout, None)


def test_native_system_ids_keeps_batch_order():
    extra = {
        "mtt::aux::system_index": TensorMap(
            Labels.single(),
            [
                TensorBlock(
                    values=torch.tensor(
                        [[float(i)] for i in NATIVE_IDS], dtype=torch.float64
                    ),
                    samples=Labels(
                        "system",
                        torch.arange(len(NATIVE_IDS), dtype=torch.int32).reshape(-1, 1),
                    ),
                    components=[],
                    properties=Labels(
                        "system_index", torch.zeros((1, 1), dtype=torch.int32)
                    ),
                )
            ],
        )
    }
    ids = _native_system_ids(extra, torch.device("cpu"))
    assert ids.tolist() == NATIVE_IDS


def test_native_system_ids_reports_a_missing_field():
    with pytest.raises(ValueError, match="mtt::aux::system_index"):
        _native_system_ids({}, torch.device("cpu"))
