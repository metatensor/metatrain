import numpy as np
import pytest
import torch
from metatomic.torch import System
from torch.utils.data import DataLoader

from metatrain.utils.data import CollateFn, Dataset, unpack_batch
from metatrain.utils.data.raw_payload import (
    RaggedMatrices,
    RawExtraPayload,
    split_raw_payloads,
)
from metatrain.utils.transfer import batch_to


def _systems(n_atoms_per_system):
    return [
        System(
            types=torch.zeros(n, dtype=torch.int32),
            positions=torch.zeros((n, 3), dtype=torch.float64),
            cell=torch.zeros((3, 3), dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        )
        for n in n_atoms_per_system
    ]


def _attach_ragged(sizes):
    """Collate transform attaching one square matrix per system, as a raw payload."""

    def transform(systems, targets, extra):
        matrices = [
            torch.full((n, n), float(i), dtype=torch.float64)
            for i, n in enumerate(sizes[: len(systems)])
        ]
        extra["matrices"] = RaggedMatrices.from_matrices(matrices)
        return systems, targets, extra

    return transform


def test_ragged_matrices_roundtrip():
    matrices = [torch.eye(3, dtype=torch.float64), torch.ones((2, 2))]
    payload = RaggedMatrices.from_matrices(matrices)

    # stored flat: sum(n_i**2) == 9 + 4, not n_systems * max(n_i)**2 == 18
    assert payload.values.numel() == 13
    assert payload.sizes == [3, 2]

    rebuilt = RaggedMatrices.rebuild(payload.tensors(), payload.metadata())
    for original, recovered in zip(matrices, rebuilt.matrices(), strict=True):
        assert torch.equal(original, recovered)


def test_ragged_matrices_are_views():
    payload = RaggedMatrices.from_matrices([torch.eye(2, dtype=torch.float64)])
    payload.matrices()[0][0, 0] = 7.0
    assert payload.values[0] == 7.0, "matrices() must not copy"


def test_ragged_matrices_empty():
    payload = RaggedMatrices.from_matrices([])
    assert payload.sizes == []
    assert payload.matrices() == []


def test_split_raw_payloads():
    payload = RaggedMatrices.from_matrices([torch.eye(2)])
    serialisable, raw = split_raw_payloads({"keep": 1, "raw": payload})
    assert serialisable == {"keep": 1}
    assert raw == {"raw": payload}


def test_raw_payload_is_abstract():
    with pytest.raises(TypeError):
        RawExtraPayload()


@pytest.mark.parametrize("num_workers", [0, 2])
def test_payload_survives_dataloader(num_workers):
    """The payload must arrive intact, whether or not it crossed a process."""
    sizes = [3, 5, 2, 4]
    dataset = Dataset.from_dict({"system": _systems(sizes)})
    collate_fn = CollateFn(target_keys=[], callables=[_attach_ragged(sizes)])

    loader = DataLoader(
        dataset,
        batch_size=len(sizes),
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    (batch,) = list(loader)
    _, _, extra_data = unpack_batch(batch)

    payload = extra_data["matrices"]
    assert isinstance(payload, RaggedMatrices)
    assert payload.sizes == sizes
    for i, matrix in enumerate(payload.matrices()):
        assert matrix.shape == (sizes[i], sizes[i])
        assert torch.all(matrix == float(i))


def test_payload_bypasses_serialisation():
    """The blob must not grow with the payload: that is the point of the exercise."""
    sizes = [8, 8]
    dataset = Dataset.from_dict({"system": _systems(sizes)})

    plain = CollateFn(target_keys=[])
    with_payload = CollateFn(target_keys=[], callables=[_attach_ragged(sizes)])

    batch = [dataset[i] for i in range(len(sizes))]
    assert with_payload(batch)[0].numel() == plain(batch)[0].numel()


def test_batch_to_moves_payload_without_casting():
    """Payloads follow the device but keep their own precision."""
    payload = RaggedMatrices.from_matrices([torch.eye(2, dtype=torch.float64)])

    _, _, extra_data = batch_to(
        [], {}, {"matrices": payload}, dtype=torch.float32, device=torch.device("cpu")
    )

    moved = extra_data["matrices"]
    assert isinstance(moved, RaggedMatrices)
    # a metric matrix is not a target value, so ``dtype`` must not reach it
    assert moved.values.dtype == torch.float64
    assert np.allclose(moved.matrices()[0].numpy(), np.eye(2))
