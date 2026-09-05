import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.utils.equivariance_penalty import (
    equivariance_variance_output_name,
    reduce_over_augmentations,
    reduce_predictions_over_augmentations,
)


def test_equivariance_variance_output_name():
    """Mirrors ``uncertainty_output_name``'s naming convention under a distinct
    suffix, so the two auxiliary quantities can never collide."""
    assert equivariance_variance_output_name("energy") == "energy_equivariance_variance"
    assert (
        equivariance_variance_output_name("mtt::shift")
        == "mtt::aux::shift_equivariance_variance"
    )


def _replicated_tensor_map(per_system_values, num_augmentations):
    """Builds a TensorMap with ``num_augmentations`` consecutive rows per original
    system, "system" values ``0, ..., n_systems * num_augmentations - 1``, matching
    the layout ``O3Augmenter.replicate_and_augment``/``undo_augmentation`` produce."""
    n_systems = len(per_system_values)
    values = torch.cat(
        [replicas for replicas in per_system_values], dim=0
    )  # (n_systems * num_augmentations, n_properties)
    samples = Labels(
        ["system"], torch.arange(n_systems * num_augmentations).reshape(-1, 1)
    )
    block = TensorBlock(
        values=values,
        samples=samples,
        components=[],
        properties=Labels.range("property", values.shape[1]),
    )
    return TensorMap(keys=Labels.single(), blocks=[block])


def test_reduce_over_augmentations_matches_manual_mean_and_variance():
    """2 systems, 3 augmentations each: the mean/variance of every system's 3
    replicas, computed here by hand with plain torch ops, must match
    ``reduce_over_augmentations`` exactly, and the returned "system" samples must
    be ``0, 1`` (the original system numbering)."""
    num_augmentations = 3
    system_0 = torch.tensor([[1.0], [2.0], [3.0]])
    system_1 = torch.tensor([[10.0], [12.0], [14.0]])
    tensor_map = _replicated_tensor_map([system_0, system_1], num_augmentations)

    mean, variance = reduce_over_augmentations(tensor_map, num_augmentations)

    expected_mean = torch.stack([system_0.mean(dim=0), system_1.mean(dim=0)])
    expected_variance = torch.stack(
        [system_0.var(dim=0, unbiased=True), system_1.var(dim=0, unbiased=True)]
    )
    torch.testing.assert_close(mean.block().values, expected_mean)
    torch.testing.assert_close(variance.block().values, expected_variance)
    torch.testing.assert_close(
        mean.block().samples.column("system"), torch.tensor([0, 1], dtype=torch.int32)
    )
    torch.testing.assert_close(
        variance.block().samples.column("system"),
        torch.tensor([0, 1], dtype=torch.int32),
    )


def test_reduce_over_augmentations_zero_variance_for_identical_replicas():
    """If every replica of a system predicts the same value (e.g. a perfectly
    equivariant model), the variance must come back exactly zero, not merely
    small."""
    num_augmentations = 4
    system_0 = torch.full((4, 2), 5.0)
    tensor_map = _replicated_tensor_map([system_0], num_augmentations)

    _, variance = reduce_over_augmentations(tensor_map, num_augmentations)
    torch.testing.assert_close(variance.block().values, torch.zeros(1, 2))


def test_reduce_over_augmentations_rejects_gradients():
    block = TensorBlock(
        values=torch.zeros(2, 1),
        samples=Labels.range("system", 2),
        components=[],
        properties=Labels.range("property", 1),
    )
    block.add_gradient(
        "positions",
        TensorBlock(
            values=torch.zeros(2, 1),
            samples=Labels.range("sample", 2),
            components=[],
            properties=Labels.range("property", 1),
        ),
    )
    tensor_map = TensorMap(keys=Labels.single(), blocks=[block])

    with pytest.raises(NotImplementedError, match="gradients"):
        reduce_over_augmentations(tensor_map, num_augmentations=2)


def test_reduce_predictions_over_augmentations_adds_variance_only_for_requested():
    """Every prediction gets reduced to its mean; only the targets named in
    ``variance_targets`` also get a variance entry, under
    ``equivariance_variance_output_name``."""
    num_augmentations = 2
    energy = _replicated_tensor_map(
        [torch.tensor([[1.0], [3.0]]), torch.tensor([[10.0], [10.0]])],
        num_augmentations,
    )
    forces = _replicated_tensor_map(
        [torch.tensor([[0.0], [2.0]]), torch.tensor([[5.0], [5.0]])],
        num_augmentations,
    )

    reduced = reduce_predictions_over_augmentations(
        {"energy": energy, "forces": forces},
        num_augmentations,
        variance_targets=["energy"],
    )

    assert set(reduced.keys()) == {
        "energy",
        "forces",
        equivariance_variance_output_name("energy"),
    }
    torch.testing.assert_close(
        reduced["energy"].block().values, torch.tensor([[2.0], [10.0]])
    )
    torch.testing.assert_close(
        reduced["forces"].block().values, torch.tensor([[1.0], [5.0]])
    )
    torch.testing.assert_close(
        reduced[equivariance_variance_output_name("energy")].block().values,
        torch.tensor([[2.0], [0.0]]),
    )
