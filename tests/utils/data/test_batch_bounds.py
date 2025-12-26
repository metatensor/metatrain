"""Tests for batch bounds functionality."""

import pytest
import torch
from metatomic.torch import System

from metatrain.utils.data import CollateFn, CollateFnWithBatchBounds


def test_collate_fn_with_batch_bounds_creation():
    """Test that CollateFnWithBatchBounds can be created with valid parameters."""
    base_collate = CollateFn(target_keys=["energy"])

    # Test with both bounds
    wrapper = CollateFnWithBatchBounds(
        collate_fn=base_collate, batch_atom_bounds=[5, 100]
    )
    assert wrapper.min_atoms_per_batch == 5
    assert wrapper.max_atoms_per_batch == 100

    # Test with only min bound
    wrapper = CollateFnWithBatchBounds(
        collate_fn=base_collate, batch_atom_bounds=[5, None]
    )
    assert wrapper.min_atoms_per_batch == 5
    assert wrapper.max_atoms_per_batch is None

    # Test with only max bound
    wrapper = CollateFnWithBatchBounds(
        collate_fn=base_collate, batch_atom_bounds=[None, 100]
    )
    assert wrapper.min_atoms_per_batch is None
    assert wrapper.max_atoms_per_batch == 100

    # Test with no bounds (should still work)
    wrapper = CollateFnWithBatchBounds(collate_fn=base_collate)
    assert wrapper.min_atoms_per_batch is None
    assert wrapper.max_atoms_per_batch is None


def test_collate_fn_with_batch_bounds_invalid_bounds():
    """Test that CollateFnWithBatchBounds rejects invalid bounds."""
    base_collate = CollateFn(target_keys=["energy"])

    # Test min > max
    with pytest.raises(ValueError, match="must be less than or equal to"):
        CollateFnWithBatchBounds(
            collate_fn=base_collate, batch_atom_bounds=[100, 50]
        )

    # Test negative min
    with pytest.raises(ValueError, match="must be at least 1"):
        CollateFnWithBatchBounds(
            collate_fn=base_collate, batch_atom_bounds=[0, 100]
        )

    # Test negative max
    with pytest.raises(ValueError, match="must be at least 1"):
        CollateFnWithBatchBounds(
            collate_fn=base_collate, batch_atom_bounds=[5, 0]
        )


def test_collate_fn_with_batch_bounds_filtering():
    """Test that batches are correctly filtered based on atom counts."""

    # Create a mock collate function that doesn't actually collate
    def mock_collate(batch):
        return "collated"

    # Create wrapper with bounds
    wrapper = CollateFnWithBatchBounds(
        collate_fn=mock_collate, batch_atom_bounds=[10, 50]
    )

    # Create a batch with too few atoms (5 atoms)
    positions_small = torch.tensor([[0.0, 0.0, 0.0]] * 5, dtype=torch.float64)
    cell_small = torch.eye(3, dtype=torch.float64) * 10.0
    system_small = System(
        types=torch.tensor([1] * 5),
        positions=positions_small,
        cell=cell_small,
        pbc=torch.tensor([True, True, True]),
    )
    batch_small = [{"system": system_small}]

    # Should return None for too few atoms
    result = wrapper(batch_small)
    assert result is None

    # Create a batch with too many atoms (60 atoms)
    positions_large = torch.tensor([[0.0, 0.0, 0.0]] * 60, dtype=torch.float64)
    cell_large = torch.eye(3, dtype=torch.float64) * 10.0
    system_large = System(
        types=torch.tensor([1] * 60),
        positions=positions_large,
        cell=cell_large,
        pbc=torch.tensor([True, True, True]),
    )
    batch_large = [{"system": system_large}]

    # Should return None for too many atoms
    result = wrapper(batch_large)
    assert result is None

    # Create a batch with the right number of atoms (20 atoms)
    positions_ok = torch.tensor([[0.0, 0.0, 0.0]] * 20, dtype=torch.float64)
    cell_ok = torch.eye(3, dtype=torch.float64) * 10.0
    system_ok = System(
        types=torch.tensor([1] * 20),
        positions=positions_ok,
        cell=cell_ok,
        pbc=torch.tensor([True, True, True]),
    )
    batch_ok = [{"system": system_ok}]

    # Should not return None for valid atom count
    result = wrapper(batch_ok)
    assert result == "collated"


def test_collate_fn_with_batch_bounds_multiple_systems():
    """Test batch bounds with multiple systems in a batch."""
    # Create a mock collate function that doesn't actually collate
    def mock_collate(batch):
        return "collated"

    # Create wrapper with bounds
    wrapper = CollateFnWithBatchBounds(
        collate_fn=mock_collate, batch_atom_bounds=[15, 30]
    )

    # Create two systems with 8 atoms each (total: 16 atoms)
    positions = torch.tensor([[0.0, 0.0, 0.0]] * 8, dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64) * 10.0
    system1 = System(
        types=torch.tensor([1] * 8),
        positions=positions,
        cell=cell,
        pbc=torch.tensor([True, True, True]),
    )
    system2 = System(
        types=torch.tensor([1] * 8),
        positions=positions,
        cell=cell,
        pbc=torch.tensor([True, True, True]),
    )
    batch = [{"system": system1}, {"system": system2}]

    # Should not raise error for atom count (total = 16, within bounds)
    result = wrapper(batch)
    assert result == "collated"
