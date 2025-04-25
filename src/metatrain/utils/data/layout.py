"""
Module for determining the layout type of a target from its metadata.
"""

from typing import List

import torch
from metatensor.torch import TensorMap


@torch.jit.script
def get_sample_kind(layout: TensorMap) -> str:
    """
    Get the sample kind of the layout.

    Will be one of:
    - "per_structure"
    - "per_atom"
    - "per_pair"
    """
    valid_sample_names_per_structure: List[str] = ["system"]
    valid_sample_names_per_atom: List[str] = ["system", "atom"]
    valid_sample_names_per_pair: List[str] = [
        "system",
        "first_atom",
        "second_atom",
        "cell_shift_a",
        "cell_shift_b",
        "cell_shift_c",
    ]

    if layout.sample_names == valid_sample_names_per_structure:
        sample_kind = "per_structure"

    elif layout.sample_names == valid_sample_names_per_atom:
        sample_kind = "per_atom"

    elif layout.sample_names == valid_sample_names_per_pair:
        sample_kind = "per_pair"

    else:
        raise ValueError(
            "The layout ``TensorMap`` of a target should have samples "
            f"names corresponding to: {valid_sample_names_per_structure}, "
            f"or  {valid_sample_names_per_atom}, "
            f"or {valid_sample_names_per_pair}, but found "
            f"'{layout.sample_names}' instead."
        )

    return sample_kind


@torch.jit.script
def get_target_type(layout: TensorMap) -> str:
    """
    Get the target type of the layout.

    Will be one of:
    - "scalar"
    - "cartesian"
    - "spherical"
    - "spherical_atomic_basis"
    """

    # examine basic properties of all blocks
    for block in layout.blocks():
        if len(block.values) != 0:
            raise ValueError(
                "The layout ``TensorMap`` of a target should have 0 "
                f"samples, but found {len(block.values)} samples."
            )

    # examine the components of the first block to decide whether this is
    # a scalar, a Cartesian tensor, a spherical tensor, or an atomic-basis
    # spherical tensor

    if len(layout) == 0:
        raise ValueError(
            "The layout ``TensorMap`` of a target should have at least one "
            "block, but found 0 blocks."
        )

    components_first_block = layout.block(0).components
    if len(components_first_block) == 0:
        target_type = "scalar"
    elif components_first_block[0].names[0].startswith("xyz"):
        target_type = "cartesian"
    elif (
        len(components_first_block) == 1
        and components_first_block[0].names[0] == "o3_mu"
    ):
        # keys of just "o3_lambda" and "o3_sigma" is a spherical target
        if layout.keys.names == [
            "o3_lambda",
            "o3_sigma",
        ]:
            target_type = "spherical"

        # keys with "o3_lambda" and "o3_sigma" and keys that indicate center types
        # (i.e. "center_type", "first_atom_type", "second_atom_type") plus
        # (optionally) other arbitrary key dimensions (i.e. "s2_pi", etc) is an
        # spherical target on an atomic basis.
        elif layout.keys.names == ["o3_lambda", "o3_sigma", "center_type"]:
            target_type = "spherical_atomic_basis"
        elif layout.keys.names == [
            "o3_lambda",
            "o3_sigma",
            "first_atom_type",
            "second_atom_type",
        ]:
            target_type = "spherical_atomic_basis"
        elif layout.keys.names == [
            "o3_lambda",
            "o3_sigma",
            "s2_pi",
            "first_atom_type",
            "second_atom_type",
        ]:
            target_type = "spherical_atomic_basis"

        else:
            raise ValueError(
                f"invalid key names: {layout.keys.names}. "
                "Targets with 1 'o3_mu' components axis are "
                "treated as either spherical targets or spherical "
                "targets on an atomic basis, and are expected to have "
                "'o3_lambda' and 'o3_sigma' key dimensions, along "
                "with ['center_type'], ['first_atom_type', "
                "'second_atom_type'], or ['s2_pi', 'first_atom_type',"
                "'second_atom_type'] key dimensions."
            )
    else:
        raise ValueError(
            "The type of the layout ``TensorMap`` could not be determined."
        )

    return target_type
