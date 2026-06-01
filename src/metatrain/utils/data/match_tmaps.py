from typing import Optional

import torch
from metatensor.torch import TensorMap
from metatomic.torch import System

from .atom_pair_helpers import get_bidirectional_edges, get_single_direction_edges
from .atomic_basis_helpers import (
    densify_atomic_basis_target,
    sparsify_atomic_basis_target,
)
from .spherical_helpers import couple_tensor_blocks, uncouple_tensor_blocks


def match_layout(
    tmap1: TensorMap,
    tmap2: TensorMap,
    target_info_layout: Optional[TensorMap] = None,
    systems: Optional[list[System]] = None,
    cg_coeffs: Optional[TensorMap] = None,
) -> TensorMap:
    """Makes sure the layout of the two tensormaps is the same.

    It raises an error if the two tensormaps are incompatible.

    Coupling and uncoupling of spherical tensor blocks is not supported
    inside torchscript.

    :param tmap1: The first tensormap, whose layout will be modified
      to match the second.
    :param tmap2: The second tensormap, whose layout will be used as
      the reference.
    :param target_info_layout: Layout of the data. It will only be used in
      case tmap1 needs to be densified or sparsified (in atom types)
      to match tmap2.
    :param systems: List of systems. It will only be used in case tmap1 needs
      to be sparsified (in atom types) to match tmap2.
    :param cg_coeffs: Clebsch-Gordan coefficients. It will only be used in
      case tmap1 needs to be coupled or uncoupled to match tmap2.
    :return: The first tensormap, with its layout modified to match
      the second.
    """

    # Check whether the tensormaps are dense or sparse in atom types.
    tmap1_isdense = (
        "atom_type" not in tmap1.keys.names
        and "first_atom_type" not in tmap1.keys.names
    )
    tmap2_isdense = (
        "atom_type" not in tmap2.keys.names
        and "first_atom_type" not in tmap2.keys.names
    )

    # Check whether the tensormaps are atom-pair or not.
    is_atom_pair_tmap1 = "first_atom" in tmap1.block(0).samples.names
    is_atom_pair_tmap2 = "first_atom" in tmap2.block(0).samples.names

    # If one of the tensormaps is atom-pair and the other is not,
    # it is impossible to match their layout.
    if is_atom_pair_tmap1 != is_atom_pair_tmap2:
        raise ValueError(
            f"Cannot match layout of {tmap1} and {tmap2}: one is an atom-pair "
            "TensorMap and the other is not."
        )

    # If both tensormaps are atom-pair, check whether they contain single-direction
    # or bidirectional edges, and possibly convert tmap1 to match tmap2.
    if is_atom_pair_tmap1:
        if tmap1_isdense:
            is_single_dir_tmap1 = torch.all(
                tmap1.block(0).samples.column("first_atom")
                <= tmap1.block(0).samples.column("second_atom")
            )
        else:
            is_single_dir_tmap1 = torch.all(
                tmap1.keys.column("first_atom_type")
                <= tmap1.keys.column("second_atom_type")
            )
        if tmap2_isdense:
            is_single_dir_tmap2 = torch.all(
                tmap2.block(0).samples.column("first_atom")
                <= tmap2.block(0).samples.column("second_atom")
            )
        else:
            is_single_dir_tmap2 = torch.all(
                tmap2.keys.column("first_atom_type")
                <= tmap2.keys.column("second_atom_type")
            )

        if is_single_dir_tmap1 and not is_single_dir_tmap2:
            tmap1 = get_bidirectional_edges(tmap1)
        elif not is_single_dir_tmap1 and is_single_dir_tmap2:
            tmap1 = get_single_direction_edges(tmap1)

    # Check whether the two tensormaps are dense or sparse in atom types,
    # and possibly convert tmap1 to match tmap2.
    if not tmap1_isdense and tmap2_isdense:
        if target_info_layout is None:
            raise ValueError(
                "Cannot densify atomic basis target without target_info_layout."
            )
        tmap1 = densify_atomic_basis_target(tmap1, target_info_layout, fill_value=0.0)
    elif tmap1_isdense and not tmap2_isdense:
        if systems is None:
            raise ValueError("Cannot sparsify atomic basis target without systems.")
        tmap1 = sparsify_atomic_basis_target(
            systems,
            tmap1,
            target_info_layout,
        )

    # Check that both tensormaps are either spherical or not.
    # If they are spherical, check if they are coupled or uncoupled
    # and modify tmap1 to match tmap2.
    tmap1_is_spherical = (
        "o3_lambda" not in tmap1.keys.names and "o3_lambda_1" not in tmap1.keys.names
    )
    tmap2_is_spherical = (
        "o3_lambda" not in tmap2.keys.names and "o3_lambda_1" not in tmap2.keys.names
    )
    if tmap1_is_spherical != tmap2_is_spherical:
        raise ValueError(
            f"Cannot match layout of {tmap1} and {tmap2}: one is a spherical "
            "TensorMap and the other is not."
        )
    elif tmap1_is_spherical:
        tmap1_is_coupled = "o3_lambda_1" in tmap1.keys.names
        tmap2_is_coupled = "o3_lambda_1" in tmap2.keys.names
        if cg_coeffs is None:
            raise ValueError(
                "Cannot couple or uncouple spherical tensor blocks without "
                "Clebsch-Gordan coefficients."
            )
        if tmap1_is_coupled and not tmap2_is_coupled:
            tmap1 = uncouple_tensor_blocks(tmap1, cg_coeffs)
        elif not tmap1_is_coupled and tmap2_is_coupled:
            tmap1 = couple_tensor_blocks(tmap1, cg_coeffs)

    return tmap1
