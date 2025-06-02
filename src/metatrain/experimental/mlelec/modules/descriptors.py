import json
from typing import List, Optional, Union

import ase
import torch

from featomic.torch import (
    systems_to_torch,
    SphericalExpansion,
    SphericalExpansionByPair,
)
from featomic.torch.clebsch_gordan import (
    EquivariantPowerSpectrum,
    EquivariantPowerSpectrumByPair,
)
import metatensor.torch as mts
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import System

from ..utils import symmetrize_samples


def get_descriptor_calculator(
    model_hypers, dtype, atomic_types=None, selected_keys=None
):
    """Compute descriptors for the systems."""

    spex_hypers = model_hypers["node"]
    pair_hypers = model_hypers["edge"]

    calc1b = SphericalExpansion(**spex_hypers)
    calc2b = SphericalExpansionByPair(**pair_hypers)

    return HamiltonianDescriptor(
        calc1b,
        calc2b,
        dtype=dtype,
        neighbor_types=atomic_types,
        selected_keys=selected_keys,
    )


class HamiltonianDescriptor(torch.nn.Module):

    def __init__(
        self,
        one_center_calculator,
        two_center_calculator,
        neighbor_types: Optional[List[int]] = None,
        *,
        selected_keys: Optional[Labels] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):

        super().__init__()
        self.one_center_calculator = one_center_calculator
        self.two_center_calculator = two_center_calculator
        self.neighbor_types = neighbor_types
        self.selected_keys = selected_keys
        self.dtype = dtype
        self.device = device

        supported_one_center_calculators = [
            "lode_spherical_expansion",
            "spherical_expansion",
        ]
        supported_two_center_calculators = ["spherical_expansion_by_pair"]

        if self.one_center_calculator.c_name not in supported_one_center_calculators:
            raise ValueError(
                f"Only [{', '.join(supported_one_center_calculators)}] are supported "
                f"for `calculator_1`, got '{self.one_center_calculator.c_name}'"
            )

        parameters_1 = json.loads(one_center_calculator.parameters)

        if self.two_center_calculator.c_name not in supported_two_center_calculators:
            raise ValueError(
                f"Only [{', '.join(supported_two_center_calculators)}] are supported "
                f"for `calculator_2`, got '{self.two_center_calculator.c_name}'"
            )

        parameters_2 = json.loads(two_center_calculator.parameters)

        if parameters_1["basis"]["type"] != "TensorProduct":
            raise ValueError("only 'TensorProduct' basis is supported for calculator_1")

        if parameters_2["basis"]["type"] != "TensorProduct":
            raise ValueError("only 'TensorProduct' basis is supported for calculator_2")

        self.node_descriptor_calculator = EquivariantPowerSpectrum(
            self.one_center_calculator,
            neighbor_types=self.neighbor_types,
            dtype=self.dtype,
            device=self.device,
        )
        self.edge_descriptor_calculator = SymmetrizedPowerSpectrumByPair(
            self.one_center_calculator,
            self.two_center_calculator,
            neighbor_types=self.neighbor_types,
            dtype=self.dtype,
            device=self.device,
        )

    def compute_metadata(self, atom_types: List[int] = None):

        if self.neighbor_types is None:
            assert (
                atom_types is not None
            ), "atom_types must be provided when self.neighbor_types is None"
        if atom_types is None:
            assert (
                self.neighbor_types is not None
            ), "self.neighbor_types must be provided when atom_types is None"

            atom_types = self.neighbor_types

        dummy_system = systems_to_torch(
            [
                ase.Atoms(
                    numbers=atom_types * 3,
                    positions=[
                        [i / 4, j / 4, 0]
                        for i in range(len(atom_types))
                        for j in range(3)
                    ],
                )
            ]
        )
        metadata_1 = self.node_descriptor_calculator.compute_metadata(
            dummy_system, neighbors_to_properties=True
        )
        metadata_2 = self.edge_descriptor_calculator.compute_metadata(dummy_system)
        return join_node_and_edge_descriptors(metadata_1, metadata_2)

    def forward(self, systems: Union[List[ase.Atoms], ase.Atoms], **kwargs):

        if isinstance(systems, ase.Atoms):
            systems = [systems]
        systems = [
            system.to(dtype=self.dtype, device=self.device)
            for system in systems_to_torch(systems)
        ]

        # dummy_system = systems_to_torch(
        #     [
        #         ase.Atoms(
        #             numbers=self.neighbor_types * 3,
        #             positions=[
        #                 [i / 4, j / 4, 0]
        #                 for i in range(len(self.neighbor_types))
        #                 for j in range(3)
        #             ],
        #         )
        #     ]
        # )
        # systems = systems + dummy_system

        node_descriptor = self.node_descriptor_calculator.compute(
            systems, selected_keys=self.selected_keys, neighbors_to_properties=True
        )  # .keys_to_properties(["neighbor_1_type", "neighbor_2_type"])
        edge_descriptor = self.edge_descriptor_calculator.compute(
            systems, selected_keys=self.selected_keys
        )

        # node_descriptor = mts.slice(
        #     node_descriptor,
        #     "samples",
        #     mts.Labels(
        #         "system",
        #         torch.arange(len(systems) - 1).reshape(-1, 1),
        #     ),
        # )
        node_descriptor = drop_empty_blocks(node_descriptor)
        # edge_descriptor = mts.slice(
        #     edge_descriptor,
        #     "samples",
        #     mts.Labels("system", torch.arange(len(systems) - 1).reshape(-1, 1)),
        # )
        edge_descriptor = drop_empty_blocks(edge_descriptor)
        return join_node_and_edge_descriptors(node_descriptor, edge_descriptor)


def drop_empty_blocks(tensor: TensorMap) -> TensorMap:
    keys_to_drop: List[torch.Tensor] = []
    for k, b in tensor.items():
        if b.values.shape[0] == 0:
            keys_to_drop.append(k.values)
    if len(keys_to_drop) > 0:
        out = mts.drop_blocks(
            tensor, Labels(tensor.keys.names, torch.stack(keys_to_drop))
        )
    else:
        out = tensor
    return out


def join_node_and_edge_descriptors(node: TensorMap, edge: TensorMap) -> TensorMap:
    """
    TMP implementation based on `block_type`
    """

    # Duplicate metadata for nodes
    device = node[0].values.device
    new_node: List[torch.Tensor] = []
    new_node_keys: List[torch.Tensor] = []
    for k, b in node.items():
        new_node_keys.append(
            torch.tensor(
                [k["o3_lambda"], k["o3_sigma"], k["center_type"], k["center_type"], 0],
                device=device,
                dtype=torch.int32,
            )
        )
        idx = b.samples.values[:, 1].unsqueeze(1)
        cell_shifts = torch.zeros(idx.shape[0], 3, dtype=torch.int32, device=device)
        new_samples = Labels(
            edge[0].samples.names, torch.hstack([b.samples.values, idx, cell_shifts])
        )
        new_node.append(
            TensorBlock(
                samples=new_samples,
                components=b.components,
                properties=b.properties,
                values=b.values,
            )
        )
    new_node = TensorMap(Labels(edge.keys.names, torch.stack(new_node_keys)), new_node)

    # Change block_type=2 to block_type=0
    block_type_2: List[TensorBlock] = []
    block_type_2_keys: List[torch.Tensor] = []
    block_type_1: List[TensorBlock] = []
    block_type_1_keys: List[torch.Tensor] = []
    for k, b in edge.items():
        if k["block_type"] == 2:
            key = k.values
            key[-1] = 0
            block_type_2.append(b)
            block_type_2_keys.append(key)
        else:
            block_type_1.append(b)
            block_type_1_keys.append(k.values)
    if len(block_type_2_keys) > 0:
        block_type_2 = TensorMap(
            Labels(edge.keys.names, torch.stack(block_type_2_keys)), block_type_2
        )
    if len(block_type_1_keys) > 0:
        block_type_1 = TensorMap(
            Labels(edge.keys.names, torch.stack(block_type_1_keys)), block_type_1
        )
    new_edge = mts.join(
        [block_type_1, block_type_2], "samples", "union", remove_tensor_name=True
    )
    new_edge = mts.permute_dimensions(new_edge, "properties", [0, 2, 3, 4, 1, 5])
    out = mts.join([new_node, new_edge], "samples", "union", remove_tensor_name=True)
    assert "tensor" not in out[0].samples.names

    out = mts.rename_dimension(out, "keys", "block_type", "s2_pi")
    out = mts.permute_dimensions(out, "keys", [0, 1, 4, 2, 3])
    return out


class SymmetrizedPowerSpectrumByPair(torch.nn.Module):
    """
    Computes a symmetrized two-center descriptor starting from an
    EquivariantPowerSpectrum and an EquivariantPowespectrumByPair and symmetrizing the
    latter under equivalent atom permutations.

    TODO: update docstrings
    This calculator should call the relevant featomic calculators
    (EquivariantPowerSpectrum, EquivariantPowerSpectrumByPair) and symmetrize them
    according to the permutation group of dimension 2.
    """

    def __init__(
        self,
        one_center_calculator,
        two_center_calculator,
        neighbor_types: Optional[List[int]] = None,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):

        super().__init__()
        self.one_center_calculator = one_center_calculator
        self.two_center_calculator = two_center_calculator
        self.neighbor_types = neighbor_types
        self.dtype = dtype
        self.device = device

        supported_one_center_calculators = [
            "lode_spherical_expansion",
            "spherical_expansion",
        ]
        supported_two_center_calculators = ["spherical_expansion_by_pair"]

        if self.one_center_calculator.c_name not in supported_one_center_calculators:
            raise ValueError(
                f"Only [{', '.join(supported_one_center_calculators)}] are supported "
                f"for `calculator_1`, got '{self.one_center_calculator.c_name}'"
            )

        parameters_1 = json.loads(one_center_calculator.parameters)

        if self.two_center_calculator.c_name not in supported_two_center_calculators:
            raise ValueError(
                f"Only [{', '.join(supported_two_center_calculators)}] are supported "
                f"for `calculator_2`, got '{self.two_center_calculator.c_name}'"
            )

        parameters_2 = json.loads(two_center_calculator.parameters)

        if parameters_1["basis"]["type"] != "TensorProduct":
            raise ValueError("only 'TensorProduct' basis is supported for calculator_1")

        if parameters_2["basis"]["type"] != "TensorProduct":
            raise ValueError("only 'TensorProduct' basis is supported for calculator_2")

        self.powerspectrum_by_pair_calculator = EquivariantPowerSpectrumByPair(
            self.one_center_calculator,
            self.two_center_calculator,
            neighbor_types=self.neighbor_types,
            dtype=self.dtype,
            device=self.device,
        )

    @property
    def name(self):
        """Name of this calculator."""
        return "SymmetrizedPowerSpectrumByPair"

    def compute_metadata(
        self,
        systems: Union[System, List[System]],
        selected_keys: Optional[Labels] = None,
        selected_samples: Optional[Labels] = None,
    ) -> TensorMap:

        metadata = self.powerspectrum_by_pair_calculator.compute_metadata(
            systems,
            selected_keys=selected_keys,
            selected_samples=selected_samples,
            neighbors_to_properties=True,
        )

        return self._symmetrize_under_permutations(metadata)

    def compute(
        self,
        systems: Union[System, List[System]],
        selected_keys: Optional[Labels] = None,
        selected_samples: Optional[Labels] = None,
    ) -> TensorMap:
        """
        Compute the symmetrized two-center descriptor.
        """

        # Compute powerspectrum by pair
        powspec_by_pair = self.powerspectrum_by_pair_calculator.compute(
            systems,
            selected_keys=selected_keys,
            selected_samples=selected_samples,
            neighbors_to_properties=True,
        )

        return self._symmetrize_under_permutations(powspec_by_pair)

    def forward(
        self,
        systems,  #: Union[ase.Atoms, List[ase.Atoms]],
    ) -> TensorMap:
        """
        :param systems: :py:class:`list` of :py:class:`tuple` or :py:class:`tuple` of
            :py:class:`tuple` of :py:class:`tuple` and :py:class:`tuple` containing the
            systems to compute the symmetrized two-center descriptor for.
        :return: :py:class:`metatensor.torch.TensorMap` containing the symmetrized
            two-center descriptor.
        """
        return self.compute(systems)

    def _symmetrize_under_permutations(self, powspec_by_pair: TensorMap) -> TensorMap:
        """
        Symmetrizes the two-center descriptor.
        """

        symmetrized_powspec_by_pair: List[TensorBlock] = []
        new_keys: List[torch.Tensor] = []

        key_names = powspec_by_pair.keys.names + ["block_type"]

        for k, b in powspec_by_pair.items():

            # Check the TensorMap is not empty
            if b.values.shape[0] == 0:
                continue

            # If same species, symmetrize under permutation
            if k["first_atom_type"] == k["second_atom_type"]:
                # Symmetrize the samples
                b_plus, b_minus = symmetrize_samples(b, k["second_atom_type"])

                # Append blocks and keys for block_type = 1
                symmetrized_powspec_by_pair.append(b_plus)
                new_keys.append(torch.cat([k.values, torch.tensor([1])], dim=0))

                # Append blocks and keys for block_type = -1
                symmetrized_powspec_by_pair.append(b_minus)
                new_keys.append(torch.cat([k.values, torch.tensor([-1])], dim=0))

            # If different species, do not symmetrize. Concatenate features
            elif k["first_atom_type"] < k["second_atom_type"]:
                keys_dict = {
                    name: int(value)
                    for name, value in zip(k.names, k.values, strict=False)
                }
                keys_dict["first_atom_type"] = int(k["second_atom_type"])
                keys_dict["second_atom_type"] = int(k["first_atom_type"])
                other_block = powspec_by_pair.block(keys_dict)

                property_names = b.properties.names
                property_names.insert(1, "neighbor_2_type")
                first_properties = b.properties.insert(
                    1,
                    "neighbor_2_type",
                    torch.tensor(
                        b.properties.values.shape[0] * [k["second_atom_type"]]
                    ),
                )
                second_properties = other_block.properties.insert(
                    1,
                    "neighbor_2_type",
                    torch.tensor(
                        other_block.properties.values.shape[0] * [k["first_atom_type"]]
                    ),
                )
                property_values = torch.cat(
                    [first_properties.values, second_properties.values], dim=0
                )

                symmetrized_powspec_by_pair.append(
                    TensorBlock(
                        samples=b.samples,
                        components=b.components,
                        properties=Labels(
                            property_names,
                            property_values,
                        ),
                        values=torch.cat([b.values, other_block.values], dim=-1),
                    )
                )
                new_keys.append(torch.cat([k.values, torch.tensor([2])], dim=0))

        edge_descriptor = TensorMap(
            keys=Labels(
                names=key_names,
                values=torch.stack(new_keys),
            ),
            blocks=symmetrized_powspec_by_pair,
        )

        return edge_descriptor
