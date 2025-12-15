import warnings
from collections import defaultdict
from typing import Callable, Optional

import graph2mat
import sisl
import torch
from e3nn import o3
from graph2mat import (
    AtomicTableWithEdges,
    BasisConfiguration,
    BasisMatrix,
    MatrixDataProcessor,
)
from graph2mat.bindings.torch import TorchBasisMatrixDataset
from graph2mat.core.data.basis import NoBasisAtom, get_change_of_basis
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import NeighborListOptions, System, register_autograd_neighbors

from metatrain.utils.data import system_to_ase

from .mtt import g2m_labels_to_tensormap


def system_to_config(
    system: System,
    data_processor: MatrixDataProcessor,
    block_dict: Optional[dict[tuple[int, int, int], torch.Tensor]] = None,
) -> BasisConfiguration:
    """Convert a Metatomic System to a Graph2Mat BasisConfiguration."""

    basis = data_processor.basis_table.atoms

    geometry = sisl.Geometry.new(system_to_ase(system))

    for atom in geometry.atoms.atom:
        for basis_atom in basis:
            if basis_atom.tag == atom.tag:
                break
        else:
            basis_atom = NoBasisAtom(atom.Z, tag=atom.tag)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            geometry.atoms.replace_atom(atom, basis_atom)

    if block_dict is None:
        matrix = None
    else:
        matrix = BasisMatrix(block_dict, geometry.nsc, geometry.orbitals)

    return BasisConfiguration.from_geometry(geometry, matrix=matrix)


def get_converters_to_spherical(
    basis_table: AtomicTableWithEdges,
) -> dict[int, torch.Tensor]:
    """Get the converters to spherical harmonics for each basis type in the basis table."""

    cob, _ = get_change_of_basis(basis_table.basis_convention, "spherical")

    p_change_of_basis = torch.tensor(cob)
    d_change_of_basis = o3.Irrep(2, 1).D_from_matrix(p_change_of_basis)

    converters = {}
    for point_basis in basis_table.basis:
        M = torch.eye(point_basis.basis_size)

        start = 0
        for mul, l, p in point_basis.basis:
            if p != (-1) ** l:
                raise ValueError(
                    "Only spherical basis with definite parity are supported."
                )

            for i in range(mul):
                end = start + (2 * l + 1)
                if l == 1:
                    M[start:end, start:end] = p_change_of_basis
                elif l == 2:
                    M[start:end, start:end] = d_change_of_basis
                else:
                    M[start:end, start:end] = o3.Irrep(l, 1).D_from_matrix(
                        p_change_of_basis
                    )
                start = end

        converters[point_basis.type] = M

    return converters


def get_graph2mat_transform(
    graph2mat_processors: dict[str, MatrixDataProcessor],
    nls_options: dict[str, NeighborListOptions],
) -> Callable:
    """Returns a transform function that processes systems and targets
    to adapt them to Graph2Mat.

    Essentially, a graph2mat batch is just a flat array, and in this
    transform we convert the target in the metatrain format to this
    flat array format.

    Also, each graph2mat instance will require a different graph (which
    can be different from the graph used by the featurizer). Therefore,
    we also compute and add the neighbor lists required by each graph2mat
    instance.
    """

    converters = {}
    for target_name in graph2mat_processors:
        converters[target_name] = get_converters_to_spherical(
            graph2mat_processors[target_name].basis_table
        )

        print(converters[target_name])

    def transform(
        systems: list[System],
        targets: dict[str, TensorMap],
        extra: dict[str, TensorMap],
    ) -> tuple[list[System], dict[str, TensorMap], dict[str, TensorMap]]:
        system_indices = (
            extra["system_index"].block(0).values.ravel().to(torch.int64).tolist()
        )

        for target_name in targets:
            configs = [
                system_to_config(system, graph2mat_processors[target_name], None)
                for system in systems
            ]

            dataset = TorchBasisMatrixDataset(
                configs,
                data_processor=graph2mat_processors[target_name],
                data_cls=graph2mat.bindings.torch.TorchBasisMatrixData,
                load_labels=False,
            )

            lattices = {
                i: sisl.Lattice(data.cell.numpy(), data.nsc.reshape(3).numpy())
                for i, data in zip(system_indices, dataset, strict=True)
            }

            block_dict_matrices = defaultdict(dict)

            tensormap_matrix = targets[target_name]

            for key in tensormap_matrix.keys:
                block = tensormap_matrix[key]

                block_values = block.values
                first_atom_type, second_atom_type = key
                conv_left = converters[target_name][int(first_atom_type)].to(
                    block_values.dtype
                )
                conv_right = converters[target_name][int(second_atom_type)].to(
                    block_values.dtype
                )

                block_values = torch.einsum(
                    "ij, bjkp, kl -> bilp", conv_left, block_values, conv_right.T
                )

                samples = block.samples.values

                for i_pair, (
                    i_system,
                    first_atom,
                    second_atom,
                    *cell_shifts,
                ) in enumerate(samples):
                    cell_index = lattices[int(i_system)].isc_off[
                        cell_shifts[0], cell_shifts[1], cell_shifts[2]
                    ]

                    block_dict_matrices[int(i_system)][
                        (int(first_atom), int(second_atom), int(cell_index))
                    ] = block_values[i_pair].squeeze(-1)

            configs = [
                system_to_config(
                    system, graph2mat_processors[target_name], block_dict_matrices[i]
                )
                for i, system in zip(system_indices, systems, strict=True)
            ]

            dataset = TorchBasisMatrixDataset(
                configs,
                data_processor=graph2mat_processors[target_name],
                data_cls=graph2mat.bindings.torch.TorchBasisMatrixData,
                load_labels=True,
            )

            all_point_labels = []
            all_edge_labels = []

            for i, data in enumerate(dataset):
                all_point_labels.append(data.point_labels)
                all_edge_labels.append(data.edge_labels)

                edge_index = data.edge_index
                distances = data.positions[edge_index].diff(dim=0)

                cell_shifts = sisl.Lattice(
                    data.cell.numpy(), data.nsc.reshape(3).numpy()
                ).sc_off[data.neigh_isc]

                neighbor_list = TensorBlock(
                    values=distances.reshape(-1, 3, 1).to(systems[i].positions.dtype),
                    samples=Labels(
                        names=[
                            "first_atom",
                            "second_atom",
                            "cell_shift_a",
                            "cell_shift_b",
                            "cell_shift_c",
                        ],
                        values=torch.hstack([edge_index.T, torch.tensor(cell_shifts)]),
                        assume_unique=True,
                    ),
                    components=[Labels.range("xyz", 3)],
                    properties=Labels.range("distance", 1),
                )

                register_autograd_neighbors(systems[i], neighbor_list)
                systems[i].add_neighbor_list(nls_options[target_name], neighbor_list)

            targets[target_name] = g2m_labels_to_tensormap(
                node_labels=torch.cat(all_point_labels, dim=0),
                edge_labels=torch.cat(all_edge_labels, dim=0),
                i=i,
            )

        return systems, targets, extra

    return transform
