import warnings
from collections import defaultdict
from typing import Callable, Optional

import graph2mat
import sisl
import torch
from graph2mat import (
    BasisConfiguration,
    BasisMatrix,
    MatrixDataProcessor,
)
from graph2mat.bindings.torch import TorchBasisMatrixDataset
from graph2mat.core.data.basis import NoBasisAtom
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import NeighborListOptions, System, register_autograd_neighbors

from metatrain.utils.data import TargetInfo, system_to_ase

from .conversions import get_target_converters
from .mtt import g2m_labels_to_tensormap


def system_to_config(
    system: System,
    data_processor: MatrixDataProcessor,
    block_dict: Optional[dict[tuple[int, int, int], torch.Tensor]] = None,
    nsc: Optional[torch.Tensor] = None,
) -> BasisConfiguration:
    """Convert a Metatomic System to a Graph2Mat BasisConfiguration."""

    basis = data_processor.basis_table.atoms

    ase_atoms = system_to_ase(system)
    for i in range(3):
        if not ase_atoms.pbc[i]:
            ase_atoms.cell[i, i] = (
                ase_atoms.positions[:, i].max() - ase_atoms.positions[:, i].min() + 20.0
            )
    geometry = sisl.Geometry.new(ase_atoms)

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
        matrix = BasisMatrix(
            block_dict, nsc if nsc is not None else geometry.nsc, geometry.orbitals
        )

    return BasisConfiguration.from_geometry(geometry, matrix=matrix)


def get_graph2mat_transform(
    graph2mat_processors: dict[str, MatrixDataProcessor],
    nls_options: dict[str, NeighborListOptions],
    matrices: dict[str, dict],
    target_infos: dict[str, TargetInfo],
    add_neighbor_lists: bool = True,
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
    for matrix_name in graph2mat_processors:
        converters[matrix_name] = get_target_converters(
            graph2mat_processors[matrix_name].basis_table,
            in_format=graph2mat_processors[matrix_name].basis_table.basis_convention,
            out_format="spherical",
        )

    orb_pointers = {}
    for matrix_name in graph2mat_processors:
        # Get pointers to the position of each orbital
        basis_table = graph2mat_processors[matrix_name].basis_table
        # Dict with keys atom
        orb_pointers[matrix_name] = {}
        for point_basis in basis_table.basis:
            i_orb = 0
            for mul, l, par in point_basis.basis:
                sigma = (-1) ** l * par
                for i in range(mul):
                    orb_pointers[matrix_name][(point_basis.type, l, sigma, i)] = i_orb
                    i_orb += 2 * l + 1

    def get_prop_to_matrix(orb_pointers, key, block, type):
        l_1 = key["o3_lambda_1"]
        l_2 = key["o3_lambda_2"]
        sigma_1 = key["o3_sigma_1"]
        sigma_2 = key["o3_sigma_2"]

        if type == "node":
            atom_type_1 = int(key["atom_type"])
            atom_type_2 = atom_type_1
        else:
            atom_type_1 = int(key["first_atom_type"])
            atom_type_2 = int(key["second_atom_type"])

        property_to_matrix = []
        for prop in block.properties:
            n_1 = prop["n_1"]
            n_2 = prop["n_2"]

            start_1 = orb_pointers[(atom_type_1, l_1, sigma_1, n_1)]
            start_2 = orb_pointers[(atom_type_2, l_2, sigma_2, n_2)]
            length_1 = 2 * l_1 + 1
            length_2 = 2 * l_2 + 1

            prop_to_matrix_1 = torch.arange(start_1, start_1 + length_1)
            prop_to_matrix_2 = torch.arange(start_2, start_2 + length_2)

            # Grid of prop_to_matrix_1 x prop_to_matrix_2
            x, y = torch.meshgrid(prop_to_matrix_1, prop_to_matrix_2, indexing="ij")
            property_to_matrix.append(torch.stack([x, y], dim=0))

        # Shape is (2, n_components_1, n_components_2, n_properties)
        return torch.stack(property_to_matrix, dim=-1)

    def get_tmap_prop_to_matrix(orb_pointers, tmap):

        tmap_type = "node" if "atom_type" in tmap.keys.names else "edge"

        blocks = []
        for key, block in tmap.items():
            values = get_prop_to_matrix(orb_pointers, key, block, tmap_type)
            blocks.append(
                TensorBlock(
                    values=values,
                    samples=Labels(
                        names=["dimension"],  # 0 for row, 1 for column
                        values=torch.tensor(
                            [[0], [1]], dtype=torch.int32, device=values.device
                        ),
                    ),
                    components=block.components,
                    properties=block.properties,
                )
            )

        return TensorMap(
            keys=tmap.keys,
            blocks=blocks,
        )

    layouts = {}
    for matrix_name, matrix_spec in matrices.items():
        node_target = matrix_spec["nodes"]
        edge_target = matrix_spec["edges"]

        node_target_info = target_infos[node_target]
        edge_target_info = target_infos[edge_target]

        layouts[matrix_name] = {
            "nodes": get_tmap_prop_to_matrix(
                orb_pointers[matrix_name], node_target_info.layout
            ),
            "edges": get_tmap_prop_to_matrix(
                orb_pointers[matrix_name], edge_target_info.layout
            ),
        }

    def transform(
        systems: list[System],
        targets: dict[str, TensorMap],
        extra: dict[str, TensorMap],
    ) -> tuple[list[System], dict[str, TensorMap], dict[str, TensorMap]]:
        system_indices = (
            extra["mtt::aux::system_index"]
            .block(0)
            .values.ravel()
            .to(torch.int64)
            .tolist()
        )

        def convert_tmaps(
            matrix_name, nodes_tmap, edges_tmap, systems, add_neighbor_list=True
        ):

            configs = [
                system_to_config(system, graph2mat_processors[matrix_name], None)
                for system in systems
            ]

            dataset = TorchBasisMatrixDataset(
                configs,
                data_processor=graph2mat_processors[matrix_name],
                data_cls=graph2mat.bindings.torch.TorchBasisMatrixData,
                load_labels=False,
            )

            lattices = {
                i: sisl.Lattice(data.cell.numpy(), data.nsc.reshape(3).numpy())
                for i, data in zip(system_indices, dataset, strict=True)
            }

            block_dict_matrices = defaultdict(dict)

            # Add nodes
            properties_to_matrix = layouts[matrix_name]["nodes"]
            is_nodes_coupled = "l_1" in nodes_tmap.block(0).properties.names
            if is_nodes_coupled:
                raise NotImplementedError("Coupled node blocks are not supported yet.")
            for key, block in nodes_tmap.items():
                atom_type = int(key["atom_type"])

                property_to_matrix = properties_to_matrix.block(key).values

                mat_block_shape = tuple(
                    basis_table.point_block_shape[
                        :, basis_table.type_to_index(atom_type)
                    ]
                )

                for i, sample in enumerate(block.samples.values):
                    i_system, atom_index = sample
                    system_matrix = block_dict_matrices[int(i_system)]
                    entry_key = (int(atom_index), int(atom_index), 0)
                    if entry_key not in system_matrix:
                        system_matrix[entry_key] = torch.full(
                            mat_block_shape,
                            torch.nan,
                            dtype=block.values.dtype,
                            device=block.values.device,
                        )

                    system_matrix[entry_key][
                        property_to_matrix[0], property_to_matrix[1]
                    ] = block.values[i]

            # Add edges
            properties_to_matrix = layouts[matrix_name]["edges"]
            for key, block in edges_tmap.items():
                at_type_1 = int(key["first_atom_type"])
                at_type_2 = int(key["second_atom_type"])
                property_to_matrix = properties_to_matrix.block(key).values

                edge_type = basis_table.point_type_to_edge_type(
                    [
                        [basis_table.type_to_index(at_type_1)],
                        [basis_table.type_to_index(at_type_2)],
                    ]
                )[0]

                mat_block_shape = tuple(basis_table.edge_block_shape[:, abs(edge_type)])
                if edge_type < 0:
                    mat_block_shape = (mat_block_shape[1], mat_block_shape[0])

                for i_pair, (
                    i_system,
                    first_atom,
                    second_atom,
                    *cell_shifts,
                ) in enumerate(block.samples.values):
                    cell_index = lattices[int(i_system)].isc_off[
                        cell_shifts[0], cell_shifts[1], cell_shifts[2]
                    ]
                    entry_key = (int(first_atom), int(second_atom), int(cell_index))
                    if entry_key not in block_dict_matrices[int(i_system)]:
                        block_dict_matrices[int(i_system)][entry_key] = torch.full(
                            mat_block_shape,
                            torch.nan,
                            dtype=block.values.dtype,
                            device=block.values.device,
                        )

                    block_dict_matrices[int(i_system)][entry_key][
                        property_to_matrix[0], property_to_matrix[1]
                    ] = block.values[i_pair]

            configs = [
                system_to_config(
                    system,
                    graph2mat_processors[matrix_name],
                    block_dict_matrices[i],
                    nsc=lattices[i].nsc,
                )
                for i, system in zip(system_indices, systems, strict=True)
            ]

            dataset = TorchBasisMatrixDataset(
                configs,
                data_processor=graph2mat_processors[matrix_name],
                data_cls=graph2mat.bindings.torch.TorchBasisMatrixData,
                load_labels=True,
            )

            all_point_labels = []
            all_edge_labels = []

            for i, data in enumerate(dataset):
                all_point_labels.append(data.point_labels)
                all_edge_labels.append(data.edge_labels)

                if add_neighbor_list:
                    edge_index = data.edge_index
                    distances = data.positions[edge_index].diff(dim=0)

                    cell_shifts = sisl.Lattice(
                        data.cell.numpy(), data.nsc.reshape(3).numpy()
                    ).sc_off[data.neigh_isc]

                    neighbor_list = TensorBlock(
                        values=distances.reshape(-1, 3, 1).to(
                            systems[i].positions.dtype
                        ),
                        samples=Labels(
                            names=[
                                "first_atom",
                                "second_atom",
                                "cell_shift_a",
                                "cell_shift_b",
                                "cell_shift_c",
                            ],
                            values=torch.hstack(
                                [edge_index.T, torch.tensor(cell_shifts)]
                            ),
                            assume_unique=True,
                        ),
                        components=[Labels.range("xyz", 3)],
                        properties=Labels.range("distance", 1),
                    )

                    register_autograd_neighbors(systems[i], neighbor_list)
                    systems[i].add_neighbor_list(
                        nls_options[matrix_name], neighbor_list
                    )

            return g2m_labels_to_tensormap(
                node_labels=torch.cat(all_point_labels, dim=0),
                edge_labels=torch.cat(all_edge_labels, dim=0),
            )

        for matrix_name, matrix_spec in matrices.items():
            node_target = matrix_spec["nodes"]
            edge_target = matrix_spec["edges"]

            targets[node_target], targets[edge_target] = convert_tmaps(
                matrix_name,
                targets[node_target],
                targets[edge_target],
                systems,
                add_neighbor_list=add_neighbor_lists,
            )

            node_scales = f"mtt::aux::scales::{node_target}"
            edge_scales = f"mtt::aux::scales::{edge_target}"

            if node_scales not in extra or edge_scales not in extra:
                continue

            extra[node_scales], extra[edge_scales] = convert_tmaps(
                matrix_name,
                extra[node_scales],
                extra[edge_scales],
                systems,
                add_neighbor_list=False,
            )

            node_per_property_scales = f"mtt::aux::per-property-scales::{node_target}"
            edge_per_property_scales = f"mtt::aux::per-property-scales::{edge_target}"

            extra[node_per_property_scales], extra[edge_per_property_scales] = (
                convert_tmaps(
                    matrix_name,
                    extra[node_per_property_scales],
                    extra[edge_per_property_scales],
                    systems,
                    add_neighbor_list=False,
                )
            )

        return systems, targets, extra

    return transform


def add_neighbor_lists(
    systems: list[System],
    graph2mat_processors: dict[str, MatrixDataProcessor],
    nls_options: dict[str, NeighborListOptions],
) -> list[System]:
    for matrix_name in graph2mat_processors:
        # TODO: This should only be done for the requested outputs
        configs = [
            system_to_config(system, graph2mat_processors[matrix_name], None)
            for system in systems
        ]

        dataset = TorchBasisMatrixDataset(
            configs,
            data_processor=graph2mat_processors[matrix_name],
            data_cls=graph2mat.bindings.torch.TorchBasisMatrixData,
            load_labels=False,
        )

        for i, data in enumerate(dataset):
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
            ).to(systems[i].positions.device)

            register_autograd_neighbors(systems[i], neighbor_list)
            systems[i].add_neighbor_list(nls_options[matrix_name], neighbor_list)

    return systems


def get_graph2mat_eval_transform(
    graph2mat_processors: dict[str, MatrixDataProcessor],
    nls_options: dict[str, NeighborListOptions],
    outputs: Optional[list[str]] = None,
) -> Callable:
    """Same as `get_graph2mat_transform`, but for evaluation."""

    def transform(
        systems: list[System],
        targets: dict[str, TensorMap],
        extra: dict[str, TensorMap],
    ) -> tuple[list[System], dict[str, TensorMap], dict[str, TensorMap]]:

        systems = add_neighbor_lists(systems, graph2mat_processors, nls_options)

        return systems, targets, extra

    return transform


def graph2mat_to_tensormap(
    batch,  # The batch
    out,  # The output of the model, which contains the predicted node and edge labels
    processor: MatrixDataProcessor,
    node_labels_name: str,
    edge_labels_name: str,
):
    # ---------------------------------------------------------
    #    ACCUMULATE THE PREDICTIONS INTO A DICTIONARY
    # ---------------------------------------------------------
    predictions = {
        "node_labels": out[node_labels_name].block().values.ravel(),
        "edge_labels": out[edge_labels_name].block().values.ravel(),
    }

    batch["node_labels"] = predictions["node_labels"]
    batch["edge_labels"] = predictions["edge_labels"]

    node_blocks = defaultdict(list)
    edge_blocks = defaultdict(list)
    node_samples = defaultdict(list)
    edge_samples = defaultdict(list)

    ptr_nodes = 0
    edge_start = 0
    ptr_edges = 0
    for i_system in range(len(batch["ptr"]) - 1):
        # ----- FIRST, WE GET THE ATOM BLOCKS
        # Get the atom index where the system starts and ends in the batch
        atom_start = batch["ptr"][i_system].item()
        atom_end = batch["ptr"][i_system + 1].item()

        # Atom types of the system
        atom_types = batch["point_types"][atom_start:atom_end].tolist()
        atom_Zs = torch.tensor(processor.basis_table.types)[atom_types].tolist()
        # Get the pointers to the position of each atom block for this system
        atom_ptrs = ptr_nodes + processor.basis_table.atom_block_pointer(atom_types)

        # Loop over atoms and store their blocks, as well as the corresponding samples
        for i_atom, atom_Z in enumerate(atom_Zs):
            start, end = atom_ptrs[i_atom : i_atom + 2]
            node_blocks[atom_Z].append(batch["node_labels"][start:end])
            node_samples[atom_Z].append((i_system, i_atom))

        # Next system will start where the current one ends
        ptr_nodes = atom_ptrs[-1].item()

        # ----- THEN, WE GET THE EDGE BLOCKS
        # Number of edges (both directions)
        n_edges = batch["n_edges"][i_system]
        # Get only one direction if the matrix is symmetric
        step = 2 if processor.symmetric_matrix else 1

        # Get edge types and atom pairs (edge indices) for this system
        edge_types = batch["edge_types"][edge_start : edge_start + n_edges : step]
        edge_indices = batch["edge_index"][:, edge_start : edge_start + n_edges : step]
        # Get the pointers to the position of each edge block for this system
        edge_ptrs = ptr_edges + processor.basis_table.edge_block_pointer(
            abs(edge_types).cpu()
        )
        edge_types = edge_types.tolist()

        # Loop over edges and store their blocks, as well as the corresponding samples
        for i_edge in range(n_edges // step):
            start, end = edge_ptrs[i_edge : i_edge + 2]
            src, dst = edge_indices[:, i_edge] - atom_start
            edge_blocks[edge_types[i_edge]].append(batch["edge_labels"][start:end])
            edge_samples[edge_types[i_edge]].append(
                (i_system, int(src), int(dst), 0, 0, 0)
            )

        # Update counters for the next system
        edge_start += int(n_edges)
        ptr_edges = edge_ptrs[-1].item()

    # -----------------------------------------------------
    #                   RESHAPE PROPERLY
    # -----------------------------------------------------
    new_node_blocks = {}
    for k, blocks in node_blocks.items():
        shape = processor.basis_table.point_block_shape[
            :, processor.basis_table.type_to_index(k)
        ]
        new_node_blocks[k] = torch.stack(blocks, dim=0).reshape(-1, *shape)
    node_blocks = new_node_blocks
    node_samples = {k: torch.tensor(v) for k, v in node_samples.items()}

    new_edge_blocks = {}
    new_edge_samples = {}
    for k, blocks in edge_blocks.items():
        # Get atom types from edge type
        sign = 1 if k >= 0 else -1
        edge_type = abs(k)
        at_type_1, at_type_2 = processor.basis_table.edge_type_to_point_types[edge_type]
        if sign == -1:
            at_type_1, at_type_2 = at_type_2, at_type_1
        at_Z_1 = processor.basis_table.types[at_type_1]
        at_Z_2 = processor.basis_table.types[at_type_2]
        shape = processor.basis_table.edge_block_shape[:, edge_type]
        if sign == -1:
            shape = (shape[1], shape[0])
        new_edge_blocks[(at_Z_1, at_Z_2)] = torch.stack(blocks, dim=0).reshape(
            -1, *shape
        )
        new_edge_samples[(at_Z_1, at_Z_2)] = torch.tensor(edge_samples[k])
    edge_blocks = new_edge_blocks
    edge_samples = new_edge_samples

    # ---------------------------------------------------------
    #         CONVERT THE DICTIONARY TO A TENSORMAP
    # ---------------------------------------------------------

    n_ls = {
        point_basis.type: [mul for mul, l, sigma in point_basis.basis]
        for point_basis in processor.basis_table.basis
    }
    # Cumsums
    for atom_type, counts in n_ls.items():
        cumsum = 0
        for i in range(len(counts)):
            count = counts[i] * (2 * i + 1)
            n_ls[atom_type][i] = cumsum
            cumsum += count
        n_ls[atom_type].append(cumsum)

    def get_tmap(input_blocks: dict, input_samples: dict):
        is_atom = isinstance(list(input_blocks.keys())[0], int)
        if is_atom:
            keys_names = [
                "o3_lambda_1",
                "o3_lambda_2",
                "o3_sigma_1",
                "o3_sigma_2",
                "atom_type",
            ]
            samples_names = ["system", "atom"]
        else:
            keys_names = [
                "o3_lambda_1",
                "o3_lambda_2",
                "o3_sigma_1",
                "o3_sigma_2",
                "first_atom_type",
                "second_atom_type",
            ]
            samples_names = [
                "system",
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ]

        blocks = []
        keys = []

        for block_key, blocks_list in input_blocks.items():
            device = blocks_list.device
            if is_atom:
                atom_type_0 = block_key
                atom_type_1 = block_key
            else:
                atom_type_0, atom_type_1 = block_key

            for l_1, l1_start in enumerate(n_ls[atom_type_0][:-1]):
                for l_2, l2_start in enumerate(n_ls[atom_type_1][:-1]):
                    if is_atom:
                        key = (l_1, l_2, 1, 1, atom_type_0)
                    else:
                        key = (l_1, l_2, 1, 1, atom_type_0, atom_type_1)
                    keys.append(key)

                    block_values = []
                    block_properties = []
                    i = l1_start
                    count_1 = (n_ls[atom_type_0][l_1 + 1] - l1_start) // (2 * l_1 + 1)
                    count_2 = (n_ls[atom_type_1][l_2 + 1] - l2_start) // (2 * l_2 + 1)
                    for n_1 in range(count_1):
                        i_end = i + 2 * l_1 + 1
                        j = l2_start
                        for n_2 in range(count_2):
                            j_end = j + 2 * l_2 + 1
                            block_values.append(blocks_list[:, i:i_end, j:j_end])
                            block_properties.append([n_1, n_2])
                            j = j_end
                        i = i_end
                    block = TensorBlock(
                        values=torch.stack(block_values, dim=-1).reshape(
                            -1, 2 * l_1 + 1, 2 * l_2 + 1, len(block_properties)
                        ),
                        samples=Labels(
                            names=samples_names,
                            values=torch.tensor(
                                input_samples[block_key], device=device
                            ),
                        ),
                        components=[
                            Labels(
                                ["o3_mu_1"],
                                torch.arange(-l_1, l_1 + 1, device=device).reshape(
                                    -1, 1
                                ),
                            ),
                            Labels(
                                ["o3_mu_2"],
                                torch.arange(-l_2, l_2 + 1, device=device).reshape(
                                    -1, 1
                                ),
                            ),
                        ],
                        properties=Labels(
                            ["n_1", "n_2"],
                            torch.tensor(block_properties, device=device),
                        ),
                    )
                    blocks.append(block)

        return TensorMap(
            keys=Labels(keys_names, torch.tensor(keys, device=device)),
            blocks=blocks,
        )

    return {
        node_labels_name: get_tmap(node_blocks, node_samples),
        edge_labels_name: get_tmap(edge_blocks, edge_samples),
    }
