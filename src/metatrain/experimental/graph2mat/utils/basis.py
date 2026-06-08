import torch
from graph2mat import AtomicTableWithEdges, PointBasis
from metatensor.torch import TensorMap


# def get_basis_table_from_yaml(basis_yaml: str) -> AtomicTableWithEdges:
#     """Reads a yaml file and initializes an AtomicTableWithEdges object.

#     :param basis_yaml: Path to the yaml file.
#     :return: The corresponding AtomicTableWithEdges object.
#     """

#     # Load the yaml basis file
#     with open(basis_yaml, "r") as f:
#         basis_yaml = yaml.safe_load(f)

#     basis = []
#     for point_basis in basis_yaml:
#         if isinstance(point_basis["R"], list):
#             point_basis["R"] = np.array(point_basis["R"])
#         if isinstance(point_basis["basis"], list):
#             point_basis["basis"] = tuple(tuple(x) for x in point_basis["basis"])
#         basis.append(PointBasis(**point_basis).to_sisl_atom(Z=point_basis["type"]))

#     return AtomicTableWithEdges(basis)


def get_basis_from_layout(layout: TensorMap, R: float) -> AtomicTableWithEdges:

    unique_atom_types = torch.unique(layout.keys["atom_type"])

    num_per_l = {int(atom_type): {} for atom_type in unique_atom_types}

    is_coupled = "l_1" in layout.block(0).properties.names

    for key, block in layout.items():
        atom_type = int(key["atom_type"])

        if is_coupled:
            unique_ls = torch.unique(block.properties["l_1"])
            for l in unique_ls:
                max_n = (
                    block.properties["n_1"][block.properties["l_1"] == l].max().item()
                )
                num_per_l[atom_type][l.item()] = max(
                    num_per_l[atom_type].get(l.item(), 0), max_n + 1
                )
        else:
            l = key["o3_lambda_1"]
            max_n = (
                block.properties["n_1"].max().item()
            )  # All samples in the block have the same l
            num_per_l[atom_type][l] = max(num_per_l[atom_type].get(l, 0), max_n + 1)

    basis = []
    for atom_type, atom_num_per_l in num_per_l.items():
        atom_irreps = []
        for l, num in atom_num_per_l.items():
            irreps = (num, l, (-1) ** l)
            atom_irreps.append(irreps)

        point_basis = {
            "type": atom_type,
            "R": R,
            "basis": tuple(atom_irreps),
        }

        basis.append(PointBasis(**point_basis).to_sisl_atom(Z=point_basis["type"]))

    return AtomicTableWithEdges(basis)
