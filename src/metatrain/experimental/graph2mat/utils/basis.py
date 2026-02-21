import numpy as np
import yaml
from graph2mat import AtomicTableWithEdges, PointBasis


def get_basis_table_from_yaml(basis_yaml: str) -> AtomicTableWithEdges:
    """Reads a yaml file and initializes an AtomicTableWithEdges object.

    :param basis_yaml: Path to the yaml file.
    :return: The corresponding AtomicTableWithEdges object.
    """

    # Load the yaml basis file
    with open(basis_yaml, "r") as f:
        basis_yaml = yaml.safe_load(f)

    basis = []
    for point_basis in basis_yaml:
        if isinstance(point_basis["R"], list):
            point_basis["R"] = np.array(point_basis["R"])
        if isinstance(point_basis["basis"], list):
            point_basis["basis"] = tuple(tuple(x) for x in point_basis["basis"])
        basis.append(PointBasis(**point_basis).to_sisl_atom(Z=point_basis["type"]))

    return AtomicTableWithEdges(basis)
