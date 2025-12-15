import torch
from e3nn import o3
from metatensor.torch import Labels, TensorBlock, TensorMap
from omegaconf import DictConfig

from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.data.target_info import _REGISTERED_TARGET_TYPES


def get_mts_components(point_basis):
    component_names = ["o3_lambda", "o3_sigma", "o3_mu", "i_zeta"]

    component_values = []
    for mul, l, sigma in point_basis.basis:
        for i_zeta in range(mul):
            for m in range(-l, l + 1):
                component_values.append([l, sigma, m, i_zeta])

    return Labels(
        names=component_names,
        values=torch.tensor(component_values, dtype=torch.int32),
    )


def _get_basis_target_info(target_name: str, target: DictConfig) -> TargetInfo:
    keys = list([[0, 0]])
    basis_size = [5]

    layout = TensorMap(
        keys=Labels(["first_atom_type", "second_atom_type"], torch.tensor(keys)),
        blocks=[
            TensorBlock(
                values=torch.empty(
                    0, basis_size[type_0], basis_size[type_1], 1, dtype=torch.float64
                ),
                samples=Labels(
                    names=[
                        "system",
                        "first_atom",
                        "second_atom",
                        "cell_shift_a",
                        "cell_shift_b",
                        "cell_shift_c",
                    ],
                    values=torch.empty((0, 6), dtype=torch.int32),
                ),
                components=[
                    Labels(
                        ["first_atom_basis_function"],
                        torch.arange(basis_size[type_0]).reshape(-1, 1),
                    ),
                    Labels(
                        ["second_atom_basis_function"],
                        torch.arange(basis_size[type_1]).reshape(-1, 1),
                    ),
                ],
                properties=Labels(["_"], torch.tensor([[0]])),
            )
            for type_0, type_1 in keys
        ],
    )

    return TargetInfo(
        layout=layout,
        quantity=target.get("quantity", ""),
        unit=target.get("unit", ""),
        description=target.get("description", ""),
    )


_REGISTERED_TARGET_TYPES["basis"] = _get_basis_target_info


def _wrap_in_tensorblock(data_values, i):
    return TensorBlock(
        values=data_values.reshape(-1, 1).to(torch.float64),
        samples=Labels(
            names=["system", "matrix_element"],
            values=torch.tensor(
                [[i] * data_values.shape[0], torch.arange(data_values.shape[0])]
            ).T,
        ),
        components=[],
        properties=Labels(["_"], torch.tensor([[0]])),
    )


def g2m_labels_to_tensormap(
    node_labels: torch.Tensor, edge_labels: torch.Tensor, i: int = 0
) -> TensorMap:
    return TensorMap(
        keys=Labels(["graph2mat_point_or_edge"], torch.tensor([[0], [1]])),
        blocks=[
            _wrap_in_tensorblock(node_labels, i),
            _wrap_in_tensorblock(edge_labels, i),
        ],
    )


def get_e3nn_target_info(target_name: str, target: dict) -> TargetInfo:
    """Get the target info corresponding to some e3nn irreps.

    :param target_name: Name of the target.
    :param target: Target dictionary containing the irreps and other info.
    :return: The corresponding TargetInfo object.
    """
    sample_names = ["system"]
    if target["per_atom"]:
        sample_names.append("atom")

    properties_name = target.get("properties_name", target_name.replace("mtt::", ""))

    irreps = o3.Irreps(target["type"]["spherical"]["irreps"])
    keys = []
    blocks = []
    for irrep in irreps:
        o3_lambda = irrep.ir.l
        o3_sigma = irrep.ir.p * ((-1) ** o3_lambda)
        num_properties = irrep.mul

        components = [
            Labels(
                names=["o3_mu"],
                values=torch.arange(
                    -o3_lambda, o3_lambda + 1, dtype=torch.int32
                ).reshape(-1, 1),
            )
        ]
        block = TensorBlock(
            # float64: otherwise metatensor can't serialize
            values=torch.empty(
                0,
                2 * o3_lambda + 1,
                num_properties,
                dtype=torch.float64,
            ),
            samples=Labels(
                names=sample_names,
                values=torch.empty((0, len(sample_names)), dtype=torch.int32),
            ),
            components=components,
            properties=Labels.range(properties_name, num_properties),
        )
        keys.append([o3_lambda, o3_sigma])
        blocks.append(block)

    layout = TensorMap(
        keys=Labels(["o3_lambda", "o3_sigma"], torch.tensor(keys, dtype=torch.int32)),
        blocks=blocks,
    )

    target_info = TargetInfo(
        quantity=target.get("quantity", ""),
        unit=target.get("unit", ""),
        layout=layout,
    )
    return target_info


def split_dataset_info(dataset_info: DatasetInfo, node_hidden_irreps: str):
    """Splits the dataset info into one info for the featurizer and one for graph2mat."""
    graph2mat_targets = {}
    featurizer_targets = {}
    for target_name, target_info in dataset_info.targets.items():
        if target_info.layout.keys.names == ["first_atom_type", "second_atom_type"]:
            graph2mat_targets[target_name] = target_info

            featurizer_targets[f"mtt::aux::graph2mat_{target_name}"] = (
                get_e3nn_target_info(
                    target_name=f"mtt::aux::graph2mat_{target_name}",
                    target={
                        "type": {"spherical": {"irreps": node_hidden_irreps}},
                        "quantity": "",
                        "unit": "",
                        "per_atom": True,
                        "properties_name": "_",
                    },
                )
            )
        else:
            featurizer_targets[target_name] = target_info

    featurizer_dataset_info = DatasetInfo(
        length_unit=dataset_info.length_unit,
        atomic_types=dataset_info.atomic_types,
        targets=featurizer_targets,
    )

    graph2mat_dataset_info = DatasetInfo(
        length_unit=dataset_info.length_unit,
        atomic_types=dataset_info.atomic_types,
        targets=graph2mat_targets,
    )

    return featurizer_dataset_info, graph2mat_dataset_info
