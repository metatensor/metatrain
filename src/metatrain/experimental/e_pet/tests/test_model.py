from __future__ import annotations

import copy

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.pet import model as pet_model_module
from metatrain.pet.model import PET
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.atomic_basis_helpers import (
    densify_atomic_basis_target,
    get_prepare_atomic_basis_targets_transform,
)
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.loss import LossAggregator
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from metatrain.experimental.e_pet import model as e_pet_model_module
from metatrain.experimental.e_pet.model import (
    EPET,
    _cartesian_rank2_to_spherical_components,
    _spherical_components_to_cartesian_rank2,
)


def _base_model_hypers() -> dict:
    hypers = copy.deepcopy(get_default_hypers("experimental.e_pet")["model"])
    hypers["pet"]["cutoff"] = 3.0
    hypers["pet"]["cutoff_width"] = 0.5
    hypers["pet"]["d_pet"] = 4
    hypers["pet"]["d_head"] = 4
    hypers["pet"]["d_node"] = 4
    hypers["pet"]["d_feedforward"] = 8
    hypers["pet"]["num_heads"] = 1
    hypers["pet"]["num_attention_layers"] = 1
    hypers["pet"]["num_gnn_layers"] = 1
    hypers["pet"]["activation"] = "SiLU"
    hypers["pet"]["featurizer_type"] = "residual"
    hypers["tensor_basis_defaults"]["soap"]["max_radial"] = 1
    hypers["tensor_basis_defaults"]["soap"]["cutoff"]["radius"] = 3.0
    hypers["tensor_basis_defaults"]["soap"]["cutoff"]["width"] = 0.5
    hypers["tensor_basis_defaults"]["extra_l1_vector_basis_branches"] = [
        copy.deepcopy(hypers["tensor_basis_defaults"]["soap"])
    ]
    return hypers


def _scalar_target_info(per_atom: bool = True):
    return get_generic_target_info(
        "scalar",
        {
            "quantity": "scalar",
            "unit": "",
            "type": "scalar",
            "num_subtargets": 2,
            "per_atom": per_atom,
        },
    )


def _spherical_target_info(
    name: str = "quadrupole",
    irreps: list[dict[str, int]] | None = None,
    per_atom: bool = True,
):
    if irreps is None:
        irreps = [{"o3_lambda": 2, "o3_sigma": 1}]
    return get_generic_target_info(
        name,
        {
            "quantity": name,
            "unit": "",
            "type": {"spherical": {"irreps": irreps}},
            "num_subtargets": 1,
            "per_atom": per_atom,
        },
    )


def _cartesian_rank2_target_info(name: str = "stress", per_atom: bool = False):
    return get_generic_target_info(
        name,
        {
            "quantity": name,
            "unit": "",
            "type": {"cartesian": {"rank": 2}},
            "num_subtargets": 1,
            "per_atom": per_atom,
        },
    )


def _scalar_dataset_info() -> DatasetInfo:
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 8],
        targets={"scalar": _scalar_target_info()},
    )


def _cartesian_rank2_dataset_info(per_atom: bool = False) -> DatasetInfo:
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 8],
        targets={"stress": _cartesian_rank2_target_info(per_atom=per_atom)},
    )


def _mixed_dataset_info() -> DatasetInfo:
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 8],
        targets={
            "scalar": _scalar_target_info(),
            "quadrupole": _spherical_target_info(),
        },
    )


def _multi_irrep_dataset_info() -> DatasetInfo:
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 8],
        targets={
            "multi": _spherical_target_info(
                name="multi",
                irreps=[
                    {"o3_lambda": 1, "o3_sigma": 1},
                    {"o3_lambda": 2, "o3_sigma": 1},
                    {"o3_lambda": 3, "o3_sigma": 1},
                ],
            )
        },
    )


def _same_irrep_two_targets_dataset_info() -> DatasetInfo:
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 8],
        targets={
            "forces_like_a": _spherical_target_info(
                name="forces_like_a",
                irreps=[{"o3_lambda": 1, "o3_sigma": 1}],
            ),
            "forces_like_b": _spherical_target_info(
                name="forces_like_b",
                irreps=[{"o3_lambda": 1, "o3_sigma": 1}],
            ),
        },
    )


def _atomic_basis_target_info(name: str = "density"):
    return get_generic_target_info(
        name,
        {
            "quantity": name,
            "unit": "",
            "type": {
                "spherical": {
                    "irreps": {
                        1: [
                            {"num": 2, "o3_lambda": 0, "o3_sigma": 1},
                            {"num": 1, "o3_lambda": 1, "o3_sigma": 1},
                        ],
                        6: [
                            {"num": 3, "o3_lambda": 0, "o3_sigma": 1},
                            {"num": 2, "o3_lambda": 1, "o3_sigma": 1},
                            {"num": 1, "o3_lambda": 2, "o3_sigma": 1},
                        ],
                        8: [
                            {"num": 1, "o3_lambda": 0, "o3_sigma": 1},
                            {"num": 1, "o3_lambda": 2, "o3_sigma": 1},
                        ],
                    }
                }
            },
            "num_subtargets": 1,
            "per_atom": True,
        },
    )


def _atomic_basis_dataset_info() -> DatasetInfo:
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 8],
        targets={"density": _atomic_basis_target_info()},
    )


def _atomic_basis_l0_only_dataset_info() -> DatasetInfo:
    target_info = get_generic_target_info(
        "density",
        {
            "quantity": "density",
            "unit": "",
            "type": {
                "spherical": {
                    "irreps": {
                        1: [{"num": 2, "o3_lambda": 0, "o3_sigma": 1}],
                        6: [{"num": 3, "o3_lambda": 0, "o3_sigma": 1}],
                        8: [{"num": 1, "o3_lambda": 0, "o3_sigma": 1}],
                    }
                }
            },
            "num_subtargets": 1,
            "per_atom": True,
        },
    )
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 8],
        targets={"density": target_info},
    )


def _build_system(model, dtype: torch.dtype = torch.float32) -> System:
    system = System(
        types=torch.tensor([1, 6, 8, 1]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.1], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=dtype,
        ),
        cell=torch.zeros((3, 3), dtype=dtype),
        pbc=torch.tensor([False, False, False]),
    )
    return get_system_with_neighbor_lists(system, model.requested_neighbor_lists())


def _atomic_basis_sparse_target(
    target_info, system: System, dtype: torch.dtype = torch.float32
) -> TensorMap:
    blocks = []
    for key, layout_block in target_info.layout.items():
        atom_type = int(key["atom_type"])
        atom_indices = torch.nonzero(system.types == atom_type).reshape(-1)
        sample_values = torch.stack(
            [
                torch.zeros_like(atom_indices, dtype=torch.int32),
                atom_indices.to(torch.int32),
            ],
            dim=1,
        )
        value_shape = [
            len(atom_indices),
            *[len(component) for component in layout_block.components],
            len(layout_block.properties),
        ]
        num_values = 1
        for size in value_shape:
            num_values *= size
        values = torch.arange(
            num_values,
            dtype=dtype,
            device=system.positions.device,
        ).reshape(value_shape)
        blocks.append(
            TensorBlock(
                values=values,
                samples=Labels(layout_block.samples.names, sample_values),
                components=layout_block.components,
                properties=layout_block.properties,
            )
        )

    return TensorMap(target_info.layout.keys, blocks)


def _system_index_extra(system_index: int = 0) -> dict[str, TensorMap]:
    return {
        "mtt::aux::system_index": TensorMap(
            keys=Labels.single(),
            blocks=[
                TensorBlock(
                    values=torch.tensor([[system_index]], dtype=torch.float64),
                    samples=Labels(
                        ["sample"], torch.tensor([[0]], dtype=torch.int32)
                    ),
                    components=[],
                    properties=Labels(
                        ["value"], torch.tensor([[0]], dtype=torch.int32)
                    ),
                )
            ],
        )
    }


def _build_periodic_system(model, dtype: torch.dtype = torch.float32) -> System:
    system = System(
        types=torch.tensor([1, 6, 8, 1]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.1], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=dtype,
        ),
        cell=2.0 * torch.eye(3, dtype=dtype),
        pbc=torch.tensor([True, True, True]),
    )
    return get_system_with_neighbor_lists(system, model.requested_neighbor_lists())


def _single_block_like(tensor_map: TensorMap, values: torch.Tensor) -> TensorMap:
    block = tensor_map.block()
    return TensorMap(
        keys=tensor_map.keys,
        blocks=[
            TensorBlock(
                values=values,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
        ],
    )


def test_e_pet_is_pet_extension() -> None:
    assert issubclass(EPET, PET)


def test_scalar_only_path_matches_pet_exactly() -> None:
    dataset_info = _scalar_dataset_info()
    e_pet_hypers = _base_model_hypers()
    pet_hypers = copy.deepcopy(e_pet_hypers["pet"])

    torch.manual_seed(17)
    pet_model = PET(pet_hypers, dataset_info)
    torch.manual_seed(17)
    e_pet_model = EPET(e_pet_hypers, dataset_info)
    pet_model = pet_model.to(torch.float32).eval()
    e_pet_model = e_pet_model.to(torch.float32).eval()

    pet_system = _build_system(pet_model)
    e_pet_system = _build_system(e_pet_model)
    requested_outputs = {"scalar": ModelOutput(per_atom=True)}

    pet_output = pet_model([pet_system], requested_outputs)["scalar"].block().values
    e_pet_output = e_pet_model([e_pet_system], requested_outputs)["scalar"].block().values
    assert torch.allclose(pet_output, e_pet_output)


def test_mixed_forward_wiring_and_metadata() -> None:
    model = EPET(_base_model_hypers(), _mixed_dataset_info()).eval()
    system = _build_system(model)
    outputs = model(
        [system],
        {
            "scalar": ModelOutput(per_atom=True),
            "quadrupole": ModelOutput(per_atom=True),
        },
    )

    assert outputs["scalar"].block().values.shape == (4, 2)
    assert outputs["quadrupole"].block().values.shape == (4, 5, 1)
    assert model.scalar_target_names == ["scalar"]
    assert model.spherical_target_names == ["quadrupole"]
    assert len(model.target_head_keys["quadrupole"]) == 1


def test_atomic_basis_target_uses_pet_densified_layout_and_single_head() -> None:
    dataset_info = _atomic_basis_dataset_info()
    target_info = dataset_info.targets["density"]
    model = EPET(_base_model_hypers(), dataset_info)
    dense_layout = densify_atomic_basis_target(target_info.layout, target_info.layout)

    assert model.target_head_keys["density"] == ["density"]
    assert set(model.block_to_head_key["density"].values()) == {"density"}
    assert model.key_labels["density"].names == ["o3_lambda", "o3_sigma"]
    assert torch.equal(model.key_labels["density"].values, dense_layout.keys.values)
    assert len(model.basis_calculators["density"]) == len(dense_layout)
    assert set(model.block_irrep_keys["density"].values()) == {"0,1", "1,1", "2,1"}
    assert all(
        "atom_type" not in dict_key
        for dict_key in model.basis_calculators["density"].keys()
    )


def test_atomic_basis_training_forward_returns_densified_blocks() -> None:
    dataset_info = _atomic_basis_dataset_info()
    target_info = dataset_info.targets["density"]
    dense_layout = densify_atomic_basis_target(target_info.layout, target_info.layout)
    model = EPET(_base_model_hypers(), dataset_info).train()
    system = _build_system(model)

    output = model([system], {"density": ModelOutput(per_atom=True)})["density"]

    assert output.keys.names == ["o3_lambda", "o3_sigma"]
    assert torch.equal(output.keys.values, dense_layout.keys.values)
    for key, block in output.items():
        dense_block = dense_layout.block(key)
        assert block.samples.names == ["system", "atom"]
        assert len(block.samples) == len(system)
        assert block.properties == dense_block.properties


def test_atomic_basis_eval_forward_sparsifies_public_layout() -> None:
    dataset_info = _atomic_basis_dataset_info()
    target_info = dataset_info.targets["density"]
    model = EPET(_base_model_hypers(), dataset_info).eval()
    system = _build_system(model)

    output = model([system], {"density": ModelOutput(per_atom=True)})["density"]

    assert output.keys.names == ["o3_lambda", "o3_sigma", "atom_type"]
    assert torch.equal(output.keys.values, target_info.layout.keys.values)
    for key, block in output.items():
        atom_type = int(key["atom_type"])
        assert torch.all(system.types[block.samples["atom"]] == atom_type)
        assert block.properties == target_info.layout.block(key).properties


def test_atomic_basis_loss_uses_pet_densified_targets() -> None:
    dataset_info = _atomic_basis_dataset_info()
    model = EPET(_base_model_hypers(), dataset_info).train()
    system = _build_system(model)
    predictions = model([system], {"density": ModelOutput(per_atom=True)})
    sparse_target = _atomic_basis_sparse_target(dataset_info.targets["density"], system)
    prepare_atomic_basis_targets, _ = get_prepare_atomic_basis_targets_transform(
        dataset_info.targets, {}
    )
    _, dense_targets, dense_extra = prepare_atomic_basis_targets(
        [system],
        {"density": sparse_target},
        _system_index_extra(),
    )
    loss = LossAggregator(
        dataset_info.targets,
        {
            "density": {
                "type": "mse",
                "weight": 1.0,
                "reduction": "mean",
                "gradients": {},
            }
        },
    )

    value = loss(predictions, dense_targets, dense_extra)

    assert torch.isfinite(value)


def test_coefficient_l2_exclusion_skips_spherical_l0_only_blocks() -> None:
    model = EPET(_base_model_hypers(), _atomic_basis_l0_only_dataset_info()).train()
    system = _build_system(model)

    model([system], {"density": ModelOutput(per_atom=True)})

    assert model.get_regularization_loss() > 0
    assert model.get_regularization_loss(exclude_spherical_l0=True).item() == 0.0


def test_coefficient_l2_exclusion_keeps_nontrivial_spherical_blocks() -> None:
    model = EPET(_base_model_hypers(), _atomic_basis_dataset_info()).train()
    system = _build_system(model)

    model([system], {"density": ModelOutput(per_atom=True)})

    assert model.get_regularization_loss(exclude_spherical_l0=True) > 0


def test_atomic_basis_rejects_irrep_head_groups() -> None:
    hypers = _base_model_hypers()
    hypers["irrep_head_groups"] = {"density": {"0,1": "head_a"}}

    with pytest.raises(
        ValueError, match="Atomic-basis targets cannot appear in irrep_head_groups"
    ):
        EPET(hypers, _atomic_basis_dataset_info())


def test_atomic_basis_rejects_shared_head_groups() -> None:
    hypers = _base_model_hypers()
    hypers["shared_head_groups"] = {"bad": ["density[0,1]"]}

    with pytest.raises(
        ValueError, match="Atomic-basis targets cannot appear in shared_head_groups"
    ):
        EPET(hypers, _atomic_basis_dataset_info())


def test_cartesian_rank2_target_uses_hidden_spherical_readout() -> None:
    model = EPET(_base_model_hypers(), _cartesian_rank2_dataset_info()).eval()
    system = _build_system(model)
    outputs = model([system], {"stress": ModelOutput(per_atom=False)})
    block = outputs["stress"].block()

    assert outputs.keys() == {"stress"}
    assert model.cartesian_rank2_target_names == ["stress"]
    assert model.spherical_target_names == ["stress"]
    assert set(model.block_irrep_keys["stress"].values()) == {"0,1", "2,1"}
    assert block.values.shape == (1, 3, 3, 1)
    assert block.components[0].names == ["xyz_1"]
    assert block.components[1].names == ["xyz_2"]
    assert torch.allclose(block.values, block.values.transpose(1, 2))


def test_cartesian_rank1_target_remains_unsupported() -> None:
    vector_target = get_generic_target_info(
        "vector",
        {
            "quantity": "vector",
            "unit": "",
            "type": {"cartesian": {"rank": 1}},
            "num_subtargets": 1,
            "per_atom": True,
        },
    )
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 8],
        targets={"vector": vector_target},
    )

    with pytest.raises(ValueError, match="rank-2"):
        EPET(_base_model_hypers(), dataset_info)


def test_cartesian_rank2_hardcoded_spherical_round_trip() -> None:
    layout = _cartesian_rank2_target_info().layout
    samples = Labels(["system"], torch.tensor([[0]], dtype=torch.int32))
    cartesian = torch.tensor(
        [
            [
                [[1.0], [2.0], [3.0]],
                [[2.0], [4.0], [5.0]],
                [[3.0], [5.0], [6.0]],
            ]
        ]
    )
    l0, l2 = _cartesian_rank2_to_spherical_components(cartesian)
    spherical_tensor_map = TensorMap(
        keys=_cartesian_rank2_to_spherical_components_target_keys(),
        blocks=[
            TensorBlock(
                values=l0,
                samples=samples,
                components=[Labels("o3_mu", torch.tensor([[0]]))],
                properties=layout.block().properties,
            ),
            TensorBlock(
                values=l2,
                samples=samples,
                components=[
                    Labels("o3_mu", torch.arange(-2, 3).reshape(-1, 1))
                ],
                properties=layout.block().properties,
            ),
        ],
    )

    reconstructed = _spherical_components_to_cartesian_rank2(
        spherical_tensor_map, layout
    ).block().values

    assert torch.allclose(reconstructed, cartesian, atol=1e-6)
    assert torch.allclose(reconstructed, reconstructed.transpose(1, 2))


def _cartesian_rank2_to_spherical_components_target_keys():
    return Labels(
        ["o3_lambda", "o3_sigma"],
        torch.tensor([[0, 1], [2, 1]], dtype=torch.int32),
    )


def test_cartesian_rank2_loss_uses_public_cartesian_tensor() -> None:
    dataset_info = _cartesian_rank2_dataset_info()
    model = EPET(_base_model_hypers(), dataset_info).eval()
    system = _build_system(model)
    predictions = model([system], {"stress": ModelOutput(per_atom=False)})
    target_values = torch.zeros_like(predictions["stress"].block().values)
    targets = {"stress": _single_block_like(predictions["stress"], target_values)}
    loss = LossAggregator(
        dataset_info.targets,
        {
            "stress": {
                "type": "invariant_huber",
                "weight": 5.0e-4,
                "reduction": "mean",
                "delta": 0.005,
                "gradients": {},
            }
        },
    )

    value = loss(predictions, targets)

    assert torch.isfinite(value)
    assert set(loss.losses) == {"stress"}


def test_cartesian_rank2_volume_normalization_matches_pet_helper() -> None:
    assert e_pet_model_module.normalize_by_volume is pet_model_module.normalize_by_volume

    raw_hypers = _base_model_hypers()
    normalized_hypers = _base_model_hypers()
    normalized_hypers["volume_normalized_targets"] = ["stress"]
    dataset_info = _cartesian_rank2_dataset_info()

    torch.manual_seed(23)
    raw_model = EPET(raw_hypers, dataset_info).eval()
    torch.manual_seed(23)
    normalized_model = EPET(normalized_hypers, dataset_info).eval()
    raw_system = _build_periodic_system(raw_model)
    normalized_system = _build_periodic_system(normalized_model)

    raw_output = raw_model([raw_system], {"stress": ModelOutput(per_atom=False)})[
        "stress"
    ].block()
    normalized_output = normalized_model(
        [normalized_system], {"stress": ModelOutput(per_atom=False)}
    )["stress"].block()

    assert torch.allclose(normalized_output.values, raw_output.values / 8.0)


def test_volume_normalization_does_not_scale_per_atom_outputs() -> None:
    hypers = _base_model_hypers()
    hypers["volume_normalized_targets"] = ["stress"]
    dataset_info = _cartesian_rank2_dataset_info(per_atom=True)

    torch.manual_seed(29)
    raw_model = EPET(_base_model_hypers(), dataset_info).eval()
    torch.manual_seed(29)
    normalized_model = EPET(hypers, dataset_info).eval()
    raw_system = _build_periodic_system(raw_model)
    normalized_system = _build_periodic_system(normalized_model)

    raw_values = raw_model([raw_system], {"stress": ModelOutput(per_atom=True)})[
        "stress"
    ].block().values
    normalized_values = normalized_model(
        [normalized_system], {"stress": ModelOutput(per_atom=True)}
    )["stress"].block().values

    assert torch.allclose(normalized_values, raw_values)


def test_e_pet_preserves_pet_diagnostic_feature_and_aux_outputs() -> None:
    model = EPET(_base_model_hypers(), _mixed_dataset_info()).eval()
    system = _build_system(model)
    outputs = model(
        [system],
        {
            "scalar": ModelOutput(per_atom=True),
            "quadrupole": ModelOutput(per_atom=True),
            "mtt::features::gnn_layers.0_node": ModelOutput(per_atom=True),
            "mtt::aux::scalar_last_layer_features": ModelOutput(per_atom=True),
        },
    )

    assert "mtt::features::gnn_layers.0_node" in outputs
    assert outputs["mtt::features::gnn_layers.0_node"].block().samples.names == [
        "system",
        "atom",
    ]
    assert outputs["mtt::aux::scalar_last_layer_features"].block().samples.names == [
        "system",
        "atom",
    ]
    assert outputs["mtt::aux::scalar_last_layer_features"].block().values.shape[0] == 4


def test_multi_irrep_target_uses_private_heads_by_default() -> None:
    model = EPET(_base_model_hypers(), _multi_irrep_dataset_info())

    assert len(model.target_head_keys["multi"]) == 3
    assert len(set(model.block_to_head_key["multi"].values())) == 3
    assert set(model.block_irrep_keys["multi"].values()) == {"1,1", "2,1", "3,1"}


def test_irrep_head_groups_share_heads_within_target_only() -> None:
    hypers = _base_model_hypers()
    hypers["irrep_head_groups"] = {
        "multi": {
            "1,1": "head_a",
            "2,1": "head_a",
        }
    }
    model = EPET(hypers, _multi_irrep_dataset_info())

    assert len(model.target_head_keys["multi"]) == 2
    block_to_head = model.block_to_head_key["multi"]
    head_by_irrep = {
        model.block_irrep_keys["multi"][block_key]: head_key
        for block_key, head_key in block_to_head.items()
    }
    assert head_by_irrep["1,1"] == head_by_irrep["2,1"]
    assert head_by_irrep["3,1"] != head_by_irrep["1,1"]


def test_same_irrep_targets_remain_independent() -> None:
    model = EPET(_base_model_hypers(), _same_irrep_two_targets_dataset_info())

    a_head = next(iter(model.block_to_head_key["forces_like_a"].values()))
    b_head = next(iter(model.block_to_head_key["forces_like_b"].values()))
    assert a_head != b_head


def test_shared_head_groups_share_heads_across_scalar_and_spherical_targets() -> None:
    hypers = _base_model_hypers()
    hypers["shared_head_groups"] = {
        "stress_head": ["scalar", "quadrupole[2,1]"]
    }
    model = EPET(hypers, _mixed_dataset_info())

    assert model.target_head_keys["scalar"] == ["shared__stress_head"]
    assert model.target_head_keys["quadrupole"] == ["shared__stress_head"]
    assert "shared__stress_head" in model.node_heads
    assert "shared__stress_head" in model.edge_heads
    assert "scalar" not in model.node_heads
    assert "scalar" not in model.edge_heads
    assert model.node_last_layers["scalar"] is not model.node_last_layers["quadrupole"]
    assert model.edge_last_layers["scalar"] is not model.edge_last_layers["quadrupole"]


def test_shared_head_groups_reject_scalar_irrep_selector() -> None:
    hypers = _base_model_hypers()
    hypers["shared_head_groups"] = {"bad": ["scalar[0,1]"]}

    with pytest.raises(
        ValueError, match="Scalar selectors cannot include an irrep suffix"
    ):
        EPET(hypers, _mixed_dataset_info())


def test_shared_head_groups_reject_missing_irrep_for_spherical_target() -> None:
    hypers = _base_model_hypers()
    hypers["shared_head_groups"] = {"bad": ["quadrupole"]}

    with pytest.raises(ValueError, match="must include an explicit irrep suffix"):
        EPET(hypers, _mixed_dataset_info())


def test_shared_head_groups_reject_overlap_with_irrep_head_groups() -> None:
    hypers = _base_model_hypers()
    hypers["irrep_head_groups"] = {"quadrupole": {"2,1": "local_head"}}
    hypers["shared_head_groups"] = {
        "stress_head": ["scalar", "quadrupole[2,1]"]
    }

    with pytest.raises(
        ValueError,
        match="cannot appear in both shared_head_groups and irrep_head_groups",
    ):
        EPET(hypers, _mixed_dataset_info())


def test_default_l1_path_uses_two_vector_basis_branches() -> None:
    dataset_info = _same_irrep_two_targets_dataset_info()
    model = EPET(_base_model_hypers(), dataset_info)

    block_key = next(iter(model.coefficient_shapes["forces_like_a"]))
    num_properties, basis_size = model.coefficient_shapes["forces_like_a"][block_key]
    assert num_properties == 1
    assert basis_size == 6


def test_irrep_head_groups_reject_unknown_target() -> None:
    hypers = _base_model_hypers()
    hypers["irrep_head_groups"] = {"ghost": {"1,1": "head_a"}}

    with pytest.raises(ValueError, match="Unknown targets in irrep_head_groups"):
        EPET(hypers, _mixed_dataset_info())


def test_irrep_head_groups_reject_unknown_irrep() -> None:
    hypers = _base_model_hypers()
    hypers["irrep_head_groups"] = {"quadrupole": {"1,1": "head_a"}}

    with pytest.raises(
        ValueError, match="Unknown irrep keys for target 'quadrupole'"
    ):
        EPET(hypers, _mixed_dataset_info())


def test_irrep_head_groups_reject_scalar_targets() -> None:
    hypers = _base_model_hypers()
    hypers["irrep_head_groups"] = {"scalar": {"0,1": "head_a"}}

    with pytest.raises(ValueError, match="Scalar targets cannot appear"):
        EPET(hypers, _mixed_dataset_info())
