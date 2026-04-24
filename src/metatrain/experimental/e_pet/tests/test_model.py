from __future__ import annotations

import copy

import torch
from metatomic.torch import ModelOutput, System

from metatrain.pet.model import PET
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from metatrain.experimental.e_pet.model import EPET


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
    hypers["tensor_basis_defaults"]["l1_species_dependent_vector_soap"] = copy.deepcopy(
        hypers["tensor_basis_defaults"]["soap"]
    )
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


def _scalar_dataset_info() -> DatasetInfo:
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 8],
        targets={"scalar": _scalar_target_info()},
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
    try:
        EPET(hypers, _mixed_dataset_info())
    except ValueError as exc:
        assert "Scalar selectors cannot include an irrep suffix" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for scalar selector with irrep suffix."
        )


def test_shared_head_groups_reject_missing_irrep_for_spherical_target() -> None:
    hypers = _base_model_hypers()
    hypers["shared_head_groups"] = {"bad": ["quadrupole"]}
    try:
        EPET(hypers, _mixed_dataset_info())
    except ValueError as exc:
        assert "must include an explicit irrep suffix" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for spherical selector without irrep suffix."
        )


def test_shared_head_groups_reject_overlap_with_irrep_head_groups() -> None:
    hypers = _base_model_hypers()
    hypers["irrep_head_groups"] = {"quadrupole": {"2,1": "local_head"}}
    hypers["shared_head_groups"] = {
        "stress_head": ["scalar", "quadrupole[2,1]"]
    }
    try:
        EPET(hypers, _mixed_dataset_info())
    except ValueError as exc:
        assert "cannot appear in both shared_head_groups and irrep_head_groups" in str(
            exc
        )
    else:
        raise AssertionError(
            "Expected ValueError for overlap between shared_head_groups and "
            "irrep_head_groups."
        )


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
    try:
        EPET(hypers, _mixed_dataset_info())
    except ValueError as exc:
        assert "Unknown targets in irrep_head_groups" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown target.")


def test_irrep_head_groups_reject_unknown_irrep() -> None:
    hypers = _base_model_hypers()
    hypers["irrep_head_groups"] = {"quadrupole": {"1,1": "head_a"}}
    try:
        EPET(hypers, _mixed_dataset_info())
    except ValueError as exc:
        assert "Unknown irrep keys for target 'quadrupole'" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown irrep.")


def test_irrep_head_groups_reject_scalar_targets() -> None:
    hypers = _base_model_hypers()
    hypers["irrep_head_groups"] = {"scalar": {"0,1": "head_a"}}
    try:
        EPET(hypers, _mixed_dataset_info())
    except ValueError as exc:
        assert "Scalar targets cannot appear in irrep_head_groups" in str(exc)
    else:
        raise AssertionError("Expected ValueError for scalar target in irrep groups.")
