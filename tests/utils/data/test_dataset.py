from pathlib import Path

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from omegaconf import OmegaConf

from metatrain.utils.data import (
    CollateFn,
    Dataset,
    DatasetInfo,
    TargetInfo,
    check_datasets,
    get_all_targets,
    get_atomic_types,
    get_stats,
    read_extra_data,
    read_systems,
    read_targets,
)


RESOURCES_PATH = Path(__file__).parents[2] / "resources"


@pytest.fixture
def layout_scalar():
    return TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.empty(0, 1),
                samples=Labels(
                    names=["system"],
                    values=torch.empty((0, 1), dtype=torch.int32),
                ),
                components=[],
                properties=Labels.range("energy", 1),
            )
        ],
    )


@pytest.fixture
def layout_spherical():
    return TensorMap(
        keys=Labels(
            names=["o3_lambda", "o3_sigma"],
            values=torch.tensor([[0, 1], [2, 1]]),
        ),
        blocks=[
            TensorBlock(
                values=torch.empty(0, 1, 1),
                samples=Labels(
                    names=["system"],
                    values=torch.empty((0, 1), dtype=torch.int32),
                ),
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.arange(0, 1, dtype=torch.int32).reshape(-1, 1),
                    ),
                ],
                properties=Labels.single(),
            ),
            TensorBlock(
                values=torch.empty(0, 5, 1),
                samples=Labels(
                    names=["system"],
                    values=torch.empty((0, 1), dtype=torch.int32),
                ),
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.arange(-2, 3, dtype=torch.int32).reshape(-1, 1),
                    ),
                ],
                properties=Labels.single(),
            ),
        ],
    )


@pytest.fixture
def layout_cartesian():
    return TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.empty(0, 3, 3, 1),
                samples=Labels(
                    names=["system"],
                    values=torch.empty((0, 1), dtype=torch.int32),
                ),
                components=[
                    Labels(
                        names=["xyz_1"],
                        values=torch.arange(0, 3, dtype=torch.int32).reshape(-1, 1),
                    ),
                    Labels(
                        names=["xyz_2"],
                        values=torch.arange(0, 3, dtype=torch.int32).reshape(-1, 1),
                    ),
                ],
                properties=Labels.single(),
            ),
        ],
    )


def test_target_info_scalar(layout_scalar):
    target_info = TargetInfo(quantity="energy", unit="kcal/mol", layout=layout_scalar)

    assert target_info.quantity == "energy"
    assert target_info.unit == "kcal/mol"
    assert target_info.gradients == []
    assert not target_info.per_atom

    expected_start = "TargetInfo(quantity='energy', unit='kcal/mol'"
    assert target_info.__repr__()[: len(expected_start)] == expected_start


def test_target_info_spherical(layout_spherical):
    target_info = TargetInfo(
        quantity="mtt::spherical", unit="kcal/mol", layout=layout_spherical
    )

    assert target_info.quantity == "mtt::spherical"
    assert target_info.unit == "kcal/mol"
    assert target_info.gradients == []
    assert not target_info.per_atom

    expected_start = "TargetInfo(quantity='mtt::spherical', unit='kcal/mol'"
    assert target_info.__repr__()[: len(expected_start)] == expected_start


def test_target_info_cartesian(layout_cartesian):
    target_info = TargetInfo(
        quantity="mtt::cartesian", unit="kcal/mol", layout=layout_cartesian
    )

    assert target_info.quantity == "mtt::cartesian"
    assert target_info.unit == "kcal/mol"
    assert target_info.gradients == []
    assert not target_info.per_atom

    expected_start = "TargetInfo(quantity='mtt::cartesian', unit='kcal/mol'"
    assert target_info.__repr__()[: len(expected_start)] == expected_start


def test_unit_none_conversion(layout_scalar):
    info = TargetInfo(quantity="energy", unit=None, layout=layout_scalar)
    assert info.unit == ""


def test_length_unit_none_conversion(layout_scalar):
    dataset_info = DatasetInfo(
        length_unit=None,
        atomic_types=[1, 2, 3],
        targets={
            "energy": TargetInfo(
                quantity="energy", unit="kcal/mol", layout=layout_scalar
            )
        },
    )
    assert dataset_info.length_unit == ""


def test_target_info_eq(layout_scalar):
    info1 = TargetInfo(quantity="energy", unit="kcal/mol", layout=layout_scalar)
    info2 = TargetInfo(quantity="energy", unit="eV", layout=layout_scalar)

    assert info1 == info1
    assert info1 != info2


def test_target_info_eq_other_objects(layout_scalar):
    info = TargetInfo(quantity="energy", unit="eV", layout=layout_scalar)

    assert not info == [1, 2, 3]


def test_dataset_info(layout_scalar):
    """Tests the DatasetInfo class."""

    targets = {
        "energy": TargetInfo(quantity="energy", unit="kcal/mol", layout=layout_scalar)
    }
    targets["mtt::U0"] = TargetInfo(
        quantity="energy", unit="kcal/mol", layout=layout_scalar
    )

    dataset_info = DatasetInfo(
        length_unit="angstrom", atomic_types=[3, 1, 2], targets=targets
    )

    assert dataset_info.length_unit == "angstrom"
    assert dataset_info.atomic_types == [1, 2, 3]
    assert dataset_info.targets["energy"].quantity == "energy"
    assert dataset_info.targets["energy"].unit == "kcal/mol"
    assert dataset_info.targets["mtt::U0"].quantity == "energy"
    assert dataset_info.targets["mtt::U0"].unit == "kcal/mol"
    assert dataset_info.device == layout_scalar.device

    expected = (
        "DatasetInfo(length_unit='angstrom', atomic_types=[1, 2, 3], "
        f"targets={targets})"
    )
    assert dataset_info.__repr__() == expected


def test_set_atomic_types(layout_scalar):
    targets = {
        "energy": TargetInfo(quantity="energy", unit="kcal/mol", layout=layout_scalar)
    }
    targets["mtt::U0"] = TargetInfo(
        quantity="energy", unit="kcal/mol", layout=layout_scalar
    )

    dataset_info = DatasetInfo(
        length_unit="angstrom", atomic_types=[3, 1, 2], targets=targets
    )

    dataset_info.atomic_types = [5, 4, 1]
    assert dataset_info.atomic_types == [1, 4, 5]

    dataset_info.atomic_types += [7, 1]
    assert dataset_info.atomic_types == [1, 4, 5, 7]


def test_dataset_info_copy(layout_scalar, layout_cartesian):
    targets = {}
    targets["energy"] = TargetInfo(quantity="energy", unit="eV", layout=layout_scalar)
    targets["mtt::my-target"] = TargetInfo(
        quantity="mtt::my-target", unit="eV/Angstrom", layout=layout_cartesian
    )
    info = DatasetInfo(length_unit="angstrom", atomic_types=[1, 6], targets=targets)

    copy = info.copy()

    assert copy == info
    assert copy is not info


def test_dataset_info_update(layout_scalar, layout_spherical):
    targets = {}
    targets["energy"] = TargetInfo(quantity="energy", unit="eV", layout=layout_scalar)

    info = DatasetInfo(length_unit="angstrom", atomic_types=[1, 6], targets=targets)

    targets2 = targets.copy()
    targets2["mtt::my-target"] = TargetInfo(
        quantity="mtt::my-target", unit="eV/Angstrom", layout=layout_spherical
    )

    info2 = DatasetInfo(length_unit="angstrom", atomic_types=[8], targets=targets2)
    info.update(info2)

    assert info.atomic_types == [1, 6, 8]
    assert info.targets["energy"] == targets["energy"]
    assert info.targets["mtt::my-target"] == targets2["mtt::my-target"]


def test_dataset_info_update_non_matching_length_unit(layout_scalar):
    targets = {}
    targets["energy"] = TargetInfo(quantity="energy", unit="eV", layout=layout_scalar)

    info = DatasetInfo(length_unit="angstrom", atomic_types=[1, 6], targets=targets)

    targets2 = targets.copy()
    targets2["mtt::my-target"] = TargetInfo(
        quantity="mtt::my-target", unit="eV/Angstrom", layout=layout_scalar
    )

    info2 = DatasetInfo(length_unit="nanometer", atomic_types=[8], targets=targets2)

    match = (
        r"Can't update DatasetInfo with a different `length_unit`: "
        r"\('angstrom' != 'nanometer'\)"
    )

    with pytest.raises(ValueError, match=match):
        info.update(info2)


def test_dataset_info_eq(layout_scalar):
    targets = {}
    targets["energy"] = TargetInfo(quantity="energy", unit="eV", layout=layout_scalar)

    info = DatasetInfo(length_unit="angstrom", atomic_types=[1, 6], targets=targets)

    targets2 = targets.copy()
    targets2["my-target"] = TargetInfo(
        quantity="mtt::my-target", unit="eV/Angstrom", layout=layout_scalar
    )
    info2 = DatasetInfo(length_unit="nanometer", atomic_types=[8], targets=targets2)

    assert info == info
    assert info != info2


def test_dataset_info_eq_other_objects(layout_scalar):
    targets = {}
    targets["energy"] = TargetInfo(quantity="energy", unit="eV", layout=layout_scalar)

    info = DatasetInfo(length_unit="angstrom", atomic_types=[1, 6], targets=targets)

    assert not info == [1, 2, 3]


def test_dataset_info_update_different_target_info(layout_scalar):
    targets = {}
    targets["energy"] = TargetInfo(quantity="energy", unit="eV", layout=layout_scalar)

    info = DatasetInfo(length_unit="angstrom", atomic_types=[1, 6], targets=targets)

    targets2 = {}
    targets2["energy"] = TargetInfo(
        quantity="energy", unit="eV/Angstrom", layout=layout_scalar
    )

    info2 = DatasetInfo(length_unit="angstrom", atomic_types=[8], targets=targets2)

    match = (
        "Can't update DatasetInfo with different target information for target 'energy'"
    )
    with pytest.raises(ValueError, match=match):
        info.update(info2)


def test_dataset_info_union(layout_scalar, layout_cartesian):
    """Tests the union method."""
    targets = {}
    targets["energy"] = TargetInfo(quantity="energy", unit="eV", layout=layout_scalar)
    targets["forces"] = TargetInfo(
        quantity="mtt::forces", unit="eV/Angstrom", layout=layout_scalar
    )
    info = DatasetInfo(length_unit="angstrom", atomic_types=[1, 6], targets=targets)

    other_targets = targets.copy()
    other_targets["mtt::stress"] = TargetInfo(
        quantity="mtt::stress", unit="GPa", layout=layout_cartesian
    )

    other_info = DatasetInfo(
        length_unit="angstrom", atomic_types=[1], targets=other_targets
    )

    union = info.union(other_info)

    assert union.length_unit == "angstrom"
    assert union.atomic_types == [1, 6]
    assert union.targets == other_targets


def test_dataset_info_no_targets():
    """Tests the properties of a DatasetInfo that has no targets."""
    dataset_info = DatasetInfo(
        length_unit="angstrom", atomic_types=[1, 2, 3], targets={}
    )

    assert dataset_info.device is None
    # Setting the device should not fail:
    dataset_info.to(device="cpu")


def test_dataset():
    """Tests the readers and the dataset class."""

    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")

    filename = str(RESOURCES_PATH / "qm9_reduced_100.xyz")
    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": filename,
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, _ = read_targets(OmegaConf.create(conf))
    dataset = Dataset.from_dict({"system": systems, "energy": targets["energy"]})
    collate_fn = CollateFn(target_keys=["energy"])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=10, collate_fn=collate_fn
    )

    for batch in dataloader:
        assert batch[1]["energy"].block().values.shape == (10, 1)


def test_get_atomic_types():
    """Tests that the species list is correctly computed with get_atomic_types."""

    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")
    conf = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    systems_2 = read_systems(RESOURCES_PATH / "ethanol_reduced_100.xyz")
    conf_2 = {
        "energy": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "ethanol_reduced_100.xyz"),
            "reader": "ase",
            "key": "energy",
            "unit": "eV",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, _ = read_targets(OmegaConf.create(conf))
    targets_2, _ = read_targets(OmegaConf.create(conf_2))
    dataset = Dataset.from_dict({"system": systems, **targets})
    dataset_2 = Dataset.from_dict({"system": systems_2, **targets_2})

    assert get_atomic_types(dataset) == [1, 6, 7, 8]
    assert get_atomic_types(dataset_2) == [1, 6, 8]
    assert get_atomic_types([dataset, dataset_2]) == [1, 6, 7, 8]


def test_get_all_targets():
    """Tests that the target list is correctly computed with get_all_targets."""

    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")
    conf = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    systems_2 = read_systems(RESOURCES_PATH / "ethanol_reduced_100.xyz")
    conf_2 = {
        "energy": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "ethanol_reduced_100.xyz"),
            "reader": "ase",
            "key": "energy",
            "unit": "eV",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, _ = read_targets(OmegaConf.create(conf))
    targets_2, _ = read_targets(OmegaConf.create(conf_2))
    dataset = Dataset.from_dict({"system": systems, **targets})
    dataset_2 = Dataset.from_dict({"system": systems_2, **targets_2})
    assert get_all_targets(dataset) == ["mtt::U0"]
    assert get_all_targets(dataset_2) == ["energy"]
    assert get_all_targets([dataset, dataset_2]) == ["energy", "mtt::U0"]


def test_check_datasets():
    """Tests the check_datasets function."""

    systems_qm9 = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")
    conf_qm9 = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    systems_ethanol = read_systems(RESOURCES_PATH / "ethanol_reduced_100.xyz")
    conf_ethanol = {
        "energy": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "ethanol_reduced_100.xyz"),
            "reader": "ase",
            "key": "energy",
            "unit": "eV",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets_qm9, _ = read_targets(OmegaConf.create(conf_qm9))
    targets_ethanol, _ = read_targets(OmegaConf.create(conf_ethanol))

    # everything ok
    train_set = Dataset.from_dict({"system": systems_qm9, **targets_qm9})
    val_set = Dataset.from_dict({"system": systems_qm9, **targets_qm9})
    check_datasets([train_set], [val_set])

    # extra species in validation dataset
    train_set = Dataset.from_dict({"system": systems_ethanol, **targets_qm9})
    val_set = Dataset.from_dict({"system": systems_qm9, **targets_qm9})
    with pytest.raises(ValueError, match="The validation dataset has a species"):
        check_datasets([train_set], [val_set])

    # extra targets in validation dataset
    train_set = Dataset.from_dict({"system": systems_qm9, **targets_qm9})
    val_set = Dataset.from_dict({"system": systems_qm9, **targets_ethanol})
    with pytest.raises(ValueError, match="The validation dataset has a target"):
        check_datasets([train_set], [val_set])

    # wrong dtype
    systems_qm9_32bit = [system.to(dtype=torch.float32) for system in systems_qm9]
    targets_qm9_32bit = {
        name: [tensor.to(dtype=torch.float32) for tensor in values]
        for name, values in targets_qm9.items()
    }
    train_set_32_bit = Dataset.from_dict(
        {"system": systems_qm9_32bit, **targets_qm9_32bit}
    )

    match = (
        "`dtype` between datasets is inconsistent, found torch.float64 and "
        "torch.float32 in validation datasets"
    )
    with pytest.raises(TypeError, match=match):
        check_datasets([train_set], [train_set_32_bit])

    match = (
        "`dtype` between datasets is inconsistent, found torch.float64 and "
        "torch.float32 in training datasets"
    )
    with pytest.raises(TypeError, match=match):
        check_datasets([train_set, train_set_32_bit], [])


def test_collate_fn():
    """Tests the collate_fn function."""

    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")
    conf_targets = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, _ = read_targets(OmegaConf.create(conf_targets))

    conf_extra_data = {
        "U0": {
            "quantity": "",
            "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
        }
    }
    extra_data, _ = read_extra_data(OmegaConf.create(conf_extra_data))

    dataset = Dataset.from_dict(
        {
            "system": systems,
            "mtt::U0": targets["mtt::U0"],
            "U0": extra_data["U0"],
        }
    )

    collate_fn = CollateFn(target_keys=["mtt::U0"])
    batch = collate_fn([dataset[0], dataset[1], dataset[2]])

    assert len(batch) == 3
    assert isinstance(batch[0], tuple)
    assert len(batch[0]) == 3
    assert isinstance(batch[1], dict)
    assert isinstance(batch[2], dict)


def test_get_stats(layout_scalar):
    """Tests the get_stats method of Dataset and Subset."""

    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")
    conf = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    systems_2 = read_systems(RESOURCES_PATH / "ethanol_reduced_100.xyz")
    conf_2 = {
        "energy": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "ethanol_reduced_100.xyz"),
            "reader": "ase",
            "key": "energy",
            "unit": "eV",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, _ = read_targets(OmegaConf.create(conf))
    targets_2, _ = read_targets(OmegaConf.create(conf_2))
    dataset = Dataset.from_dict({"system": systems, **targets})
    dataset_2 = Dataset.from_dict({"system": systems_2, **targets_2})

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 6],
        targets={
            "mtt::U0": TargetInfo(quantity="energy", unit="eV", layout=layout_scalar),
            "energy": TargetInfo(quantity="energy", unit="eV", layout=layout_scalar),
        },
    )

    stats = get_stats(dataset, dataset_info)
    stats_2 = get_stats(dataset_2, dataset_info)

    assert "100 structures" in stats
    assert "mtt::U0" in stats
    assert "energy" in stats_2
    assert "mean " in stats
    assert "std " in stats
    assert "mean " in stats_2
    assert "std " in stats_2
    assert "stress" not in stats_2
    assert "eV" in stats
    assert "eV" in stats_2


def test_instance_torchscript_compatible(layout_scalar):
    dataset_info = DatasetInfo(
        length_unit=None,
        atomic_types=[1, 2, 3],
        targets={
            "energy": TargetInfo(
                quantity="energy", unit="kcal/mol", layout=layout_scalar
            )
        },
    )

    torch.jit.script(dataset_info)


def test_class_torchscript_compatible():
    torch.jit.script(DatasetInfo)
