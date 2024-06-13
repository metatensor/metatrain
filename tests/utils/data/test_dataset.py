from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from metatrain.utils.data import (
    Dataset,
    DatasetInfo,
    TargetInfo,
    TargetInfoDict,
    check_datasets,
    collate_fn,
    get_all_targets,
    get_atomic_types,
    read_systems,
    read_targets,
)


RESOURCES_PATH = Path(__file__).parents[2] / "resources"


def test_target_info_default():
    target_info = TargetInfo(quantity="energy", unit="kcal/mol")

    assert target_info.quantity == "energy"
    assert target_info.unit == "kcal/mol"
    assert target_info.per_atom is False
    assert target_info.gradients == set()


def test_target_info_gradients():
    target_info = TargetInfo(
        quantity="energy", unit="kcal/mol", per_atom=True, gradients=["positions"]
    )

    assert target_info.quantity == "energy"
    assert target_info.unit == "kcal/mol"
    assert target_info.per_atom is True
    assert target_info.gradients == {"positions"}


def test_unit_none_conversion():
    info = TargetInfo(quantity="energy", unit=None)
    assert info.unit == ""


def test_target_info_copy():
    info = TargetInfo(quantity="energy", unit="eV", gradients={"positions"})
    copy = info.copy()
    assert copy == info
    assert copy is not info


def test_target_info_update():
    info1 = TargetInfo(quantity="energy", unit="eV", gradients={"position"})
    info2 = TargetInfo(quantity="energy", unit="eV", gradients={"strain"})
    info1.update(info2)
    assert set(info1.gradients) == {"position", "strain"}


def test_target_info_union():
    info1 = TargetInfo(quantity="energy", unit="eV", gradients={"position"})
    info2 = TargetInfo(quantity="energy", unit="eV", gradients={"strain"})
    info_new = info1.union(info2)
    assert isinstance(info_new, TargetInfo)
    assert set(info_new.gradients) == {"position", "strain"}


def test_target_info_update_non_matching_quantity():
    info1 = TargetInfo(quantity="energy", unit="eV")
    info2 = TargetInfo(quantity="force", unit="eV")
    match = r"Can't update TargetInfo with a different `quantity`: \(energy != force\)"
    with pytest.raises(ValueError, match=match):
        info1.update(info2)


def test_target_info_update_non_matching_unit():
    info1 = TargetInfo(quantity="energy", unit="eV")
    info2 = TargetInfo(quantity="energy", unit="kcal")
    match = r"Can't update TargetInfo with a different `unit`: \(eV != kcal\)"
    with pytest.raises(ValueError, match=match):
        info1.update(info2)


def test_target_info_update_non_matching_per_atom():
    info1 = TargetInfo(quantity="energy", unit="eV", per_atom=True)
    info2 = TargetInfo(quantity="energy", unit="eV", per_atom=False)
    match = "Can't update TargetInfo with a different `per_atom` property: "
    with pytest.raises(ValueError, match=match):
        info1.update(info2)


def test_target_info_dict_setitem_new_entry():
    tid = TargetInfoDict()
    info = TargetInfo(quantity="energy", unit="eV", gradients={"position"})
    tid["energy"] = info
    assert tid["energy"] == info


def test_target_info_dict_setitem_update_entry():
    tid = TargetInfoDict()
    info1 = TargetInfo(quantity="energy", unit="eV", gradients={"position"})
    info2 = TargetInfo(quantity="energy", unit="eV", gradients={"strain"})
    tid["energy"] = info1
    tid["energy"] = info2
    assert set(tid["energy"].gradients) == {"position", "strain"}


def test_target_info_dict_setitem_value_error():
    tid = TargetInfoDict()
    with pytest.raises(ValueError, match="value to set is not a `TargetInfo` instance"):
        tid["energy"] = "not a TargetInfo"


def test_target_info_dict_union():
    tid1 = TargetInfoDict()
    tid1["energy"] = TargetInfo(quantity="energy", unit="eV", gradients={"position"})

    tid2 = TargetInfoDict()
    tid2["myenergy"] = TargetInfo(quantity="energy", unit="eV", gradients={"strain"})

    merged = tid1.union(tid2)
    assert merged["energy"] == tid1["energy"]
    assert merged["myenergy"] == tid2["myenergy"]


def test_target_info_dict_merge_error():
    tid1 = TargetInfoDict()
    tid1["energy"] = TargetInfo(quantity="energy", unit="eV", gradients={"position"})

    tid2 = TargetInfoDict()
    tid2["energy"] = TargetInfo(
        quantity="energy", unit="kcal/mol", gradients={"strain"}
    )

    match = r"Can't update TargetInfo with a different `unit`: \(eV != kcal/mol\)"
    with pytest.raises(ValueError, match=match):
        tid1.union(tid2)


def test_target_info_dict_intersection():
    tid1 = TargetInfoDict()
    tid1["energy"] = TargetInfo(quantity="energy", unit="eV", gradients={"position"})
    tid1["myenergy"] = TargetInfo(quantity="energy", unit="eV", gradients={"strain"})

    tid2 = TargetInfoDict()
    tid2["myenergy"] = TargetInfo(quantity="energy", unit="eV", gradients={"strain"})

    intersection = tid1.intersection(tid2)
    assert len(intersection) == 1
    assert intersection["myenergy"] == tid1["myenergy"]

    # Test `&` operator
    intersection_and = tid1 & tid2
    assert intersection_and == intersection


def test_target_info_dict_intersection_error():
    tid1 = TargetInfoDict()
    tid1["energy"] = TargetInfo(quantity="energy", unit="eV", gradients={"position"})
    tid1["myenergy"] = TargetInfo(quantity="energy", unit="eV", gradients={"strain"})

    tid2 = TargetInfoDict()
    tid2["myenergy"] = TargetInfo(
        quantity="energy", unit="kcal/mol", gradients={"strain"}
    )

    match = (
        r"Intersected items with the same key are not the same. Intersected "
        r"keys are myenergy"
    )

    with pytest.raises(ValueError, match=match):
        tid1.intersection(tid2)


def test_target_info_dict_difference():
    # TODO test `-` operator
    tid1 = TargetInfoDict()
    tid1["energy"] = TargetInfo(quantity="energy", unit="eV", gradients={"position"})
    tid1["myenergy"] = TargetInfo(quantity="energy", unit="eV", gradients={"strain"})

    tid2 = TargetInfoDict()
    tid2["myenergy"] = TargetInfo(
        quantity="energy", unit="kcal/mol", gradients={"strain"}
    )

    difference = tid1.difference(tid2)
    assert len(difference) == 1
    assert difference["energy"] == tid1["energy"]

    difference_sub = tid1 - tid2
    assert difference_sub == difference


def test_dataset_info():
    """Tests the DatasetInfo class."""

    targets = TargetInfoDict(energy=TargetInfo(quantity="energy", unit="kcal/mol"))
    targets["mtt::U0"] = TargetInfo(quantity="energy", unit="kcal/mol")

    dataset_info = DatasetInfo(
        length_unit="angstrom", atomic_types={1, 2, 3}, targets=targets
    )

    assert dataset_info.length_unit == "angstrom"
    assert dataset_info.atomic_types == {1, 2, 3}
    assert dataset_info.targets["energy"].quantity == "energy"
    assert dataset_info.targets["energy"].unit == "kcal/mol"
    assert dataset_info.targets["mtt::U0"].quantity == "energy"
    assert dataset_info.targets["mtt::U0"].unit == "kcal/mol"


def test_length_unit_none_conversion():
    dataset_info = DatasetInfo(
        length_unit=None,
        atomic_types={1, 2, 3},
        targets=TargetInfoDict(energy=TargetInfo(quantity="energy", unit="kcal/mol")),
    )
    assert dataset_info.length_unit == ""


def test_dataset_info_copy():
    targets = TargetInfoDict()
    targets["energy"] = TargetInfo(quantity="energy", unit="eV")
    targets["forces"] = TargetInfo(quantity="mtt::forces", unit="eV/Angstrom")
    info = DatasetInfo(length_unit="angstrom", atomic_types={1, 6}, targets=targets)

    copy = info.copy()

    assert copy == info
    assert copy is not info


def test_dataset_info_update():
    targets = TargetInfoDict()
    targets["energy"] = TargetInfo(quantity="energy", unit="eV")

    info = DatasetInfo(length_unit="angstrom", atomic_types={1, 6}, targets=targets)

    targets2 = targets.copy()
    targets2["forces"] = TargetInfo(quantity="mtt::forces", unit="eV/Angstrom")

    info2 = DatasetInfo(length_unit="angstrom", atomic_types={8}, targets=targets2)
    info.update(info2)

    assert info.atomic_types == {1, 6, 8}
    assert info.targets["energy"] == targets["energy"]
    assert info.targets["forces"] == targets2["forces"]


def test_dataset_info_update_non_matching_length_unit():
    targets = TargetInfoDict()
    targets["energy"] = TargetInfo(quantity="energy", unit="eV")

    info = DatasetInfo(length_unit="angstrom", atomic_types={1, 6}, targets=targets)

    targets2 = targets.copy()
    targets2["forces"] = TargetInfo(quantity="mtt::forces", unit="eV/Angstrom")

    info2 = DatasetInfo(length_unit="nanometer", atomic_types={8}, targets=targets2)

    match = (
        r"Can't update DatasetInfo with a different `length_unit`: "
        r"\(angstrom != nanometer\)"
    )

    with pytest.raises(ValueError, match=match):
        info.update(info2)


def test_dataset_info_update_different_target_info():
    targets = TargetInfoDict()
    targets["energy"] = TargetInfo(quantity="energy", unit="eV")

    info = DatasetInfo(length_unit="angstrom", atomic_types={1, 6}, targets=targets)

    targets2 = TargetInfoDict()
    targets2["energy"] = TargetInfo(quantity="energy", unit="eV/Angstrom")

    info2 = DatasetInfo(length_unit="angstrom", atomic_types={8}, targets=targets2)

    match = r"Can't update TargetInfo with a different `unit`: \(eV != eV/Angstrom\)"
    with pytest.raises(ValueError, match=match):
        info.update(info2)


def test_dataset_info_union():
    """Tests the union method."""
    targets = TargetInfoDict()
    targets["energy"] = TargetInfo(quantity="energy", unit="eV")
    targets["forces"] = TargetInfo(quantity="mtt::forces", unit="eV/Angstrom")
    info = DatasetInfo(length_unit="angstrom", atomic_types={1, 6}, targets=targets)

    other_targets = targets.copy()
    other_targets["mtt::stress"] = TargetInfo(quantity="mtt::stress", unit="GPa")

    other_info = DatasetInfo(
        length_unit="angstrom", atomic_types={1}, targets=other_targets
    )

    union = info.union(other_info)

    assert union.length_unit == "angstrom"
    assert union.atomic_types == {1, 6}
    assert union.targets == other_targets


def test_dataset():
    """Tests the readers and the dataset class."""

    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")

    filename = str(RESOURCES_PATH / "qm9_reduced_100.xyz")
    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": filename,
            "file_format": ".xyz",
            "key": "U0",
            "unit": "eV",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, _ = read_targets(OmegaConf.create(conf))
    dataset = Dataset({"system": systems, "energy": targets["energy"]})
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
            "file_format": ".xyz",
            "key": "U0",
            "unit": "eV",
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
            "file_format": ".xyz",
            "key": "energy",
            "unit": "eV",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, _ = read_targets(OmegaConf.create(conf))
    targets_2, _ = read_targets(OmegaConf.create(conf_2))
    dataset = Dataset({"system": systems, **targets})
    dataset_2 = Dataset({"system": systems_2, **targets_2})

    assert get_atomic_types(dataset) == {1, 6, 7, 8}
    assert get_atomic_types(dataset_2) == {1, 6, 8}
    assert get_atomic_types([dataset, dataset_2]) == {1, 6, 7, 8}


def test_get_all_targets():
    """Tests that the target list is correctly computed with get_all_targets."""

    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")
    conf = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
            "file_format": ".xyz",
            "key": "U0",
            "unit": "eV",
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
            "file_format": ".xyz",
            "key": "energy",
            "unit": "eV",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, _ = read_targets(OmegaConf.create(conf))
    targets_2, _ = read_targets(OmegaConf.create(conf_2))
    dataset = Dataset({"system": systems, **targets})
    dataset_2 = Dataset({"system": systems_2, **targets_2})
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
            "file_format": ".xyz",
            "key": "U0",
            "unit": "eV",
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
            "file_format": ".xyz",
            "key": "energy",
            "unit": "eV",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets_qm9, _ = read_targets(OmegaConf.create(conf_qm9))
    targets_ethanol, _ = read_targets(OmegaConf.create(conf_ethanol))

    # everything ok
    train_set = Dataset({"system": systems_qm9, **targets_qm9})
    val_set = Dataset({"system": systems_qm9, **targets_qm9})
    check_datasets([train_set], [val_set])

    # extra species in validation dataset
    train_set = Dataset({"system": systems_ethanol, **targets_qm9})
    val_set = Dataset({"system": systems_qm9, **targets_qm9})
    with pytest.raises(ValueError, match="The validation dataset has a species"):
        check_datasets([train_set], [val_set])

    # extra targets in validation dataset
    train_set = Dataset({"system": systems_qm9, **targets_qm9})
    val_set = Dataset({"system": systems_qm9, **targets_ethanol})
    with pytest.raises(ValueError, match="The validation dataset has a target"):
        check_datasets([train_set], [val_set])

    # wrong dtype
    systems_qm9_64_bit = read_systems(
        RESOURCES_PATH / "qm9_reduced_100.xyz", dtype=torch.float64
    )
    train_set_64_bit = Dataset({"system": systems_qm9_64_bit, **targets_qm9})
    match = (
        "`dtype` between datasets is inconsistent, found torch.float32 and "
        "torch.float64 found in `val_datasets`"
    )
    with pytest.raises(TypeError, match=match):
        check_datasets([train_set], [train_set_64_bit])

    match = (
        "`dtype` between datasets is inconsistent, found torch.float32 and "
        "torch.float64 found in `train_datasets`"
    )
    with pytest.raises(TypeError, match=match):
        check_datasets([train_set, train_set_64_bit], [val_set])


def test_collate_fn():
    """Tests the collate_fn function."""

    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")
    conf = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
            "file_format": ".xyz",
            "key": "U0",
            "unit": "eV",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, _ = read_targets(OmegaConf.create(conf))
    dataset = Dataset({"system": systems, "mtt::U0": targets["mtt::U0"]})

    batch = collate_fn([dataset[0], dataset[1], dataset[2]])

    assert len(batch) == 2
    assert isinstance(batch[0], tuple)
    assert len(batch[0]) == 3
    assert isinstance(batch[1], dict)


def test_get_stats():
    """Tests the get_stats method of Dataset and Subset."""

    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")
    conf = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
            "file_format": ".xyz",
            "key": "U0",
            "unit": "eV",
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
            "file_format": ".xyz",
            "key": "energy",
            "unit": "eV",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, _ = read_targets(OmegaConf.create(conf))
    targets_2, _ = read_targets(OmegaConf.create(conf_2))
    dataset = Dataset({"system": systems, **targets})
    dataset_2 = Dataset({"system": systems_2, **targets_2})

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types={1, 6},
        targets={
            "mtt::U0": TargetInfo(quantity="energy", unit="eV"),
            "energy": TargetInfo(quantity="energy", unit="eV"),
        },
    )

    stats = dataset.get_stats(dataset_info)
    stats_2 = dataset_2.get_stats(dataset_info)

    assert "size 100" in stats
    assert "mtt::U0" in stats
    assert "energy" in stats_2
    assert "mean=" in stats
    assert "std=" in stats
    assert "mean=" in stats_2
    assert "std=" in stats_2
    assert "stress" not in stats_2
