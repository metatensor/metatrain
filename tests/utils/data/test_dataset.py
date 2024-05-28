from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from metatensor.models.utils.data import (
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
    assert target_info.gradients == []


def test_target_info():
    target_info = TargetInfo(
        quantity="energy", unit="kcal/mol", per_atom=True, gradients=["positions"]
    )

    assert target_info.quantity == "energy"
    assert target_info.unit == "kcal/mol"
    assert target_info.per_atom is True
    assert target_info.gradients == ["positions"]

    target_info.add_gradients(["positions", "strain"])
    assert target_info.gradients == ["strain", "positions"]

def test_unit_none_conversion():
    info = TargetInfo(quantity="energy", unit=None)
    assert info.unit == ""

def test_target_info_copy():
    pass


def test_target_info_update():
    info1 = TargetInfo(quantity="energy", unit="eV", gradients=["position"])
    info2 = TargetInfo(quantity="energy", unit="eV", gradients=["strain"])
    info1.update(info2)
    assert set(info1.gradients) == ["position", "strain"]


def test_target_info_union():
    info1 = TargetInfo(quantity="energy", unit="eV", gradients=["position"])
    info2 = TargetInfo(quantity="energy", unit="eV", gradients=["strain"])
    info_new = info1.union(info2)
    assert isinstance(info_new, TargetInfo)
    assert set(info_new.gradients) == ["position", "strain"]


def test_target_info_update_non_matching_quantity():
    info1 = TargetInfo(quantity="energy", unit="eV")
    info2 = TargetInfo(quantity="force", unit="eV")
    match = "Can't update DatasetInfo with a different `quantity`"
    with pytest.raises(ValueError, match=match):
        info1.update(info2)


def test_target_info_update_non_matching_unit():
    info1 = TargetInfo(quantity="energy", unit="eV")
    info2 = TargetInfo(quantity="energy", unit="kcal")
    match = "Can't update DatasetInfo with a different `unit`: "
    with pytest.raises(ValueError, match=match):
        info1.update(info2)


def test_target_info_update_non_matching_per_atom():
    info1 = TargetInfo(quantity="energy", unit="eV", per_atom=True)
    info2 = TargetInfo(quantity="enrgy", unit="eV", per_atom=False)
    match = "Can't update TargetInfo with a different `per_atom` property: "
    with pytest.raises(ValueError, match=match):
        info1.update(info2)


def test_target_info_dict_setitem_new_entry():
    tid = TargetInfoDict()
    info = TargetInfo(quantity="energy", unit="eV", gradients=["position"])
    tid["energy"] = info
    assert tid["energy"] == info


def test_target_info_dict_setitem_update_entry():
    tid = TargetInfoDict()
    info1 = TargetInfo(quantity="energy", unit="eV", gradients=["position"])
    info2 = TargetInfo(quantity="energy", unit="eV", gradients=["strain"])
    tid["energy"] = info1
    tid["energy"] = info2
    assert set(tid["energy"].gradients) == ["position", "strain"]


def test_target_info_dict_setitem_value_error():
    tid = TargetInfoDict()
    with pytest.raises(ValueError, match="value to set is not a `TargetInfo` instance"):
        tid["energy"] = "not a TargetInfo"

def test_target_info_dict_union():
    pass

def test_target_info_dict_intersection():
    # TODO test also magic function
    pass

def test_target_info_dict_difference():
    # TODO test also magic function
    pass

def test_dataset_info():
    """Tests the DatasetInfo class."""

    targets = TargetInfoDict(energy=TargetInfo(quantity="energy", unit="kcal/mol"))
    targets["mtm::U0"] = TargetInfo(quantity="energy", unit="kcal/mol")

    dataset_info = DatasetInfo(
        length_unit="angstrom", atomic_types={1, 2, 3}, targets=targets
    )

    assert dataset_info.length_unit == "angstrom"
    assert dataset_info.atomic_types == [1, 2, 3]
    assert dataset_info.targets["energy"].quantity == "energy"
    assert dataset_info.targets["energy"].unit == "kcal/mol"
    assert dataset_info.targets["mtm::U0"].quantity == "energy"
    assert dataset_info.targets["mtm::U0"].unit == "kcal/mol"

def test_length_unit_none_conversion():
    dataset_info = DatasetInfo(
        length_unit=None,
        atomic_types={1, 2, 3},
        targets=TargetInfoDict(energy=TargetInfo(quantity="energy", unit="kcal/mol")),
    )
    assert dataset_info.length_unit == ""

def test_dataset_info_copy():
    pass

def test_dataset_info_update_non_matching_length():
    pass

def test_dataset_info_update():
    pass

def test_dataset_info_union():
    """Tests the merge method."""

    old_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types={1, 6},
        targets={
            "energy": TargetInfo(quantity="energy", unit="eV"),
            "mtm::forces": TargetInfo(quantity="mtm::forces", unit="eV/Angstrom"),
        },
    )

    new_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types={1}
        targets={
            "energy": TargetInfo(quantity="energy", unit="eV"),
            "mtm::forces": TargetInfo(quantity="mtm::forces", unit="eV/Angstrom"),
            "mtm::stress": TargetInfo(quantity="mtm::stress", unit="GPa"),
        },
    )

    merged = old_info.union(new_info)

    assert merged.length_unit == "angstrom"
    assert merged.atomic_types == [1, 6]
    assert merged.targets["energy"].quantity == "energy"
    assert merged.targets["energy"].unit == "eV"
    assert merged.targets["mtm::forces"].quantity == "mtm::forces"
    assert merged.targets["mtm::forces"].unit == "eV/Angstrom"
    assert merged.targets["mtm::stress"].quantity == "mtm::stress"
    assert merged.targets["mtm::stress"].unit == "GPa"



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
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets = read_targets(OmegaConf.create(conf))
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
        "mtm::U0": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
            "file_format": ".xyz",
            "key": "U0",
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
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets = read_targets(OmegaConf.create(conf))
    targets_2 = read_targets(OmegaConf.create(conf_2))
    dataset = Dataset({"system": systems, **targets})
    dataset_2 = Dataset({"system": systems_2, **targets_2})
    assert get_atomic_types(dataset) == [1, 6, 7, 8]
    assert get_atomic_types(dataset_2) == [1, 6, 8]
    assert get_atomic_types([dataset, dataset_2]) == [1, 6, 7, 8]


def test_get_all_targets():
    """Tests that the target list is correctly computed with get_all_targets."""

    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")
    conf = {
        "mtm::U0": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
            "file_format": ".xyz",
            "key": "U0",
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
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets = read_targets(OmegaConf.create(conf))
    targets_2 = read_targets(OmegaConf.create(conf_2))
    dataset = Dataset({"system": systems, **targets})
    dataset_2 = Dataset({"system": systems_2, **targets_2})
    assert get_all_targets(dataset) == ["mtm::U0"]
    assert get_all_targets(dataset_2) == ["energy"]
    assert get_all_targets([dataset, dataset_2]) == ["energy", "mtm::U0"]


def test_check_datasets():
    """Tests the check_datasets function."""

    systems_qm9 = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")
    conf_qm9 = {
        "mtm::U0": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
            "file_format": ".xyz",
            "key": "U0",
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
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets_qm9 = read_targets(OmegaConf.create(conf_qm9))
    targets_ethanol = read_targets(OmegaConf.create(conf_ethanol))

    # everything ok
    training_set = Dataset({"system": systems_qm9, **targets_qm9})
    validation_set = Dataset({"system": systems_qm9, **targets_qm9})
    check_datasets([training_set], [validation_set])

    # extra species in validation dataset
    training_set = Dataset({"system": systems_ethanol, **targets_qm9})
    validation_set = Dataset({"system": systems_qm9, **targets_qm9})
    with pytest.raises(ValueError, match="The validation dataset has a species"):
        check_datasets([training_set], [validation_set])

    # extra targets in validation dataset
    training_set = Dataset({"system": systems_qm9, **targets_qm9})
    validation_set = Dataset({"system": systems_qm9, **targets_ethanol})
    with pytest.raises(ValueError, match="The validation dataset has a target"):
        check_datasets([training_set], [validation_set])

    # wrong dtype
    systems_qm9_64_bit = read_systems(
        RESOURCES_PATH / "qm9_reduced_100.xyz", dtype=torch.float64
    )
    training_set_64_bit = Dataset({"system": systems_qm9_64_bit, **targets_qm9})
    match = (
        "`dtype` between datasets is inconsistent, found torch.float32 and "
        "torch.float64 found in `validation_datasets`"
    )
    with pytest.raises(TypeError, match=match):
        check_datasets([training_set], [training_set_64_bit])

    match = (
        "`dtype` between datasets is inconsistent, found torch.float32 and "
        "torch.float64 found in `train_datasets`"
    )
    with pytest.raises(TypeError, match=match):
        check_datasets([training_set, training_set_64_bit], [validation_set])


def test_collate_fn():
    """Tests the collate_fn function."""

    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")
    conf = {
        "mtm::U0": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
            "file_format": ".xyz",
            "key": "U0",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets = read_targets(OmegaConf.create(conf))
    dataset = Dataset({"system": systems, "mtm::U0": targets["mtm::U0"]})

    batch = collate_fn([dataset[0], dataset[1], dataset[2]])

    assert len(batch) == 2
    assert isinstance(batch[0], tuple)
    assert len(batch[0]) == 3
    assert isinstance(batch[1], dict)
