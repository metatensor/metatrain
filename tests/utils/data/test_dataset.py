from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from metatensor.models.utils.data import (
    Dataset,
    DatasetInfo,
    TargetInfo,
    check_datasets,
    collate_fn,
    get_all_targets,
    get_atomic_types,
    merge_dataset_info,
    read_systems,
    read_targets,
)


RESOURCES_PATH = Path(__file__).parents[2] / "resources"


def test_dataset_info():
    """Tests the DatasetInfo class."""

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 2, 3],
        targets={
            "energy": TargetInfo(quantity="energy", unit="kcal/mol"),
            "mtm::U0": TargetInfo(quantity="energy", unit="kcal/mol"),
        },
    )

    assert dataset_info.length_unit == "angstrom"
    assert dataset_info.atomic_types == [1, 2, 3]
    assert dataset_info.targets["energy"].quantity == "energy"
    assert dataset_info.targets["energy"].unit == "kcal/mol"
    assert dataset_info.targets["mtm::U0"].quantity == "energy"
    assert dataset_info.targets["mtm::U0"].unit == "kcal/mol"


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


def test_merge_dataset_info():
    """Tests the merge_dataset_info function."""

    old_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 6],
        targets={
            "energy": TargetInfo(quantity="energy", unit="eV"),
            "mtm::forces": TargetInfo(quantity="mtm::forces", unit="eV/Angstrom"),
        },
    )

    new_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1],
        targets={
            "energy": TargetInfo(quantity="energy", unit="eV"),
            "mtm::forces": TargetInfo(quantity="mtm::forces", unit="eV/Angstrom"),
            "mtm::stress": TargetInfo(quantity="mtm::stress", unit="GPa"),
        },
    )

    merged, novel_types, novel_targets = merge_dataset_info(old_info, new_info)

    assert merged.length_unit == "angstrom"
    assert merged.atomic_types == [1, 6]
    assert merged.targets["energy"].quantity == "energy"
    assert merged.targets["energy"].unit == "eV"
    assert merged.targets["mtm::forces"].quantity == "mtm::forces"
    assert merged.targets["mtm::forces"].unit == "eV/Angstrom"
    assert merged.targets["mtm::stress"].quantity == "mtm::stress"
    assert merged.targets["mtm::stress"].unit == "GPa"

    assert novel_types == []
    assert novel_targets["mtm::stress"].quantity == "mtm::stress"
    assert novel_targets["mtm::stress"].unit == "GPa"
