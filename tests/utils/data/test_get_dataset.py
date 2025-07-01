from pathlib import Path

from omegaconf import OmegaConf

from metatrain.utils.data import get_dataset


RESOURCES_PATH = Path(__file__).parents[2] / "resources"


def test_get_dataset():
    options = {
        "systems": {
            "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
            "reader": "ase",
        },
        "targets": {
            "energy": {
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
        },
        "extra_data": {
            "extra_energy": {
                "quantity": "",
                "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
                "reader": "ase",
                "key": "U0",
                "unit": "eV",
                "type": "scalar",
                "per_atom": False,
                "num_subtargets": 1,
            }
        },
    }

    dataset, target_info, _ = get_dataset(OmegaConf.create(options))

    dataset[0].system
    dataset[0].energy
    dataset[0]["extra_energy"]
    assert "energy" in target_info
    assert target_info["energy"].quantity == "energy"
    assert target_info["energy"].unit == "eV"
