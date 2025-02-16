import torch
from metatensor.torch.atomistic import ModelMetadata
from omegaconf import OmegaConf

from metatrain.gap import GAP, Trainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.data.target_info import get_energy_target_info

from . import DATASET_PATH, DEFAULT_HYPERS


def test_export():
    """Tests that export works with injected metadata"""

    systems = read_systems(DATASET_PATH)

    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": DATASET_PATH,
            "reader": "ase",
            "key": "U0",
            "unit": "kcal/mol",
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

    target_info_dict = {}
    target_info_dict["energy"] = get_energy_target_info({"unit": "eV"})

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=target_info_dict
    )

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"})
        },
    )
    model = GAP(DEFAULT_HYPERS["model"], dataset_info)

    # we have to train gap before we can export...
    trainer = Trainer(DEFAULT_HYPERS["training"])
    trainer.train(
        model=model,
        dtype=torch.float64,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=".",
    )

    exported = model.export(metadata=ModelMetadata(name="test"))

    # test correct metadata
    assert "This is the test model" in str(exported.metadata())
