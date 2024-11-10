import torch
from omegaconf import OmegaConf

from metatrain.experimental.gap import GAP, Trainer
from metatrain.utils.data import Dataset, DatasetInfo, TargetInfo
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.testing import energy_layout

from . import DATASET_PATH, DEFAULT_HYPERS


torch.set_default_dtype(torch.float64)  # GAP only supports float64


def test_torchscript():
    """Tests that the model can be jitted."""
    target_info_dict = {}
    target_info_dict["mtt::U0"] = TargetInfo(
        quantity="energy", unit="eV", layout=energy_layout
    )

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=target_info_dict
    )
    conf = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": DATASET_PATH,
            "reader": "ase",
            "key": "U0",
            "unit": "kcal/mol",
            "type": "scalar",
            "per_atom": False,
            "num_properties": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, _ = read_targets(OmegaConf.create(conf))
    systems = read_systems(DATASET_PATH)

    dataset = Dataset.from_dict({"system": systems, "mtt::U0": targets["mtt::U0"]})

    hypers = DEFAULT_HYPERS.copy()
    gap = GAP(DEFAULT_HYPERS["model"], dataset_info)
    trainer = Trainer(hypers["training"])
    trainer.train(
        model=gap,
        dtype=torch.float64,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=".",
    )
    scripted_gap = torch.jit.script(gap)

    ref_output = gap.forward(systems[:5], {"mtt::U0": gap.outputs["mtt::U0"]})
    scripted_output = scripted_gap.forward(
        systems[:5], {"mtt::U0": gap.outputs["mtt::U0"]}
    )

    assert torch.allclose(
        ref_output["mtt::U0"].block().values,
        scripted_output["mtt::U0"].block().values,
    )


def test_torchscript_save():
    """Tests that the model can be jitted and saved."""
    targets = {}
    targets["mtt::U0"] = TargetInfo(quantity="energy", unit="eV", layout=energy_layout)

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=targets
    )
    gap = GAP(DEFAULT_HYPERS["model"], dataset_info)
    torch.jit.save(
        torch.jit.script(gap),
        "gap.pt",
    )
    torch.jit.load("gap.pt")
