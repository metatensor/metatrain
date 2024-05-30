import torch
from omegaconf import OmegaConf

from metatensor.models.experimental.gap import GAP, Trainer
from metatensor.models.utils.architectures import get_default_hypers
from metatensor.models.utils.data import Dataset, DatasetInfo, TargetInfo
from metatensor.models.utils.data.readers import read_systems, read_targets

from . import DATASET_PATH


DEFAULT_HYPERS = get_default_hypers("experimental.gap")


torch.set_default_dtype(torch.float64)  # GAP only supports float64


def test_torchscript():
    """Tests that the model can be jitted."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "mtm::U0": TargetInfo(
                quantity="energy",
                unit="eV",
            ),
        },
    )
    conf = {
        "mtm::U0": {
            "quantity": "energy",
            "read_from": DATASET_PATH,
            "file_format": ".xyz",
            "key": "U0",
            "unit": "kcal/mol",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, _ = read_targets(OmegaConf.create(conf), dtype=torch.float64)
    systems = read_systems(DATASET_PATH, dtype=torch.float64)

    # for system in systems:
    #    system.types = torch.ones(len(system.types), dtype=torch.int32)
    dataset = Dataset({"system": systems, "mtm::U0": targets["mtm::U0"]})

    hypers = DEFAULT_HYPERS.copy()
    gap = GAP(DEFAULT_HYPERS["model"], dataset_info)
    trainer = Trainer(hypers["training"])
    gap = trainer.train(gap, [torch.device("cpu")], [dataset], [dataset], ".")
    scripted_gap = torch.jit.script(gap)

    ref_output = gap.forward(systems[:5], {"mtm::U0": gap.outputs["mtm::U0"]})
    scripted_output = scripted_gap.forward(
        systems[:5], {"mtm::U0": gap.outputs["mtm::U0"]}
    )

    assert torch.allclose(
        ref_output["mtm::U0"].block().values,
        scripted_output["mtm::U0"].block().values,
    )


def test_torchscript_save():
    """Tests that the model can be jitted and saved."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "mtm::U0": TargetInfo(
                quantity="energy",
                unit="eV",
            ),
        },
    )
    gap = GAP(DEFAULT_HYPERS["model"], dataset_info)
    torch.jit.save(
        torch.jit.script(gap),
        "gap.pt",
    )
    torch.jit.load("gap.pt")
