import torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput
from omegaconf import OmegaConf

from metatensor.models.experimental.gap import DEFAULT_HYPERS, Model, train
from metatensor.models.utils.data import Dataset, DatasetInfo, TargetInfo
from metatensor.models.utils.data.readers import read_systems, read_targets

from . import DATASET_PATH


torch.set_default_dtype(torch.float64)  # GAP only supports float64


def test_torchscript():
    """Tests that the model can be jitted."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
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
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets = read_targets(OmegaConf.create(conf), dtype=torch.float64)
    systems = read_systems(DATASET_PATH, dtype=torch.float64)

    for system in systems:
        system.types = torch.ones(len(system.types), dtype=torch.int32)
    dataset = Dataset({"system": systems, "U0": targets["mtm::U0"]})

    hypers = DEFAULT_HYPERS.copy()
    gap = train([dataset], [dataset], dataset_info, [torch.device("cpu")], hypers)
    scripted_gap = torch.jit.script(gap)

    ref_output = gap(systems[:5], {"mtm::U0": gap.capabilities.outputs["mtm::U0"]})
    scripted_output = scripted_gap(
        systems[:5], {"mtm::U0": gap.capabilities.outputs["mtm::U0"]}
    )

    print(ref_output["mtm::U0"].block().values)
    print(scripted_output["mtm::U0"].block().values)

    assert torch.allclose(
        ref_output["mtm::U0"].block().values,
        scripted_output["mtm::U0"].block().values,
    )


def test_torchscript_save():
    """Tests that the model can be jitted and saved."""

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
    )
    gap = Model(capabilities, DEFAULT_HYPERS["model"])
    torch.jit.save(
        torch.jit.script(gap, {"energy": gap.capabilities.outputs["energy"]}),
        "gap.pt",
    )
