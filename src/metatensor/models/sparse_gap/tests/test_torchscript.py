import torch
from metatensor.learn.data import Dataset
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput
from omegaconf import OmegaConf

from metatensor.models.sparse_gap import DEFAULT_HYPERS, Model, train
from metatensor.models.utils.data.readers import read_structures, read_targets

from . import DATASET_PATH


def test_torchscript():
    """Tests that the model can be jitted."""

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        species=[1, 6, 7, 8],
        outputs={
            "U0": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
    )
    conf = {
        "U0": {
            "quantity": "energy",
            "read_from": DATASET_PATH,
            "file_format": ".xyz",
            "key": "U0",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    sparse_gap = Model(capabilities, DEFAULT_HYPERS["model"]).to(torch.float64)
    targets = read_targets(OmegaConf.create(conf))
    structures = read_structures(DATASET_PATH)
    # PR COMMENT this is a temporary hack until kernel is properly implemented that can
    #            deal with tensor maps with different species pairs
    for structure in structures:
        structure.species = torch.ones(len(structure.species), dtype=torch.int32)
    dataset = Dataset(structure=structures, U0=targets["U0"])

    hypers = DEFAULT_HYPERS.copy()
    hypers["training"]["num_epochs"] = 2
    sparse_gap = train([dataset], [dataset], capabilities, hypers)
    scripted_sparse_gap = torch.jit.script(
        sparse_gap, {"U0": sparse_gap.capabilities.outputs["U0"]}
    )

    ref_output = sparse_gap(
        structures[:5], {"U0": sparse_gap.capabilities.outputs["U0"]}
    )
    scripted_output = scripted_sparse_gap(
        structures[:5], {"U0": sparse_gap.capabilities.outputs["U0"]}
    )

    assert torch.allclose(
        ref_output["U0"].block().values,
        scripted_output["U0"].block().values,
    )


def test_torchscript_save():
    """Tests that the model can be jitted and saved."""

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        species=[1, 6, 7, 8],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
    )
    sparse_gap = Model(capabilities, DEFAULT_HYPERS["model"]).to(torch.float64)
    torch.jit.save(
        torch.jit.script(
            sparse_gap, {"energy": sparse_gap.capabilities.outputs["energy"]}
        ),
        "sparse_gap.pt",
    )
