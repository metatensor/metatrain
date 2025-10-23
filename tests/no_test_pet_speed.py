import cProfile

import pytest
import torch
import torch._dynamo
from metatomic.torch import ModelEvaluationOptions, ModelOutput

from metatrain.utils.data import Dataset, read_systems, read_targets
from metatrain.utils.io import load_model
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)


def load_dataset(model, device: str):
    dataset_path = "tests/resources/ethanol_reduced_100.xyz"
    systems = read_systems(dataset_path)
    requested_neighbor_lists = get_requested_neighbor_lists(model)
    systems_with_nb_lists = [
        get_system_with_neighbor_lists(system, requested_neighbor_lists)
        for system in systems
    ]

    # Read the dataset's targets.
    target_config = {
        "energy": {
            "quantity": "energy",
            "read_from": dataset_path,
            "reader": "ase",
            "key": "energy",
            "unit": "kcal/mol",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        },
    }
    targets, infos = read_targets(target_config)  # type: ignore

    # Wrap in a `metatrain` compatible way.
    dataset = Dataset.from_dict({"system": systems_with_nb_lists, **targets})

    test_systems = [
        sample["system"].to(dtype=torch.float32, device=device) for sample in dataset
    ]
    return test_systems


@pytest.mark.parametrize("compile", [False, True], ids=["compiled", "raw"])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_compile_mad_val100(compile, device):
    "Test the compilation of PET and applying it to 100 samples of MAD-val."

    if device == "cuda" and not torch.cuda.is_available():
        # TODO: warn about the skipped test?
        return

    torch._dynamo.reset()

    # load model and dataset
    model = load_model("models/pet-mad-v1.1.0.ckpt")
    # model = load_model("https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.1.0/models/pet-mad-v1.1.0.ckpt")
    # model = load_model("https://huggingface.co/lab-cosmo/pet-mad/blob/v1.1.0/models/pet-mad-v1.1.0.ckpt")
    test_systems = load_dataset(model, device)

    # optional: compile model
    if compile:
        model.compile()

    # run a forward pass with the model (this is what should be profiled)
    options = ModelEvaluationOptions(
        length_unit="angstrom",
        outputs=dict(energy=ModelOutput()),
    )
    with cProfile.Profile() as pr:
        out = model(test_systems, options, False)
        pr.dump_stats(
            "prof/pet-%s-%s.prof" % ({True: "compiled", False: "raw"}[compile], device)
        )
    assert torch.all(torch.isfinite(out[0].values))
