"""Tests for PET with the ZBL short-range repulsion enabled.

Nothing else turns ``zbl`` on -- the regression tests set it to ``False`` --
so neither the unit check nor a training run with ZBL had any coverage.
"""

import copy

import pytest
import torch
from omegaconf import OmegaConf

from metatrain.pet import PET, Trainer
from metatrain.utils.additive import ZBL
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DATASET_WITH_FORCES_PATH, DEFAULT_HYPERS, MODEL_HYPERS


def _dataset():
    """Read the carbon set as a single energy-and-forces target.

    :return: The dataset, its systems and its target information.
    """
    systems = read_systems(DATASET_WITH_FORCES_PATH)
    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": DATASET_WITH_FORCES_PATH,
            "reader": "ase",
            "key": "energy",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": {"read_from": DATASET_WITH_FORCES_PATH, "key": "force"},
            "stress": False,
            "virial": False,
        }
    }
    targets, target_info = read_targets(OmegaConf.create(conf))
    dataset = Dataset.from_dict({"system": systems, "energy": targets["energy"]})
    return dataset, systems, target_info


def test_zbl_is_an_additive_model():
    """Asking for ZBL gives the model a ZBL to add, over the atomic types of the
    dataset."""
    dataset, _, target_info = _dataset()
    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[6], targets=target_info
    )
    model_hypers = copy.deepcopy(MODEL_HYPERS)
    model_hypers["zbl"] = True
    model = PET(model_hypers, dataset_info)

    zbl_models = [m for m in model.additive_models if isinstance(m, ZBL)]
    assert len(zbl_models) == 1
    assert zbl_models[0].dataset_info.atomic_types == [6]


@pytest.mark.parametrize("length_unit", ["angstrom", "Angstrom", "ANGSTROM", "A"])
def test_zbl_accepts_every_spelling_of_angstrom(length_unit):
    """However metatomic lets a dataset spell its angstroms, ZBL takes it.

    Only the exact string "angstrom" used to pass, so a dataset declaring
    "Angstrom" — which metatomic accepts, and which the tests here and the
    documented examples all use — lost ZBL to a message that read "only
    supports angstrom units, but a Angstrom unit was provided".
    """
    _, _, target_info = _dataset()
    ZBL({}, DatasetInfo(length_unit, atomic_types=[6], targets=target_info))


@pytest.mark.parametrize("length_unit", ["nanometer", "nm", "bohr", ""])
def test_zbl_rejects_lengths_that_are_not_angstroms(length_unit):
    """Anything else, including a dataset that never gave a unit, is refused."""
    _, _, target_info = _dataset()
    with pytest.raises(ValueError, match="ZBL only supports angstrom"):
        ZBL({}, DatasetInfo(length_unit, atomic_types=[6], targets=target_info))


def test_zbl_training():
    """PET trains and evaluates with ZBL enabled."""
    dataset, systems, target_info = _dataset()
    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[6], targets=target_info
    )
    model_hypers = copy.deepcopy(MODEL_HYPERS)
    model_hypers["zbl"] = True
    model = PET(model_hypers, dataset_info)

    hypers = copy.deepcopy(DEFAULT_HYPERS)
    hypers["training"]["num_epochs"] = 1
    hypers["training"]["num_workers"] = 0
    hypers["training"]["scheduler_patience"] = 1
    hypers["training"]["atomic_baseline"] = {}

    # the `loss: mse` shorthand is expanded by the yaml layer, which we bypass
    loss = {"energy": init_with_defaults(LossSpecification)}
    loss["energy"]["gradients"] = {"positions": init_with_defaults(LossSpecification)}
    loss = OmegaConf.create(loss)
    OmegaConf.resolve(loss)
    hypers["training"]["loss"] = loss

    Trainer(hypers["training"]).train(
        model=model,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=".",
    )

    systems = [system.to(torch.float32) for system in systems]
    for system in systems:
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    evaluate_model(model, systems[:5], targets=target_info, is_training=False)
