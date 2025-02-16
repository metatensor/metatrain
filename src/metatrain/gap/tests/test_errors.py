import copy
import random
import re

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from metatrain.gap import GAP, Trainer
from metatrain.utils.data import Dataset, DatasetInfo, read_systems, read_targets
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)

from . import DATASET_ETHANOL_PATH, DEFAULT_HYPERS


torch.set_default_dtype(torch.float64)  # GAP only supports float64


# reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def test_ethanol_regression_train_and_invariance():
    """test the error if the number of sparse point
    is bigger than the number of environments
    """

    systems = read_systems(DATASET_ETHANOL_PATH)

    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": DATASET_ETHANOL_PATH,
            "reader": "ase",
            "key": "energy",
            "unit": "kcal/mol",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": {
                "read_from": DATASET_ETHANOL_PATH,
                "reader": "ase",
                "key": "forces",
            },
            "stress": False,
            "virial": False,
        }
    }

    targets, _ = read_targets(OmegaConf.create(conf))
    dataset = Dataset.from_dict(
        {"system": systems[:2], "energy": targets["energy"][:2]}
    )

    hypers = copy.deepcopy(DEFAULT_HYPERS)
    hypers["model"]["krr"]["num_sparse_points"] = 30

    target_info_dict = {
        "energy": get_energy_target_info(
            {"unit": "kcal/mol"}, add_position_gradients=True
        )
    }

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=target_info_dict
    )

    gap = GAP(hypers["model"], dataset_info)
    trainer = Trainer(hypers["training"])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Number of sparse points (30) "
            "should be smaller than the number of environments (18)"
        ),
    ):
        trainer.train(
            model=gap,
            dtype=torch.float64,
            devices=[torch.device("cpu")],
            train_datasets=[dataset],
            val_datasets=[dataset],
            checkpoint_dir=".",
        )


@pytest.mark.parametrize("per_atom", [True, False])
def test_vector_output(per_atom):
    """Tests that the model can predict a (spherical) vector output."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "forces": get_generic_target_info(
                {
                    "quantity": "forces",
                    "unit": "",
                    "type": {
                        "spherical": {"irreps": [{"o3_lambda": 1, "o3_sigma": 1}]}
                    },
                    "num_subtargets": 100,
                    "per_atom": per_atom,
                }
            )
        },
    )

    with pytest.raises(ValueError, match="GAP only supports total-energy-like outputs"):
        GAP(DEFAULT_HYPERS["model"], dataset_info)
