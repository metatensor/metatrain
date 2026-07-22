import pytest
import torch
from metatomic.torch import ModelOutput, System

from metatrain.composition import CompositionModel, Trainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)

from . import DATASET_PATH


def test_non_empty_hypers_raises():
    """Test that hypers must be an empty dict."""
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )
    with pytest.raises(ValueError, match="hypers takes an empty dictionary"):
        CompositionModel(hypers={"some_key": "value"}, dataset_info=dataset_info)


def test_unsupported_target_raises():
    """Test that vector targets raise an error."""
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "vector_out": get_generic_target_info(
                "vector_out",
                {
                    "quantity": "vector",
                    "unit": "",
                    "type": {"cartesian": {"rank": 1}},
                    "num_subtargets": 3,
                    "sample_kind": "system",
                },
            )
        },
    )
    with pytest.raises(ValueError, match="does not support target"):
        CompositionModel(hypers={}, dataset_info=dataset_info)


def test_train_float32_raises():
    """Test that training on float32 systems raises: the composition model only
    supports float64."""
    systems = [
        System(
            positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            types=torch.tensor([8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ).to(torch.float32),
    ]
    dataset = Dataset.from_dict({"system": systems, "energy": [1.0]})

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )
    model = CompositionModel(hypers={}, dataset_info=dataset_info)
    trainer = Trainer(hypers={})

    with pytest.raises(
        ValueError,
        match=(
            "The composition model only supports float64 "
            "during training. Got dtype: torch.float32."
        ),
    ):
        trainer.train(
            model=model,
            dtype=torch.float32,
            devices=[torch.device("cpu")],
            train_datasets=[dataset],
            val_datasets=[dataset],
            checkpoint_dir="",
        )


def test_forward_unknown_output_raises():
    """Test that forward with unknown output raises."""
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )
    model = CompositionModel(hypers={}, dataset_info=dataset_info)
    model.eval()

    systems = read_systems(DATASET_PATH)
    with pytest.raises(ValueError, match="not supported"):
        model(systems[:1], {"nonexistent": ModelOutput(quantity="energy")})
