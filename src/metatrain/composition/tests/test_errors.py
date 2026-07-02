import pytest

from metatrain.composition import CompositionModel
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info

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
    from metatrain.utils.data.target_info import get_generic_target_info

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


def test_forward_unknown_output_raises():
    """Test that forward with unknown output raises."""
    from metatrain.utils.data.readers import read_systems

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
    from metatomic.torch import ModelOutput

    with pytest.raises(ValueError, match="not supported"):
        model(systems[:1], {"nonexistent": ModelOutput(quantity="energy")})
