from pathlib import Path

import ase.io
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from metatrain.utils.data import get_dataset
from metatrain.utils.omegaconf import expand_dataset_config


RESOURCES_PATH = Path(__file__).parents[2] / "resources"


def test_get_dataset():
    options = {
        "systems": {
            "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
            "reader": "ase",
        },
        "targets": {
            "energy": {
                "quantity": "energy",
                "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
                "reader": "ase",
                "key": "U0",
                "unit": "eV",
                "type": "scalar",
                "sample_kind": "system",
                "num_subtargets": 1,
                "forces": False,
                "stress": False,
                "virial": False,
            }
        },
        "extra_data": {
            "extra_energy": {
                "quantity": "",
                "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
                "reader": "ase",
                "key": "U0",
                "unit": "eV",
                "type": "scalar",
                "sample_kind": "system",
                "num_subtargets": 1,
            }
        },
    }

    dataset, target_info, _ = get_dataset(OmegaConf.create(options))

    dataset[0].system
    dataset[0].energy
    dataset[0]["extra_energy"]
    assert "energy" in target_info
    assert target_info["energy"].quantity == "energy"
    assert target_info["energy"].unit == "eV"


def test_get_dataset_sample_weights(tmp_path):
    """Per-sample loss weights (sample_weight_key) are read into extra_data as a
    weight TensorMap mirroring the structure of the target (including gradients)."""
    frames = ase.io.read(str(RESOURCES_PATH / "ethanol_reduced_100.xyz"), ":5")
    rng = np.random.default_rng(0)
    for atoms in frames:
        atoms.info["energy_weight"] = float(rng.uniform(0.5, 2.0))
        atoms.arrays["force_weight"] = rng.uniform(0.5, 2.0, size=len(atoms))
    weighted_file = tmp_path / "ethanol_weighted.xyz"
    ase.io.write(str(weighted_file), frames)

    options = expand_dataset_config(
        OmegaConf.create(
            {
                "systems": {"read_from": str(weighted_file), "length_unit": "angstrom"},
                "targets": {
                    "energy": {
                        "key": "energy",
                        "unit": "eV",
                        "sample_weight_key": "energy_weight",
                        "forces": {
                            "key": "forces",
                            "sample_weight_key": "force_weight",
                        },
                        "stress": False,
                    }
                },
            }
        )
    )[0]

    dataset, _, extra_data_info = get_dataset(options)

    # The weights end up under the reserved "<target>_weights" extra_data key.
    assert "energy_weights" in extra_data_info

    for i_system, atoms in enumerate(frames):
        weights = dataset[i_system]["energy_weights"]
        block = weights.block()

        # value weights: one per-structure weight broadcast over the block
        assert block.values.shape == (1, 1)
        assert block.values.item() == pytest.approx(atoms.info["energy_weight"])

        # gradient (force) weights: per-atom weight broadcast over the xyz components
        grad = block.gradient("positions")
        assert grad.values.shape == (len(atoms), 3, 1)
        expected = torch.tensor(atoms.arrays["force_weight"], dtype=torch.float64)
        for xyz in range(3):
            torch.testing.assert_close(grad.values[:, xyz, 0], expected)
