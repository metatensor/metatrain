import copy
from pathlib import Path

import pytest

from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.testing import (
    ArchitectureTests,
    CheckpointTests,
    TrainingTests,
)


@pytest.mark.filterwarnings("ignore:custom data:UserWarning")
class FlashMDTests(ArchitectureTests):
    architecture = "experimental.flashmd"

    @pytest.fixture
    def minimal_model_hypers(self):
        hypers = get_default_hypers(self.architecture)["model"]
        hypers = copy.deepcopy(hypers)
        hypers["d_pet"] = 1
        hypers["d_head"] = 1
        hypers["d_node"] = 1
        hypers["d_feedforward"] = 1
        hypers["num_heads"] = 1
        hypers["num_attention_layers"] = 1
        hypers["num_gnn_layers"] = 1
        return hypers

    @pytest.fixture
    def dataset_path(self):
        return str(Path(__file__).parents[0] / "data/flashmd.xyz")

    @pytest.fixture
    def dataset_targets(self, dataset_path):
        positions_target = {
            "quantity": "position",
            "read_from": dataset_path,
            "reader": "ase",
            "key": "future_positions",
            "unit": "A",
            "type": {
                "cartesian": {
                    "rank": 1,
                }
            },
            "per_atom": True,
            "num_subtargets": 1,
        }

        momenta_target = {
            "quantity": "momentum",
            "read_from": dataset_path,
            "reader": "ase",
            "key": "future_momenta",
            "unit": "(eV*u)^(1/2)",
            "type": {
                "cartesian": {
                    "rank": 1,
                }
            },
            "per_atom": True,
            "num_subtargets": 1,
        }

        return {
            "positions": positions_target,
            "momenta": momenta_target,
        }


class TestCheckpoints(CheckpointTests, FlashMDTests):
    architecture = "experimental.flashmd_symplectic"
    
    @pytest.fixture
    def default_hypers(self):
        hypers = get_default_hypers(self.architecture)
        hypers = copy.deepcopy(hypers)
        hypers["training"]["timestep"] = 30.0
        hypers["training"]["batch_size"] = 1
        return hypers


class TestTraining(TrainingTests, FlashMDTests):
    @pytest.fixture
    def default_hypers(self):
        hypers = get_default_hypers(self.architecture)
        hypers = copy.deepcopy(hypers)
        hypers["training"]["timestep"] = 30.0
        hypers["training"]["batch_size"] = 2
        return hypers
