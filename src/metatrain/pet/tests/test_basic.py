import pytest

from metatrain.utils.testing.autograd import AutogradTests
from metatrain.utils.testing.exported import ExportedTests
from metatrain.utils.testing.torchscript import TorchscriptTests
from metatrain.utils.testing.output import OutputTests

class TestOutput(OutputTests):
    architecture = "pet"

    is_equivariant_model = False

    @pytest.fixture
    def n_features(self, model_hypers):
        num_readout_layers = (
            1
            if model_hypers["featurizer_type"] == "feedforward"
            else model_hypers["num_gnn_layers"]
        )
        
        return (model_hypers["d_node"] + model_hypers["d_pet"]) * num_readout_layers
    
    @pytest.fixture
    def n_last_layer_features(self, model_hypers):
        num_readout_layers = (
            1
            if model_hypers["featurizer_type"] == "feedforward"
            else model_hypers["num_gnn_layers"]
        )
        
        return model_hypers["d_head"] * num_readout_layers * 2

class TestAutograd(AutogradTests):
    architecture = "pet"

class TestTorchscript(TorchscriptTests):
    architecture = "pet"
    float_hypers = ["cutoff", "cutoff_width"]

class TestExported(ExportedTests):
    architecture = "pet"


