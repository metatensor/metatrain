import copy
import pytest

from metatrain.utils.testing.autograd import AutogradTests
from metatrain.utils.testing.exported import ExportedTests
from metatrain.utils.testing.torchscript import TorchscriptTests
from metatrain.utils.testing.output import OutputTests

class TestOutput(OutputTests):
    architecture = "soap_bpnn"

    supports_vector_outputs = False

    @pytest.fixture
    def n_features(self):
        return 128
    
    @pytest.fixture
    def n_last_layer_features(self):
        return 128
    
    @pytest.fixture
    def single_atom_energy(self):
        return 0.0
    
class TestAutograd(AutogradTests):
    architecture = "soap_bpnn"
    
class TestTorchscript(TorchscriptTests):
    architecture = "soap_bpnn"
    float_hypers = ["soap.cutoff.radius", "soap.cutoff.width"]

    def test_torchscript_with_identity(self, model_hypers, dataset_info):
        hypers = copy.deepcopy(model_hypers)
        hypers["bpnn"]["layernorm"] = False
        self.test_torchscript(model_hypers=hypers, dataset_info=dataset_info)

class TestExported(ExportedTests):
    architecture = "soap_bpnn"




