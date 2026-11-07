from metatrain.utils.architectures import preload_documentation_module

from .hook import TensorBasis


preload_documentation_module("soap_bpnn")

__hook__ = TensorBasis
