from metatrain.utils.architectures import preload_documentation_module

from .extender import TensorBasis


preload_documentation_module("soap_bpnn")

__extender__ = TensorBasis
