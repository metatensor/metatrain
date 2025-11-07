# type: ignore
# This file is not complete so we can't run mypy on it.
import torch
from metatomic.torch import DatasetInfo, ModelMetadata

from metatrain.utils.abc import ModelInterface


# Definition of hyperparameters and their defaults:
class ModelHypers:
    alpha = 1.2
    mode = "strict"
    ...


class MyModel(ModelInterface):
    __checkpoint_version__ = 1
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]
    __default_metadata__ = ModelMetadata(
        references={"implementation": ["ref1"], "architecture": ["ref2"]}
    )
    __hypers_cls__ = ModelHypers

    def __init__(self, hypers: dict, dataset_info: DatasetInfo):
        super().__init__(hypers, dataset_info)

        # To access hyperparameters, one can use self.hypers, which
        # by default will be {'alpha': 1.2, 'mode': 'strict'}
        self.hypers["mode"]
        ...

    # Here one would implement the rest of the abstract methods
