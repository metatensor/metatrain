import torch

from metatrain.utils.data import DatasetInfo

from .extender import ExtenderTests


class TorchscriptTests(ExtenderTests):
    def test_torchscript(self, hypers: dict, dataset_info: DatasetInfo) -> None:
        extender = self.extender_cls(hypers, dataset_info)

        torch.jit.script(extender)
