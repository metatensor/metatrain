import torch

from metatrain.utils.data import DatasetInfo

from .hook import HookTests


class TorchscriptTests(HookTests):
    def test_torchscript(self, hypers: dict, dataset_info: DatasetInfo) -> None:
        hook = self.hook_cls(hypers, dataset_info)

        torch.jit.script(hook)
