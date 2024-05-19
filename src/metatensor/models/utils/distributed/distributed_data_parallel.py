import torch


class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    def __init__(self, module, *args, **kwargs):
        super(DistributedDataParallel, self).__init__(module, *args, **kwargs)
        self.capabilities = module.capabilities
