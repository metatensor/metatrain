import torch


class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    """
    DistributedDataParallel wrapper that inherits from
    :py:class`torch.nn.parallel.DistributedDataParallel`
    and adds the capabilities attribute to it.

    :param module: The module to be parallelized.
    :param args: Arguments to be passed to the parent class.
    :param kwargs: Keyword arguments to be passed to the parent class
    """

    def __init__(self, module: torch.nn.Module, *args, **kwargs):
        super(DistributedDataParallel, self).__init__(module, *args, **kwargs)
        self.outputs = module.outputs
