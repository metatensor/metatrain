import os

import torch


def torch_jit_script_unless_coverage(function):
    """
    Decorator to use instead of ``torch.jit.script``, that does nothing when collecting
    code coverage.

    We can not collect coverage for TorchScript functions, so we bypass compilation in
    this case.
    """
    if os.environ.get("COVERAGE_RUN") is None:
        return torch.jit.script(function)
    else:
        return function
