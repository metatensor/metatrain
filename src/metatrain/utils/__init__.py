import os
from typing import Callable

import torch


def torch_jit_script_unless_coverage(function: Callable) -> Callable:
    """
    Decorator to use instead of ``torch.jit.script``, that does nothing when collecting
    code coverage.

    We can not collect coverage for TorchScript functions, so we bypass compilation in
    this case.

    :param function: The function to decorate.

    :return: The decorated function.
    """
    if os.environ.get("COVERAGE_RUN") is None:
        return torch.jit.script(function)
    else:
        return function
