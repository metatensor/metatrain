import torch


def dtype_to_str(dtype: torch.dtype) -> str:
    """
    Convert a torch dtype to its string representation.

    :param dtype: torch dtype to convert
    :returns: string representation of the torch dtype

    Example
    -------
    >>> import torch
    >>> dtype_to_str(torch.float64)
    "float64"
    >>> dtype_to_str(torch.int32)
    "int32"
    """
    return str(dtype).split(".")[-1]
