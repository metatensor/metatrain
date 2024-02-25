# This is fixing a small bug in the attention implementation
# in torch that prevents it from being torchscriptable.

import os

import torch


file = os.path.join(os.path.dirname(torch.__file__), "nn", "modules", "activation.py")

with open(file, "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if (
            "elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:"  # noqa: E501
            in line
        ):
            lines[i] = line.replace(
                "elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:",  # noqa: E501
                "elif self.in_proj_bias is not None:\n"
                "            if query.dtype != self.in_proj_bias.dtype:",
            )
            lines[i + 1] = (
                '                why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) do not match"\n'  # noqa: E501
            )

with open(file, "w") as f:
    f.writelines(lines)
