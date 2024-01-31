# Since torch.jit.save cannot handle Labels.single(), we need to replace it with
# Labels(names=["_"], values=_dispatch.zeros_like(block.values, (1, 1)))
# in metatensor-operations. This is a hacky way to do it.

import os
import metatensor.operations
import metatensor.torch.atomistic

file = os.path.join(
    os.path.dirname(metatensor.operations.__file__),
    "reduce_over_samples.py"
)

# Find the line that contains "Labels.single()"
# and replace "Labels.single()" with 
# "Labels(names=["_"], values=_dispatch.zeros_like(block.values, (1, 1)))"
with open(file, "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "samples_label = Labels.single()" in line:
            lines[i] = line.replace(
                "samples_label = Labels.single()",
                "samples_label = Labels(names=[\"_\"], values=_dispatch.zeros_like(block.values, (1, 1)))"
            )
            break

with open(file, "w") as f:
    f.writelines(lines)

file = os.path.join(
    os.path.dirname(metatensor.torch.atomistic.__file__),
    "units.py"
)

# At the moment, unit handling is incorrect in metatensor.torch.atomistic
# This will fix it:
with open(file, "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "return self._conversions[to_unit] / self._conversions[from_unit]" in line:
            lines[i] = line.replace(
                "return self._conversions[to_unit] / self._conversions[from_unit]",
                "return self._conversions[from_unit] / self._conversions[to_unit]"
            )
            break

with open(file, "w") as f:
    f.writelines(lines)
