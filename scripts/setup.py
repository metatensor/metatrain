import os
import metatensor.operations

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
