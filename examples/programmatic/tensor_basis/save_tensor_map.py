import subprocess
import sys

import ase.io
import metatensor.torch as mts
import numpy as np
import torch
from featomic.torch.clebsch_gordan import cartesian_to_spherical


# %%
#
# Read a subset of 1000 molecules from the QM7x dataset in the XYZ format decorated with
# the polarizability (Cartesian) tensor.
# Extract the polarizability from the ase.Atoms.info dictionary.
#
molecules = ase.io.read("qm7x_reduced_100.xyz", ":")
polarizabilities = np.array(
    [molecule.info["polarizability"].reshape(3, 3) for molecule in molecules]
)

# %%
#
# Create a ``metatensor.torch.TensorMap`` containing the Cartesian polarizability tensor
# values and the respective metadata

cartesian_tensormap = mts.TensorMap(
    keys=mts.Labels.single(),
    blocks=[
        mts.TensorBlock(
            samples=mts.Labels.range("system", len(molecules)),
            components=[mts.Labels.range(name, 3) for name in ["xyz_1", "xyz_2"]],
            properties=mts.Labels(["polarizability"], torch.tensor([[0]])),
            values=torch.from_numpy(polarizabilities).unsqueeze(-1),
        )
    ],
)

# %%
#
# Extract from the Cartesian polarizability tensor its irreducible spherical components

spherical_tensormap = mts.remove_dimension(
    cartesian_to_spherical(cartesian_tensormap, components=["xyz_1", "xyz_2"]),
    "keys",
    "_",
)

# %%
#
# We drop the block with ``o3_sigma=-1``, as polarizability should be symmetric and
# therefore any non-zero pseudo-vector component is spurious.
spherical_tensormap = mts.drop_blocks(
    spherical_tensormap, mts.Labels(["o3_sigma"], torch.tensor([[-1]]))
)
# %%
#
# Let's save the spherical components of the polarizability tensor to disk
mts.save("spherical_polarizability.mts", spherical_tensormap)
