"""
Learning electron densities
===========================

This tutorial demonstrates how to train a model for the electron density of an
atomic system.
"""

import ase.io
import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import NeighborListOptions, systems_to_torch

from metatrain.utils.data import DiskDatasetWriter
from metatrain.utils.io import load_model
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


def _get_fake_electron_density(atoms: ase.Atoms, structure_number: int) -> TensorMap:
    # Returns a random electron-density-like TensorMap object
    all_densities = {}
    for o3_lambda in range(6):
        for atomic_number in [1, 6, 7, 8, 9]:
            base_n_properties = 5 if atomic_number == 1 else 10
            all_densities[(o3_lambda, atomic_number)] = torch.tensor(
                np.random.normal(
                    size=(
                        np.sum(atoms.numbers == atomic_number),
                        2 * o3_lambda + 1,
                        base_n_properties - o3_lambda,
                    )
                ),
                dtype=torch.float64,
            )

    return TensorMap(
        keys=Labels(
            names=["o3_lambda", "o3_sigma", "center_type"],
            values=torch.tensor(
                [
                    [o3_lambda, 1, atomic_number]
                    for o3_lambda, atomic_number in all_densities.keys()
                ]
            ),
        ),
        blocks=[
            TensorBlock(
                values=values,
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor(
                        [
                            [structure_number, i]
                            for i, is_correct_type in enumerate(
                                atoms.numbers == atomic_number
                            )
                            if is_correct_type
                        ]
                    ).reshape(-1, 2),
                ),
                components=[
                    Labels(
                        "o3_mu",
                        torch.arange(
                            -o3_lambda, o3_lambda + 1, dtype=torch.int
                        ).reshape(-1, 1),
                    )
                ],
                properties=Labels.range("properties", values.shape[2]),
            )
            for (o3_lambda, atomic_number), values in all_densities.items()
        ],
    )


disk_dataset_writer = DiskDatasetWriter("qm9_reduced_100.zip")
for i in range(100):
    frame = ase.io.read("qm9_reduced_100.xyz", index=i)
    system = systems_to_torch(frame, dtype=torch.float64)
    system = get_system_with_neighbor_lists(
        system,
        [NeighborListOptions(cutoff=5.0, full_list=True, strict=True)],
    )
    electron_density = _get_fake_electron_density(frame, i)
    disk_dataset_writer.write_sample(
        system, {"mtt::electron_density": electron_density}
    )
del disk_dataset_writer

# %%
#
# Now that the dataset has been saved to disk, we can train a model on it.
# The model was trained using the following training options.
#
# .. literalinclude:: options.yaml
#    :language: yaml

# You can train the same model yourself with

# .. literalinclude:: train.sh
#    :language: bash

#
# Once the model has been trained, we can load it and use it:

load_model("model.pt", extensions_directory="extensions/")

# %%
#
# Analysis and plotting (@Joe)

...
