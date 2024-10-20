import torch
import numpy as np
from metatensor.torch.atomistic import ModelOutput, System

from metatrain.experimental.nanopet.model import NanoPET
from metatrain.utils.data import DatasetInfo, TargetInfo, TargetInfoDict
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from metatrain.utils.architectures import get_default_hypers

DEFAULT_HYPERS = get_default_hypers("experimental.nanopet")
MODEL_HYPERS = DEFAULT_HYPERS["model"]


dataset_info = DatasetInfo(
    length_unit="Angstrom",
    atomic_types=[1, 6, 7, 8],
    targets=TargetInfoDict(energy=TargetInfo(quantity="energy", unit="eV")),
)

MODEL_HYPERS["cutoff_width"] = 0.4
model = NanoPET(MODEL_HYPERS, dataset_info)

systems = [
    System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, float(a)]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False])
    ) for a in np.arange(4.5, 5.5, 0.001)
]
systems = [get_system_with_neighbor_lists(system, model.requested_neighbor_lists()) for system in systems]
outputs = {"energy": ModelOutput(per_atom=False)}

outputs = model(systems, outputs)
print(outputs["energy"].block().values.squeeze())
