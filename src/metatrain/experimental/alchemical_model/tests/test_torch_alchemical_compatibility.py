import random

import numpy as np
import torch
from ase.io import read
from metatensor.torch.atomistic import ModelEvaluationOptions, NeighborListOptions
from torch_alchemical.data import AtomisticDataset
from torch_alchemical.models import AlchemicalModel as AlchemicalModelUpstream
from torch_alchemical.transforms import NeighborList
from torch_alchemical.utils import get_list_of_unique_atomic_numbers
from torch_geometric.loader import DataLoader

from metatrain.experimental.alchemical_model import AlchemicalModel
from metatrain.experimental.alchemical_model.utils import (
    systems_to_torch_alchemical_batch,
)
from metatrain.utils.data import DatasetInfo, TargetInfo, TargetInfoDict, read_systems
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import ALCHEMICAL_DATASET_PATH, MODEL_HYPERS


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

systems = read_systems(ALCHEMICAL_DATASET_PATH)
nl_options = NeighborListOptions(
    cutoff=5.0,
    full_list=True,
)
systems = [get_system_with_neighbor_lists(system, [nl_options]) for system in systems]

frames = read(ALCHEMICAL_DATASET_PATH, ":")
dataset = AtomisticDataset(
    frames,
    target_properties=["energies", "forces"],
    transforms=[NeighborList(cutoff_radius=5.0)],
)
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
batch = next(iter(dataloader))


def test_systems_to_torch_alchemical_batch():
    batch_dict = systems_to_torch_alchemical_batch(systems, nl_options)
    torch.testing.assert_close(batch_dict["positions"], batch.pos)
    torch.testing.assert_close(batch_dict["cells"], batch.cell)
    torch.testing.assert_close(batch_dict["numbers"], batch.numbers)
    index_1, counts_1 = torch.unique(batch_dict["batch"], return_counts=True)
    index_2, counts_2 = torch.unique(batch.batch, return_counts=True)
    torch.testing.assert_close(index_1, index_2)
    torch.testing.assert_close(counts_1, counts_2)
    offset_1, counts_1 = torch.unique(batch_dict["edge_offsets"], return_counts=True)
    offset_2, counts_2 = torch.unique(batch.edge_offsets, return_counts=True)
    torch.testing.assert_close(offset_1, offset_2)
    torch.testing.assert_close(counts_1, counts_2)
    torch.testing.assert_close(batch_dict["batch"], batch.batch)


def test_alchemical_model_inference():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    unique_numbers = get_list_of_unique_atomic_numbers(frames)

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=unique_numbers,
        targets=TargetInfoDict(energy=TargetInfo(quantity="energy", unit="eV")),
    )

    alchemical_model = AlchemicalModel(MODEL_HYPERS, dataset_info)

    evaluation_options = ModelEvaluationOptions(
        length_unit=dataset_info.length_unit,
        outputs=alchemical_model.outputs,
    )

    exported = alchemical_model.export()

    output = exported(systems, evaluation_options, check_consistency=True)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    original_model = AlchemicalModelUpstream(
        unique_numbers=unique_numbers,
        **MODEL_HYPERS["soap"],
        **MODEL_HYPERS["bpnn"],
    ).eval()
    original_output = original_model(
        positions=batch.pos,
        cells=batch.cell,
        numbers=batch.numbers,
        edge_indices=batch.edge_index,
        edge_offsets=batch.edge_offsets,
        batch=batch.batch,
    )
    torch.testing.assert_close(output["energy"].block().values, original_output)
