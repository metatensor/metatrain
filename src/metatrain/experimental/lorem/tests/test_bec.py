import copy

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.experimental.lorem import LOREM
from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


def _bec_target_info() -> TargetInfo:
    block = TensorBlock(
        values=torch.empty(0, 3, 3, 1, dtype=torch.float64),
        samples=Labels(
            names=["system", "atom"],
            values=torch.empty((0, 2), dtype=torch.int32),
        ),
        components=[
            Labels("xyz_1", torch.arange(3).reshape(-1, 1)),
            Labels("xyz_2", torch.arange(3).reshape(-1, 1)),
        ],
        properties=Labels.range("bec", 1),
    )
    return TargetInfo(
        layout=TensorMap(keys=Labels.single(), blocks=[block]),
        quantity="bec",
        unit="e",
    )


def _dataset_info():
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            ),
            "bec": _bec_target_info(),
        },
    )


def _hypers():
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["max_degree"] = 2
    hypers["max_degree_lr"] = 1
    hypers["num_features"] = 8
    hypers["num_spherical_features"] = 2
    hypers["num_radial"] = 4
    hypers["num_message_passing"] = 0
    return hypers


def _chain_system():
    return System(
        types=torch.tensor([6, 6, 8, 8]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )


def test_bec_requires_max_degree_2():
    hypers = _hypers()
    hypers["max_degree"] = 1
    with pytest.raises(ValueError, match="max_degree >= 2"):
        LOREM(hypers, _dataset_info())


def test_bec_acoustic_sum_rule():
    model = LOREM(_hypers(), _dataset_info())
    model.eval()
    system = get_system_with_neighbor_lists(
        _chain_system(), model.requested_neighbor_lists()
    )
    outputs = {"bec": ModelOutput(sample_kind="atom")}
    bec = model([system], outputs)["bec"].block().values
    assert bec.shape == (4, 3, 3, 1)
    torch.testing.assert_close(
        bec.sum(dim=0), torch.zeros(3, 3, 1), atol=1e-5, rtol=1e-5
    )


def test_bec_torchscript():
    model = LOREM(_hypers(), _dataset_info())
    scripted = torch.jit.script(model)
    system = get_system_with_neighbor_lists(
        _chain_system(), model.requested_neighbor_lists()
    )
    outputs = {
        "energy": ModelOutput(sample_kind="system"),
        "bec": ModelOutput(sample_kind="atom"),
    }
    eager = model([system], outputs)["bec"].block().values
    compiled = scripted([system], outputs)["bec"].block().values
    torch.testing.assert_close(eager, compiled, atol=1e-5, rtol=1e-5)
