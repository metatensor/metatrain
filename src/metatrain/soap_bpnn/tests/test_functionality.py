import pytest
import torch
from metatomic.torch import ModelOutput, System

from metatrain.soap_bpnn import SoapBpnn
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


@pytest.mark.parametrize("per_atom", [True, False])
def test_nc_stress(per_atom):
    """Tests that the model can predict a symmetric rank-2 tensor as the NC stress."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "non_conservative_stress": get_generic_target_info(
                "non_conservative_stress",
                {
                    "quantity": "stress",
                    "unit": "",
                    "type": {"cartesian": {"rank": 2}},
                    "num_subtargets": 1,
                    "per_atom": per_atom,
                },
            )
        },
    )

    model = SoapBpnn(MODEL_HYPERS, dataset_info)

    system = System(
        types=torch.tensor([6]),
        positions=torch.tensor([[0.0, 0.0, 1.0]]),
        cell=torch.eye(3),
        pbc=torch.tensor([True, True, True]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = {"non_conservative_stress": ModelOutput(per_atom=per_atom)}
    stress = model([system], outputs)["non_conservative_stress"].block().values
    assert torch.allclose(stress, stress.transpose(-3, -2))
