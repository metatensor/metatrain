import ase
from metatensor.torch.atomistic import systems_to_torch

from metatensor.models.experimental.alchemical_model import AlchemicalModel
from metatensor.models.utils.data import DatasetInfo, TargetInfo

from . import MODEL_HYPERS


def test_prediction_subset_elements():
    """Tests that the model can predict on a subset of the elements it was trained
    on."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": TargetInfo(
                quantity="energy",
                unit="eV",
            )
        },
    )

    soap_bpnn = AlchemicalModel(MODEL_HYPERS, dataset_info)

    system = ase.Atoms("O2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    soap_bpnn(
        [systems_to_torch(system)],
        {"energy": soap_bpnn.outputs["energy"]},
    )
