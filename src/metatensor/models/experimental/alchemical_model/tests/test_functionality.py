import ase
from metatensor.torch.atomistic import systems_to_torch, ModelEvaluationOptions

from metatensor.models.experimental.alchemical_model import AlchemicalModel
from metatensor.models.utils.data import DatasetInfo, TargetInfo
from metatensor.models.utils.neighbor_lists import get_system_with_neighbor_lists

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

    model = AlchemicalModel(MODEL_HYPERS, dataset_info)

    system = ase.Atoms("O2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    system = systems_to_torch(system)
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    evaluation_options = ModelEvaluationOptions(
        length_unit=dataset_info.length_unit,
        outputs=model.outputs,
    )

    exported = model.export()
    exported([system], evaluation_options, check_consistency=True)
