import torch
from metatensor.torch.atomistic import systems_to_torch

from metatensor.models.experimental.alchemical_model import AlchemicalModel, Trainer
from metatensor.models.utils.data import DatasetInfo, TargetInfo

from . import MODEL_HYPERS


systems = read_systems(DATASET_PATH)

conf = {
    "mtm::U0": {
        "quantity": "energy",
        "read_from": DATASET_PATH,
        "file_format": ".xyz",
        "key": "U0",
        "forces": False,
        "stress": False,
        "virial": False,
    }
}
targets = read_targets(OmegaConf.create(conf))
dataset = Dataset({"system": systems, "mtm::U0": targets["mtm::U0"]})

hypers = DEFAULT_HYPERS.copy()
hypers["training"]["num_epochs"] = 2

dataset_info = DatasetInfo(
    length_unit="Angstrom",
    atomic_types=[1, 6, 7, 8],
    targets={
        "mtm::U0": TargetInfo(
            quantity="energy",
            unit="eV",
        ),
    },
)
model = AlchemicalModel(MODEL_HYPERS, dataset_info)

hypers["training"]["num_epochs"] = 1
trainer = Trainer(hypers["training"])
trainer.train(model, [torch.device("cpu")], [dataset], [dataset], ".")

# Predict on the first five systems
evaluation_options = ModelEvaluationOptions(
    length_unit=dataset_info.length_unit,
    outputs=model.outputs,
)

exported = model.export()

output = exported(systems[:5], evaluation_options, check_consistency=True)

expected_output = torch.tensor(
    [
        [-124.329589843750],
        [-111.787399291992],
        [-135.241882324219],
        [-177.760009765625],
        [-148.990478515625],
    ]
)

# if you need to change the hardcoded values:
# torch.set_printoptions(precision=12)
# print(output["mtm::U0"].block().values)

torch.testing.assert_close(
    output["mtm::U0"].block().values,
    expected_output,
)
