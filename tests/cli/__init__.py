from pathlib import Path

from metatrain.utils.architectures import get_default_hypers


MODEL_HYPERS = get_default_hypers("soap_bpnn")["model"]

RESOURCES_PATH = Path(__file__).parents[1] / "resources"

DATASET_PATH_QM9 = RESOURCES_PATH / "qm9_reduced_100.xyz"
DATASET_PATH_ETHANOL = RESOURCES_PATH / "ethanol_reduced_100.xyz"
DATASET_PATH_CARBON = RESOURCES_PATH / "carbon_reduced_100.xyz"
DATASET_PATH_QM7X = RESOURCES_PATH / "qm7x_reduced_100.xyz"
EVAL_OPTIONS_PATH = RESOURCES_PATH / "eval.yaml"
MODEL_PATH = RESOURCES_PATH / "model-32-bit.pt"
MODEL_PATH_64_BIT = RESOURCES_PATH / "model-64-bit.ckpt"
MODEL_PATH_PET = RESOURCES_PATH / "model-pet.ckpt"
OPTIONS_PATH = RESOURCES_PATH / "options.yaml"
OPTIONS_PET_PATH = RESOURCES_PATH / "options-pet.yaml"
OPTIONS_EXTRA_DATA_PATH = RESOURCES_PATH / "options-extra-data.yaml"
