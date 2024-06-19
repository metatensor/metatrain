from metatrain.utils.architectures import get_default_hypers
from pathlib import Path

MODEL_HYPERS = get_default_hypers("experimental.soap_bpnn")["model"]

RESOURCES_PATH = Path(__file__).parents[1] / "resources"

DATASET_PATH_QM9 = RESOURCES_PATH / "qm9_reduced_100.xyz"
DATASET_PATH_ETHANOL = RESOURCES_PATH / "ethanol_reduced_100.xyz"
EVAL_OPTIONS_PATH = RESOURCES_PATH / "eval.yaml"
MODEL_PATH = RESOURCES_PATH / "model-32-bit.pt"
MODEL_PATH_64_BIT = RESOURCES_PATH / "model-64-bit.ckpt"
OPTIONS_PATH = RESOURCES_PATH / "options.yaml"
