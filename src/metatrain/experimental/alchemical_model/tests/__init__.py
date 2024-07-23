from pathlib import Path
from metatrain.utils.architectures import get_default_hypers


DATASET_PATH = str(Path(__file__).parents[5] / "tests/resources/qm9_reduced_100.xyz")

CARBON_DATASET_PATH = str(
    Path(__file__).parents[5] / "tests/resources/carbon_reduced_100.xyz"
)

DEFAULT_HYPERS = get_default_hypers("experimental.alchemical_model")
MODEL_HYPERS = DEFAULT_HYPERS["model"]
