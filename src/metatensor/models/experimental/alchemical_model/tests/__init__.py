from pathlib import Path
from metatensor.models.utils.architectures import get_default_hypers


DATASET_PATH = str(Path(__file__).parents[6] / "tests/resources/qm9_reduced_100.xyz")

ALCHEMICAL_DATASET_PATH = str(
    Path(__file__).parents[6] / "tests/resources/alchemical_reduced_10.xyz"
)

DEFAULT_HYPERS = get_default_hypers("experimental.alchemical_model")
MODEL_HYPERS = DEFAULT_HYPERS["model"]
