from pathlib import Path

from metatrain.utils.architectures import get_default_hypers


DATASET_PATH = str(Path(__file__).parents[4] / "tests/resources/qm9_reduced_100.xyz")
DATASET_WITH_FORCES_PATH = str(
    Path(__file__).parents[4] / "tests/resources/carbon_reduced_100.xyz"
)

DEFAULT_HYPERS_PET = get_default_hypers("pet")
DEFAULT_HYPERS_LLPR = get_default_hypers("llpr")
MODEL_HYPERS_PET = DEFAULT_HYPERS_PET["model"]
MODEL_HYPERS_LLPR = DEFAULT_HYPERS_LLPR["model"]
