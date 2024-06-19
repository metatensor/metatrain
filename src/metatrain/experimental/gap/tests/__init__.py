from pathlib import Path

from metatrain.utils.architectures import get_default_hypers

DEFAULT_HYPERS = get_default_hypers("experimental.gap")
DATASET_PATH = str(Path(__file__).parents[5] / "tests/resources/qm9_reduced_100.xyz")

DATASET_ETHANOL_PATH = str(
    Path(__file__).parents[5] / "tests/resources/ethanol_reduced_100.xyz"
)
