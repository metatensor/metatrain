from pathlib import Path

from metatrain.utils.architectures import get_default_hypers


DEFAULT_HYPERS = get_default_hypers("experimental.mace")
MODEL_HYPERS = DEFAULT_HYPERS["model"]
SPHERICAL_DISK_DATASET_PATH = str(
    Path(__file__).parents[5] / "tests/resources/spherical_disk_dataset.zip"
)
