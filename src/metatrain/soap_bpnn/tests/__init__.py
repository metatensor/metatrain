from pathlib import Path

from metatrain.utils.architectures import get_default_hypers


DATASET_PATH = str(Path(__file__).parents[4] / "tests/resources/qm9_reduced_100.xyz")
SPHERICAL_DISK_DATASET_PATH = str(
    Path(__file__).parents[4] / "tests/resources/spherical_disk_dataset.zip"
)

DEFAULT_HYPERS = get_default_hypers("soap_bpnn")
MODEL_HYPERS = DEFAULT_HYPERS["model"]
