from pathlib import Path

from metatrain.utils.architectures import get_default_hypers


DATASET_PATH = str(Path(__file__).parents[4] / "tests/resources/qm9_reduced_100.xyz")
print(DATASET_PATH)

DEFAULT_HYPERS = get_default_hypers("pet")
MODEL_HYPERS = DEFAULT_HYPERS["model"]
