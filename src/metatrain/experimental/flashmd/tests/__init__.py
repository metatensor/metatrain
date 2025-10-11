from pathlib import Path

from metatrain.utils.architectures import get_default_hypers


DEFAULT_HYPERS = get_default_hypers("experimental.flashmd")
MODEL_HYPERS = DEFAULT_HYPERS["model"]
DATASET_PATH = str(Path(__file__).parents[0] / "data/flashmd.xyz")
