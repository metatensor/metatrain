from pathlib import Path
from metatrain.utils.architectures import get_default_hypers

MODEL_HYPERS = get_default_hypers("experimental.soap_bpnn")["model"]

RESOURCES_PATH = Path(__file__).parents[1] / "resources"
