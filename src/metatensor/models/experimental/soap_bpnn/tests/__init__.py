from pathlib import Path
from metatensor.models.utils.architectures import get_default_hypers

DATASET_PATH = str(
    Path(__file__).parent.resolve()
    / "../../../../../../tests/resources/qm9_reduced_100.xyz"
)


DEFAULT_HYPERS = get_default_hypers("experimental.soap_bpnn")
MODEL_HYPERS = DEFAULT_HYPERS["model"]
