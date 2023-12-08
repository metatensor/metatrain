from pathlib import Path

from metatensor.models import ARCHITECTURE_CONFIG_PATH
from omegaconf import OmegaConf


DEAFAULT_HYPERS = OmegaConf.to_container(
    OmegaConf.load(ARCHITECTURE_CONFIG_PATH / "soap_bpnn.yaml")
)
DATASET_PATH = str(
    Path(__file__).parent.resolve()
    / "../../../../../tests/resources/qm9_reduced_100.xyz"
)
