from .model import Model  # noqa: F401
from .train import train  # noqa: F401

from metatensor.models import ARCHITECTURE_CONFIG_PATH
from omegaconf import OmegaConf

DEAFAULT_HYPERS = OmegaConf.to_container(
    OmegaConf.load(ARCHITECTURE_CONFIG_PATH / "soap_bpnn.yaml")
)

DEFAULT_MODEL_HYPERS = DEAFAULT_HYPERS["model"]
DEFAULT_TRAIN_HYPERS = DEAFAULT_HYPERS["train"]
