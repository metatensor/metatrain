from omegaconf import OmegaConf

from ... import ARCHITECTURE_CONFIG_PATH


ARCHITECTURE_NAME = "experimental.pet_jax"
DEFAULT_HYPERS = OmegaConf.to_container(
    OmegaConf.load(ARCHITECTURE_CONFIG_PATH / f"{ARCHITECTURE_NAME}.yaml")
)
DEFAULT_MODEL_HYPERS = DEFAULT_HYPERS["model"]


class Model:
    pass
