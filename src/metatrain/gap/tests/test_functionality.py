from omegaconf import OmegaConf

from metatrain.utils.architectures import check_architecture_options

from . import DEFAULT_HYPERS


def test_valid_defaults():
    """Tests that the default hypers pass the architecture options check."""
    hypers = OmegaConf.create(DEFAULT_HYPERS)
    check_architecture_options(name="gap", options=OmegaConf.to_container(hypers))
