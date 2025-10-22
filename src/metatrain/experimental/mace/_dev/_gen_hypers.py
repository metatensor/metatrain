"""
Tools to generate default hyperparameter YAML files for MACE.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from metatrain.experimental.mace.utils.hypers import (
    MACE_MODEL_ARG_KEYS,
    get_mace_defaults,
)


def regenerate_default_hypers():
    """Writes the default_hypers.yaml file for the MACE architechture.

    It fills the default_hypers_template.temp_yaml template with the default arguments
    from MACE, and then writes the result to default_hypers.yaml."""
    # Extract defaults without triggering required arguments
    mace_defaults = get_mace_defaults()

    # Get jinja2 template
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("default-hypers-template.temp_yaml")

    # Filter only model hyperparameters to pass them to the template
    mace_model_defaults = [(k, mace_defaults[k]) for k in MACE_MODEL_ARG_KEYS]

    # Render template and write to file
    with open(Path(__file__).parent.parent / "default-hypers.yaml", "w") as f:
        f.write(
            template.render(
                mace_defaults=mace_defaults, mace_model_defaults=mace_model_defaults
            )
        )


if __name__ == "__main__":
    regenerate_default_hypers()
