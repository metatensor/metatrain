import importlib
from pathlib import Path

import yaml

from metatrain.utils.hypers import get_hypers_annotation, init_with_defaults


def write_hypers_files(model: str) -> None:
    """Generate YAML file with defaults for a given architecture.

    Given a model name, this function imports the corresponding
    module, finds out what the hyperparameters are for the model
    and its trainer, and generates the `default-hypers.yaml`, a
    YAML file with the default hyperparameters.

    :param model: The model to generate the files for.
    """
    # Import the model module
    module = importlib.import_module(f"metatrain.{model}")
    # Get hyperparameters specifications for model and trainer
    model_hypers = get_hypers_annotation(module.__model__)
    trainer_hypers = get_hypers_annotation(module.__trainer__)

    assert module.__file__ is not None  # for mypy
    model_dir = Path(module.__file__).parent

    # Create the dictionary with all default hyperparameters
    yaml_defaults = {
        "architecture": {
            "name": model,
            "model": init_with_defaults(model_hypers),
            "training": init_with_defaults(trainer_hypers),
        },
    }
    # And write them to a YAML file
    with open(model_dir / "default-hypers.yaml", "w") as f:
        f.write("# This file is auto-generated. Do not edit directly.\n")

        yaml.dump(yaml_defaults, f, sort_keys=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate YAML file with defaults for a given architecture"
    )
    parser.add_argument(
        "model", type=str, help="The architecture to generate hyperparameters for."
    )
    args = parser.parse_args()

    write_hypers_files(args.model)
