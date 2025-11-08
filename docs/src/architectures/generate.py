from pathlib import Path
from typing import TypedDict

from metatrain.utils.architectures import (
    find_all_architectures,
    import_architecture,
    write_hypers_yaml,
)
from metatrain.utils.hypers import get_hypers_cls


class ArchitectureTemplateVariables(TypedDict):
    """Variables to use inside the architecture documentation templates.

    These are applied using python's built-in :meth:`str.format` method,
    therefore in the text one should use ``{variable_1}`` to refer to
    ``variable_1``. For example, a file with the following content:

    .. code-block:: rst

        This is the documentation for {architecture}.

    generates a documentation file that for the architecture ``pet`` would be:

    .. code-block:: rst

        This is the documentation for pet.

    """

    architecture: str
    """The name of the architecture.

    This excludes any 'experimental.' or 'deprecated.' prefix."""
    architecture_path: str
    """The full python import path to the architecture.

    E.g.: ``"metatrain.experimental.my_architecture"``
    """
    default_hypers_path: str
    """Path to the yaml file with the default hyperparameters for this
    architecture.

    This is a path relative to the ``docs/src/architectures/generated``
    directory.
    """
    model_hypers_path: str
    """The full python import path to the model's hypers class of this
    architecture.

    E.g.: ``"metatrain.pet.hypers.PETHypers"``
    """
    trainer_hypers_path: str
    """The full python import path to the trainer's hypers class of this
    architecture.

    E.g.: ``"metatrain.pet.hypers.PETTrainerHypers"``
    """


def setup_architectures_docs():
    """Generate the architecture documentation files.

    This function goes through all available architectures, and for each of them
    generates a yaml file with the default hyperparameters (so that it can be
    easily included in the documentation). Also, it takes the files
    ``<architecture_name>.rst`` from the ``templates`` directory, processes them,
    and writes the resulting documentation files to the ``generated`` directory.
    If no architecture-specific template is found, it falls back to using
    ``generic.rst`` as the template.
    """
    # Get paths to directories
    architectures_dir = Path(__file__).parent
    templates_dir = architectures_dir / "templates"
    generated_dir = architectures_dir / "generated"
    hypers_dir = architectures_dir / "default_hypers"

    # If the default_hypers directory does not exist, create it
    hypers_dir.mkdir(exist_ok=True)
    # Same for the generated directory
    generated_dir.mkdir(exist_ok=True)

    for architecture_name in find_all_architectures():
        architecture_real_name = architecture_name.replace("experimental.", "").replace(
            "deprecated.", ""
        )

        # Write default hypers file
        yaml_path = hypers_dir / f"{architecture_real_name}-default-hypers.yaml"
        write_hypers_yaml(architecture_name, yaml_path)

        # Generate the architecture rst file with its documentation
        architecture = import_architecture(architecture_name)
        model_hypers = get_hypers_cls(architecture.__model__)
        trainer_hypers = get_hypers_cls(architecture.__trainer__)

        # Get the template to use, first try an architecture-specific one,
        # then fall back to the generic one
        for template_name in [f"{architecture_real_name}.rst", "generic.rst"]:
            template_path = templates_dir / template_name
            if template_path.exists():
                template = template_path.read_text()
                break

        template_variables: ArchitectureTemplateVariables = dict(
            architecture=architecture_real_name,
            architecture_path="metatrain." + architecture_name,
            default_hypers_path=".." / yaml_path.relative_to(architectures_dir),
            model_hypers_path=f"{model_hypers.__module__}.{model_hypers.__name__}",
            trainer_hypers_path=f"{trainer_hypers.__module__}.{trainer_hypers.__name__}",
        )

        docs_content = template.format(**template_variables)

        # Fill template and write file
        with open(generated_dir / f"{architecture_real_name}.rst", "w") as f:
            f.write(docs_content)
