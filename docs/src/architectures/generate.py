import ast
from pathlib import Path
from typing import TypedDict

from jinja2 import Environment, FileSystemLoader

from metatrain.utils.architectures import (
    find_all_architectures,
    get_architecture_path,
    get_hypers_classes,
    preload_documentation_module,
    write_hypers_yaml,
)
from metatrain.utils.hypers import get_hypers_list


ARCHITECTURES_DIR = Path(__file__).parent
TEMPLATES_DIR = ARCHITECTURES_DIR / "templates"
DEFAULT_HYPERS_DIR = ARCHITECTURES_DIR / "default_hypers"
GENERATED_DIR = ARCHITECTURES_DIR / "generated"


JINJA_ENV = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
)


SECTIONS = [
    "installation",
    "default_hypers",
    "model_hypers",
    "trainer_hypers",
    "references",
]


class ArchitectureDocVariables(TypedDict):
    """Variables to use inside the architecture documentation.

    The docstring of the architecture will be processed as a
    ``jinja`` template. You can find documentation about them
    `here <https://jinja.palletsprojects.com/en/stable/templates>`_ , but
    the simplest functionality consists of using variables enclosed in
    double curly braces ``{{variable_name}}``, which will be replaced by
    their corresponding value.

    For example, a file with the following content:

    .. code-block:: rst

        This is the documentation for {{architecture}}.

    generates a documentation file that for the architecture ``pet`` would be:

    .. code-block:: rst

        This is the documentation for pet.

    There are some special variables that start with ``SECTION_``. These contain
    the content of different sections of the documentation, and they will be
    appended to the docstring if they are not already present. For example, given
    the docstring:

    .. code-block:: python

        \"""
        My architecture
        ===============

        This is my architecture.

        {{SECTION_DEFAULT_HYPERS}}

        Some important section
        ======================

        Explain something important here.
        \"""

    The final documentation will append to the docstring all the sections except
    ``SECTION_DEFAULT_HYPERS``, since it is already present.

    Following you can find a description of all the available variables. The
    sections are appended in the order documented here.
    """

    SECTION_INSTALLATION: str
    """Section containing installation instructions for this architecture."""
    SECTION_DEFAULT_HYPERS: str
    """Section containing a yaml file with the default hyperparameters for
    this architecture."""
    SECTION_MODEL_HYPERS: str
    """Section containing the description of the model hyperparameters for
    this architecture."""
    SECTION_TRAINER_HYPERS: str
    """Section containing the description of the trainer hyperparameters for
    this architecture."""
    SECTION_REFERENCES: str
    """Section containing references for this architecture. It will render the
    references that have been used as ``:footcite:p:`` during the architecture
    documentation."""

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

    E.g.: ``"metatrain.pet.documentation.ModelHypers"``
    """
    trainer_hypers_path: str
    """The full python import path to the trainer's hypers class of this
    architecture.

    E.g.: ``"metatrain.pet.documentation.TrainerHypers"``
    """
    model_hypers: list[str]
    """List of model hyperparameter names for this architecture."""
    trainer_hypers: list[str]
    """List of trainer hyperparameter names for this architecture."""


def setup_architectures_docs():
    """Generate the architecture documentation files.

    This function goes through all available architectures, and for each of them
    generates a yaml file with the default hyperparameters (so that it can be
    easily included in the documentation) and their rst documentation file.

    See :ref:`newarchitecture-documentation-page` for more information.
    """
    # If the default_hypers directory does not exist, create it
    DEFAULT_HYPERS_DIR.mkdir(exist_ok=True)
    # Same for the generated directory
    GENERATED_DIR.mkdir(exist_ok=True)

    for architecture_name in find_all_architectures():
        # Load documentation module in an isolated way to avoid
        # requiring dependencies for every architecture.
        preload_documentation_module(architecture_name)

        architecture_real_name = architecture_name.replace("experimental.", "").replace(
            "deprecated.", ""
        )

        # Write default hypers file
        yaml_path = DEFAULT_HYPERS_DIR / f"{architecture_real_name}-default-hypers.yaml"
        write_hypers_yaml(architecture_name, yaml_path)

        generate_rst(architecture_name, yaml_path=yaml_path)


def generate_rst(
    architecture_name: str,
    yaml_path: Path,
):
    """Generate the rst documentation file for a given architecture.

    :param architecture_name: The name of the architecture to generate the
        documentation for.
    :param yaml_path: Path to the yaml file with the default hyperparameters
        for this architecture.
    """
    # Get the name of the architecture without any prefix.
    architecture_real_name = architecture_name.replace("experimental.", "").replace(
        "deprecated.", ""
    )

    # Get the full python import path to the architecture
    arch_path = f"metatrain.{architecture_name}"

    # Get the docstring from the documentation.py file
    doc_file = get_architecture_path(architecture_name) / "documentation.py"
    with open(doc_file, "r") as f:
        module = ast.parse(f.read(), filename=str(doc_file))
        docstring = ast.get_docstring(module)
        if docstring is None:
            raise ValueError(
                f"The documentation.py file for architecture "
                f"'{architecture_name}' does not have a module docstring."
            )

    hypers_classes = get_hypers_classes(architecture_name)

    # Prepare template variables
    template_variables = dict(
        architecture=architecture_real_name,
        architecture_path=arch_path,
        default_hypers_path=".." / yaml_path.relative_to(ARCHITECTURES_DIR),
        model_hypers_path=f"{arch_path}.documentation.ModelHypers",
        trainer_hypers_path=f"{arch_path}.documentation.TrainerHypers",
        model_hypers=get_hypers_list(hypers_classes["model"]),
        trainer_hypers=get_hypers_list(hypers_classes["trainer"]),
    )

    # Read section templates and render them
    for section in SECTIONS:
        template = JINJA_ENV.get_template(f"{section}.rst")
        template_variables[f"SECTION_{section.upper()}"] = template.render(
            **template_variables
        )

    # Prepend docstring with reference and append missing sections
    docstring = f".. _arch-{template_variables['architecture']}:" + "\n\n" + docstring
    # Check for missing sections and add them to the end of the docstring
    for section in SECTIONS:
        section_var = "{{SECTION_" + section.upper() + "}}"
        if section_var not in docstring:
            docstring += f"\n\n{section_var}"

    # Render docstring template
    docstring = JINJA_ENV.from_string(docstring).render(**template_variables)

    # Write to file
    with open(GENERATED_DIR / f"{architecture_real_name}.rst", "w") as f:
        f.write(docstring + "\n")
