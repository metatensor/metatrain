import ast
from pathlib import Path
from typing import TypedDict

from jinja2 import Environment, FileSystemLoader

from metatrain.utils import hooks as hooks_module
from metatrain.utils.hooks.helpers import (
    find_all_hooks,
    get_hypers_class,
    preload_documentation_module,
    write_hypers_yaml,
)
from metatrain.utils.hypers import get_hypers_list


HOOKS_DIR = Path(__file__).parent
TEMPLATES_DIR = HOOKS_DIR / "templates"
DEFAULT_HYPERS_DIR = HOOKS_DIR / "default_hypers"
GENERATED_DIR = HOOKS_DIR / "generated"


JINJA_ENV = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
)


SECTIONS = [
    "installation",
    "hook_hypers",
    "references",
]


class HookDocVariables(TypedDict):
    """Variables to use inside the hook documentation.

    The docstring of the hook will be processed as a
    ``jinja`` template. You can find documentation about them
    `here <https://jinja.palletsprojects.com/en/stable/templates>`_ , but
    the simplest functionality consists of using variables enclosed in
    double curly braces ``{{variable_name}}``, which will be replaced by
    their corresponding value.

    For example, a file with the following content:

    .. code-block:: rst

        This is the documentation for {{hook}}.

    generates a documentation file that for the hook ``tensor_basis`` would be:

    .. code-block:: rst

        This is the documentation for tensor_basis.

    There are some special variables that start with ``SECTION_``. These contain
    the content of different sections of the documentation, and they will be
    appended to the docstring if they are not already present. For example, given
    the docstring:

    .. code-block:: python

        \"""
        My hook
        =======

        This is my hook.

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
    """Section containing installation instructions for this hook."""
    SECTION_HOOK_HYPERS: str
    """Section containing the description of the hook hyperparameters for
    this hook."""
    SECTION_REFERENCES: str
    """Section containing references for this hook. It will render the
    references that have been used as ``:footcite:p:`` during the hook
    documentation."""

    hook: str
    """The name of the hook.

    This excludes any 'experimental.' or 'deprecated.' prefix."""
    default_hypers_path: str
    """Path to the yaml file with the default hyperparameters for this
    hook.

    This is a path relative to the ``docs/src/hooks/generated``
    directory.
    """
    hook_hypers_path: str
    """The full python import path to the hook's hypers class of this
    hook.

    E.g.: ``"metatrain.utils.hooks.<hook_name>.Hypers"``
    """
    hook_hypers: list[str]
    """List of hyperparameter names for this hook."""


def setup_hooks_docs():
    """Generate the hook documentation files.

    This function goes through all available hooks, and for each of them
    generates a yaml file with the default hyperparameters (so that it can be
    easily included in the documentation) and their rst documentation file.

    See :ref:`newarchitecture-documentation-page` for more information.
    """
    # If the default_hypers directory does not exist, create it
    DEFAULT_HYPERS_DIR.mkdir(exist_ok=True)
    # Same for the generated directory
    GENERATED_DIR.mkdir(exist_ok=True)

    for name in find_all_hooks():
        # Load documentation module in an isolated way to avoid
        # requiring dependencies for every architecture.
        preload_documentation_module(name)

        # Write default hypers file
        yaml_path = DEFAULT_HYPERS_DIR / f"{name}-default-hypers.yaml"
        write_hypers_yaml(name, yaml_path, include_name=True)

        generate_rst(name, yaml_path=yaml_path)


def generate_rst(
    hook_name: str,
    yaml_path: Path,
):
    """Generate the rst documentation file for a given hook.

    :param hook_name: The name of the hook to generate the
        documentation for.
    :param yaml_path: Path to the yaml file with the default hyperparameters
        for this architecture.
    """

    # Get the full python import path to the hook
    hook_path = f"metatrain.utils.hooks.{hook_name}"

    # Get the docstring from the documentation.py file
    doc_file = Path(hooks_module.__file__).parent / hook_name / "documentation.py"
    with open(doc_file, "r") as f:
        module = ast.parse(f.read(), filename=str(doc_file))
        docstring = ast.get_docstring(module)
        if docstring is None:
            raise ValueError(
                f"The documentation.py file for hook "
                f"'{hook_name}' does not have a module docstring."
            )

    hypers_class = get_hypers_class(hook_name)

    # Prepare template variables
    template_variables = dict(
        hook=hook_name,
        default_hypers_path=".." / yaml_path.relative_to(HOOKS_DIR),
        hook_hypers_path=f"{hook_path}.documentation.Hypers",
        hook_hypers=get_hypers_list(hypers_class),
    )

    # Read section templates and render them
    for section in SECTIONS:
        template = JINJA_ENV.get_template(f"{section}.rst")
        template_variables[f"SECTION_{section.upper()}"] = template.render(
            **template_variables
        )

    # Check for missing sections and add them to the end of the docstring
    for section in SECTIONS:
        section_var = "{{SECTION_" + section.upper() + "}}"
        if section_var not in docstring:
            docstring += f"\n\n{section_var}"

    # Render docstring template
    docstring = JINJA_ENV.from_string(docstring).render(**template_variables)

    # Write to file
    with open(GENERATED_DIR / f"{hook_name}.rst", "w") as f:
        f.write(docstring + "\n")
