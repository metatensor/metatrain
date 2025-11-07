import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import tomli  # Replace by tomllib from std library once docs are build with Python 3.11


# When importing metatensor-torch, this will change the definition of the classes
# to include the documentation
os.environ["METATENSOR_IMPORT_FOR_SPHINX"] = "1"
os.environ["METATOMIC_IMPORT_FOR_SPHINX"] = "1"
os.environ["PYTORCH_JIT"] = "0"
os.environ["METATENSOR_DEBUG_EXTENSIONS_LOADING"] = "1"

import metatrain  # noqa: E402
from metatrain.utils.architectures import (
    find_all_architectures,
    import_architecture,
    write_hypers_yaml,
)  # noqa: E402
from metatrain.utils.hypers import get_hypers_cls  # noqa: E402


ROOT = os.path.abspath(os.path.join("..", ".."))

# We use a second (pseudo) sphinx project located in `docs/generate_examples` to run the
# examples and generate the actual output for our shinx-gallery. This is necessary
# because here we have to set `METATENSOR_IMPORT_FOR_SPHINX` and
# `METATOMIC_IMPORT_FOR_SPHINX` to `"1"` allowing the correct generation of the class
# and function docstrings which are seperate from the actual code.
#
# We register and use the same sphinx gallery configuration as in the pseudo project.
sys.path.append(os.path.join(ROOT, "docs"))
from generate_examples.conf import sphinx_gallery_conf  # noqa
from src.architectures.generate import setup_architectures_docs  # noqa


# -- Project information -----------------------------------------------------

# The master toctree document.
master_doc = "index"

with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fp:
    project_dict = tomli.load(fp)["project"]

project = project_dict["name"]
author = ", ".join(a["name"] for a in project_dict["authors"])

copyright = f"{datetime.now().date().year}, {author}"

# The full version, including alpha/beta/rc tags
release = metatrain.__version__


# -- General configuration ---------------------------------------------------


def generate_examples():
    # we can not run sphinx-gallery in the same process as the normal sphinx, since they
    # need to import metatensor.torch and metatomic.torch differently (with and without
    # {METATENSOR/METATOMIC}_IMPORT_FOR_SPHINX=1). So instead we run it inside a small
    # script, and include the corresponding output later.
    del os.environ["METATENSOR_IMPORT_FOR_SPHINX"]
    del os.environ["METATOMIC_IMPORT_FOR_SPHINX"]
    del os.environ["PYTORCH_JIT"]
    script = os.path.join(ROOT, "docs", "generate_examples", "generate-examples.py")
    subprocess.run([sys.executable, script])
    os.environ["METATENSOR_IMPORT_FOR_SPHINX"] = "1"
    os.environ["METATOMIC_IMPORT_FOR_SPHINX"] = "1"
    os.environ["PYTORCH_JIT"] = "0"


def setup_architectures_docs():
    """Generate the architecture documentation files.

    This function goes through all available architectures, and for each of them
    generates a yaml file with the default hyperparameters (so that it can be
    easily included in the documentation). Also, if there is no documentation
    rst file for a given architecture, this function auto-generates one based
    on a template.
    """
    # Get paths to directories
    architectures_dir = Path(__file__).parent / "architectures"
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

        # Fill template and write file
        with open(generated_dir / f"{architecture_real_name}.rst", "w") as f:
            f.write(
                template.format(
                    architecture=architecture_real_name,
                    architecture_path="metatrain." + architecture_name,
                    default_hypers_path=".." / yaml_path.relative_to(architectures_dir),
                    model_hypers_path=model_hypers.__module__
                    + "."
                    + model_hypers.__name__,
                    trainer_hypers_path=trainer_hypers.__module__
                    + "."
                    + trainer_hypers.__name__,
                )
            )


def setup(app):
    setup_architectures_docs()
    generate_examples()
    setup_architectures_docs()


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_sitemap",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_toggleprompt",
    "sphinx_gallery.gen_gallery",
    "chemiscope.sphinx",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "Thumbs.db",
    ".DS_Store",
    "examples/sg_execution_times.rst",
    "examples/ase/index.rst",
    "sg_execution_times.rst",
    "architectures/templates/*",
    "architectures/README.md",
]


python_use_unqualified_type_names = True

autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "both"
autodoc_typehints_format = "short"
autodoc_type_aliases = {
    "SomeType": "int",
}

intersphinx_mapping = {
    "ase": ("https://ase-lib.org/", None),
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "metatensor": ("https://docs.metatensor.org/latest/", None),
    "metatomic": ("https://docs.metatensor.org/metatomic/latest/", None),
    "omegaconf": ("https://omegaconf.readthedocs.io/en/latest/", None),
    "pydantic": ("https://docs.pydantic.dev/latest", None),
}

# The path to the bibtex file
bibtex_bibfiles = ["../static/refs.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

# sitemap/SEO settings
html_baseurl = "https://docs.metatensor.org/metatrain/latest/"  # prefix for the sitemap
sitemap_url_scheme = "{link}"  # avoids language settings
html_extra_path = ["robots.txt"]  # extra files to move

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = [os.path.join(ROOT, "docs", "static")]
html_favicon = "logo/metatrain-64.png"

html_theme_options = {
    "light_logo": "images/metatrain-horizontal.png",
    "dark_logo": "images/metatrain-horizontal-dark.png",
    "sidebar_hide_name": True,
    "footer_icons": [
        {
            "name": "GitHub",
            "url": project_dict["urls"]["repository"],
            "html": "",
            "class": "fa-brands fa-github fa-2x",
        },
    ],
}

# font-awesome logos (used in the footer)
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
    "styles.css",
]
