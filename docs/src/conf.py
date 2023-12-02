# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'metatensor-models'
copyright = '2023, metatensor-models developers'
author = 'metatensor-models developers'

# Parse the version from the module.
with open(os.path.join(os.path.dirname(__file__), '..', '..', 'metatensor_models', '__init__.py')) as f:
    version = re.match(r'.*__version__ = \'(.*?)\'', f.read(), re.S).group(1)

release = version


# -- General configuration ---------------------------------------------------

needs_sphinx = "4.0.0"

python_use_unqualified_type_names = True

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_dark_mode',
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_toggleprompt",
]

default_dark_mode = True

autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "both"
autodoc_typehints_format = "short"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
