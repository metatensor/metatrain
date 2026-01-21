# Pseudo-sphinx configuration to run sphinx-gallery as a command line tool

import os

from chemiscope.sphinx import ChemiscopeScraper


extensions = [
    "sphinx_gallery.gen_gallery",
]

HERE = os.path.dirname(__file__)
ROOT = os.path.realpath(os.path.join(HERE, "..", ".."))

sphinx_gallery_conf = {
    "filename_pattern": r"/*\.py",
    "copyfile_regex": r".*\.(pt|sh|xyz|yaml)",
    "example_extensions": {".py", ".sh"},
    "default_thumb_file": os.path.join(ROOT, "docs/src/logo/metatrain-512.png"),
    "examples_dirs": "../../examples",
    "gallery_dirs": "generated_examples",
    "min_reported_time": 5,
    "matplotlib_animations": True,
    "image_scrapers": ["matplotlib", ChemiscopeScraper()],
    "remove_config_comments": True,
    "within_subsection_order": "FileNameSortKey",
}
