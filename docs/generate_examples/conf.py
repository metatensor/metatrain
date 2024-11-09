# Pseudo-sphinx configuration to run sphinx-gallery as a command line tool

import os


extensions = [
    "sphinx_gallery.gen_gallery",
]

HERE = os.path.dirname(__file__)
ROOT = os.path.realpath(os.path.join(HERE, "..", ".."))

sphinx_gallery_conf = {
    "filename_pattern": "/*",
    "copyfile_regex": r".*\.(pt|sh|xyz|yaml)",
    "examples_dirs": [
        os.path.join(ROOT, "examples", "ase"),
        os.path.join(ROOT, "examples", "programmatic", "llpr"),
        os.path.join(ROOT, "examples", "zbl")
    ],
    "gallery_dirs": [
        os.path.join(ROOT, "docs", "src", "examples", "ase"),
        os.path.join(ROOT, "docs", "src", "examples", "programmatic", "llpr"),
        os.path.join(ROOT, "docs", "src", "examples", "zbl")
    ],
    "min_reported_time": 5,
    "matplotlib_animations": True,
}
