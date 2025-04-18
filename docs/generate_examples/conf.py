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
    "default_thumb_file": os.path.join(ROOT, "docs/src/logo/metatrain-512.png"),
    "examples_dirs": [
        os.path.join(ROOT, "examples", "ase"),
        os.path.join(ROOT, "examples", "programmatic", "llpr"),
        os.path.join(ROOT, "examples", "zbl"),
        os.path.join(ROOT, "examples", "programmatic", "use_architectures_outside"),
        os.path.join(ROOT, "examples", "programmatic", "disk_dataset"),
    ],
    "gallery_dirs": [
        os.path.join(ROOT, "docs", "src", "examples", "ase"),
        os.path.join(ROOT, "docs", "src", "examples", "programmatic", "llpr"),
        os.path.join(ROOT, "docs", "src", "examples", "zbl"),
        os.path.join(ROOT, "docs", "src", "examples", "programmatic", "use_architectures_outside"),
        os.path.join(ROOT, "docs", "src", "examples", "programmatic", "disk_dataset"),
    ],
    "min_reported_time": 5,
    "matplotlib_animations": True,
}
