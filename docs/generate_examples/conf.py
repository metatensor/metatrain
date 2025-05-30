# Pseudo-sphinx configuration to run sphinx-gallery as a command line tool

import os


extensions = [
    "sphinx_gallery.gen_gallery",
]

HERE = os.path.dirname(__file__)
ROOT = os.path.realpath(os.path.join(HERE, "..", ".."))

sphinx_gallery_conf = {
    "filename_pattern": r"/*\.py",
    "copyfile_regex": r".*\.(pt|sh|xyz|yaml)",
    "ignore_pattern": r"train\.sh",
    "example_extensions": {".py", ".sh"},
    "default_thumb_file": os.path.join(ROOT, "docs/src/logo/metatrain-512.png"),
    "examples_dirs": [
        os.path.join(ROOT, "examples", "ase"),
        os.path.join(ROOT, "examples", "programmatic", "llpr"),
        os.path.join(ROOT, "examples", "zbl"),
        os.path.join(ROOT, "examples", "programmatic", "use_architectures_outside"),
        os.path.join(ROOT, "examples", "programmatic", "disk_dataset"),
        os.path.join(ROOT, "examples", "basic_usage"),
    ],
    "gallery_dirs": [
        os.path.join(ROOT, "docs", "src", "examples", "ase"),
        os.path.join(ROOT, "docs", "src", "examples", "programmatic", "llpr"),
        os.path.join(ROOT, "docs", "src", "examples", "zbl"),
        os.path.join(ROOT, "docs", "src", "examples", "programmatic", "use_architectures_outside"),
        os.path.join(ROOT, "docs", "src", "examples", "programmatic", "disk_dataset"),
        os.path.join(ROOT, "docs", "src", "examples", "basic_usage"),
    ],
    "min_reported_time": 5,
    "matplotlib_animations": True,
}
