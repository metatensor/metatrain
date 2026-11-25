import subprocess
import sys
from pathlib import Path

import pytest


pytest.importorskip("licensecheck")  # only installed on Python >= 3.12

import tomllib  # noqa: E402


ROOT = Path(__file__).parents[1]

# These packages do not declare a license yet
IGNORED = ["torch-spex", "physical-basis"]


def test_dependency_licenses():
    """All dependencies, including extras, have BSD-3-Clause compatible licenses."""
    with open(ROOT / "pyproject.toml", "rb") as fd:
        extras = tomllib.load(fd)["project"]["optional-dependencies"]

    result = subprocess.run(
        [sys.executable, "-m", "licensecheck", "--zero", "--show-only-failing"]
        + ["--extras", *extras, "--ignore-packages", *IGNORED],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout
