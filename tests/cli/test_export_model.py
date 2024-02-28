"""Test command line interface for the export functions.

Actual unit tests for the function are performed in `tests/utils/test_export`.
"""

import subprocess
from pathlib import Path

import pytest


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / "resources"


@pytest.mark.parametrize("output", [None, "exported.pt"])
def test_export(monkeypatch, tmp_path, output):
    """Test that the export cli runs without an error raise."""
    monkeypatch.chdir(tmp_path)
    command = ["metatensor-models", "export", str(RESOURCES_PATH / "bpnn-model.ckpt")]

    if output is not None:
        command += ["-o", output]
    else:
        output = "exported-model.pt"

    subprocess.check_call(command)
    assert Path(output).is_file()
