import subprocess
from pathlib import Path

import pytest


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / "resources"


@pytest.mark.parametrize("output", [None, "exported.pt"])
def test_export(monkeypatch, tmp_path, output):
    """Test that the export cli runs without an error raise."""
    monkeypatch.chdir(tmp_path)
    command = ["metatensor-models", "export", str(RESOURCES_PATH / "bpnn-model.pt")]

    if output is not None:
        command += ["-o", output]
    else:
        output = "exported-model.pt"

    subprocess.check_call(command)
    assert Path(output).is_file()


def test_export_warning(monkeypatch, tmp_path):
    """Test that the export cli raises an error when no units are present."""

    monkeypatch.chdir(tmp_path)

    out = subprocess.check_output(
        ["metatensor-models", "export", str(RESOURCES_PATH / "bpnn-model.pt")],
        stderr=subprocess.STDOUT,
    )

    assert "No units were provided" in str(out)
