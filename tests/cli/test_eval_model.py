import shutil
import subprocess
from pathlib import Path

import ase.io
import pytest


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / "resources"


@pytest.mark.parametrize("output", [None, "foo.xyz"])
def test_eval(output, monkeypatch, tmp_path):
    """Test that training via the training cli runs without an error raise."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")
    shutil.copy(RESOURCES_PATH / "bpnn-model.pt", "bpnn-model.pt")

    command = [
        "metatensor-models",
        "eval",
        "-m",
        "bpnn-model.pt",
        "-s",
        "qm9_reduced_100.xyz",
    ]

    if output is not None:
        command += ["-o", output]
    else:
        output = "output.xyz"

    subprocess.check_call(command)

    frames = ase.io.read(output, ":")
    frames[0].info["energy"]
