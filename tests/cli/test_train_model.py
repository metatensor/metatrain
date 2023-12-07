import shutil
import subprocess
from pathlib import Path


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / "resources"


def test_train(monkeypatch, tmp_path):
    """Test that training via the training cli runs without an error raise."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")
    shutil.copy(RESOURCES_PATH / "parameters.yaml", "parameters.yaml")
    subprocess.check_call(
        [
            "metatensor-models",
            "train",
            "--config-dir=.",
            "--config-name=parameters.yaml",
        ]
    )
