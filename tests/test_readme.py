import re
import shutil
import subprocess
from pathlib import Path

from .conftest import DATASET_PATH_QM9


README_PATH = Path(__file__).parent.parent / "README.md"


def test_train_readme_example(monkeypatch, tmp_path):
    """Test that training runs with the example in the README."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")

    # Parse the README to extract the training example
    with open(README_PATH, "r", encoding="utf-8") as f:
        readme = f.read()

    match = re.search(r"```yaml\n(.*?)```", readme, re.DOTALL)
    assert match, "Could not find training example in README"
    yaml_string = match.group(1)
    yaml_string = yaml_string.replace(
        "num_epochs: 5", "num_epochs: 1"
    )  # Reduce epochs for testing

    # Create a temporary options.yaml file
    with open("options.yaml", "w") as f:
        f.write(yaml_string)

    command = ["mtt", "train", "options.yaml"]
    subprocess.check_call(command)
