from pathlib import Path
from typing import Dict


DEFAULT_HYPERS: Dict = {"model": {}}
DATASET_PATH = str(Path(__file__).parents[4] / "tests/resources/qm9_reduced_100.xyz")
