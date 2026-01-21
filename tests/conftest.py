import math
import os
import subprocess
from pathlib import Path

import pytest

from metatrain.utils.architectures import get_default_hypers


def pytest_xdist_auto_num_workers():
    """Limit the number of workers used by pytest"""
    n_processes = os.cpu_count() or 1
    return min(12, math.ceil(n_processes * 0.8))

MODEL_HYPERS = get_default_hypers("soap_bpnn")["model"]

# -------------------------------
#      PATHS TO RESOURCES
# -------------------------------
RESOURCES_PATH = Path(__file__).parent / "resources"

DATASET_PATH_QM9 = RESOURCES_PATH / "qm9_reduced_100.xyz"
DATASET_PATH_ETHANOL = RESOURCES_PATH / "ethanol_reduced_100.xyz"
DATASET_PATH_CARBON = RESOURCES_PATH / "carbon_reduced_100.xyz"
DATASET_PATH_QM7X = RESOURCES_PATH / "qm7x_reduced_100.xyz"
DATASET_PATH_DOS = RESOURCES_PATH / "dos_100.xyz"
EVAL_OPTIONS_PATH = RESOURCES_PATH / "eval.yaml"
OPTIONS_PATH = RESOURCES_PATH / "options.yaml"
OPTIONS_PET_PATH = RESOURCES_PATH / "options-pet.yaml"
OPTIONS_EXTRA_DATA_PATH = RESOURCES_PATH / "options-extra-data.yaml"

# -------------------------------
#    PATHS TO TRAINED MODELS
# -------------------------------
# These files are generated from training on each test run 
# and take some time to generate. Therefore, we make them a
# fixture so that they are generated lazily only if there
# is a test that requires them. Since we support parallel
# test runs, we use the unique identifier for the test run to
# so that the multiple workers do not replicate work and wait
# for each other.
def ensure_path(mode: str, uid: str) -> Path:
    """Checks if the path for a model exists, and if not
    runs the training script to generate it."""
    path = RESOURCES_PATH / f"model-{mode}-{uid}.pt"
    if not path.exists():
        subprocess.run(
            ["bash", str(RESOURCES_PATH / "run_trainings.sh"), mode, uid], check=True
        )
    return path


@pytest.fixture(scope="session")
def MODEL_PATH(testrun_uid) -> Path:
    return ensure_path("32-bit", testrun_uid)


@pytest.fixture(scope="session")
def MODEL_PATH_64_BIT(testrun_uid) -> Path:
    return ensure_path("64-bit", testrun_uid)


@pytest.fixture(scope="session")
def MODEL_PATH_PET(testrun_uid) -> Path:
    return ensure_path("pet", testrun_uid)
