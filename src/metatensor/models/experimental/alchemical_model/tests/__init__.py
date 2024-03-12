from pathlib import Path

DATASET_PATH = str(
    Path(__file__).parent.resolve()
    / "../../../../../../tests/resources/qm9_reduced_100.xyz"
)

ALCHEMICAL_DATASET_PATH = str(
    Path(__file__).parent.resolve()
    / "../../../../../../tests/resources/alchemical_reduced_10.xyz"
)
