import importlib
from collections import namedtuple


DepStatus = namedtuple("DepStatus", ["present", "message"])
if importlib.util.find_spec("wandb"):
    WANDB_AVAILABLE = DepStatus(True, "present")
else:
    WANDB_AVAILABLE = DepStatus(False, "wandb not installed")
