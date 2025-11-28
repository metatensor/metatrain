from collections import namedtuple
from importlib.util import find_spec


DepStatus = namedtuple("DepStatus", ["present", "message"])
if find_spec("wandb"):
    WANDB_AVAILABLE = DepStatus(True, "present")
else:
    WANDB_AVAILABLE = DepStatus(False, "wandb not installed")
