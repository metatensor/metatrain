from pathlib import Path
import torch

PACKAGE_ROOT = Path(__file__).parent.resolve()

CONFIG_PATH = PACKAGE_ROOT / "cli" / "conf"
ARCHITECTURE_CONFIG_PATH = CONFIG_PATH / "architecture"

__version__ = "2023.11.29"

torch.set_default_dtype(torch.float64)
