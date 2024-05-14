from pathlib import Path
import secrets

PACKAGE_ROOT = Path(__file__).parent.resolve()

CONFIG_PATH = PACKAGE_ROOT / "cli" / "conf"
ARCHITECTURE_CONFIG_PATH = CONFIG_PATH / "architecture"


# A constant as "session" variable to set the random seed to a fixed value that do not
# change within the execution of the program.
RANDOM_SEED = secrets.randbelow(2**32)

__version__ = "2023.11.29"
