import secrets
from pathlib import Path

from ._version import __version__  # noqa: F401


PACKAGE_ROOT = Path(__file__).parent.resolve()


# A constant as "session" variable to set the random seed to a fixed value that do not
# change within the execution of the program.
RANDOM_SEED = secrets.randbelow(2**32)
