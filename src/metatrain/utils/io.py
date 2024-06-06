import warnings
from pathlib import Path
from typing import Union


def check_suffix(filename: Union[str, Path], suffix: str) -> Union[str, Path]:
    """Check the suffix of a file name and adds if it not existing.

    If ``filename`` does not end with ``suffix`` the ``suffix`` is added and a warning
    will be issued.

    :param filename: Name of the file to be checked.
    :param suffix: Expected filesuffix i.e. ``.txt``.
    :returns: Checked and probably extended file name.
    """
    path_filename = Path(filename)

    if path_filename.suffix != suffix:
        warnings.warn(
            f"The file name should have a '{suffix}' extension. The user "
            f"requested the file with name '{filename}', but it will be saved as "
            f"'{filename}{suffix}'.",
            stacklevel=1,
        )
        path_filename = path_filename.parent / (path_filename.name + suffix)

    if type(filename) is str:
        return str(path_filename)
    else:
        return path_filename
