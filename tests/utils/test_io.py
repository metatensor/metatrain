from pathlib import Path

import pytest

from metatrain.utils.io import check_file_extension


@pytest.mark.parametrize("filename", ["example.txt", Path("example.txt")])
def test_check_suffix(filename):
    result = check_file_extension(filename, ".txt")

    assert str(result) == "example.txt"
    assert isinstance(result, type(filename))


@pytest.mark.parametrize("filename", ["example", Path("example")])
def test_warning_on_missing_suffix(filename):
    match = r"The file name should have a '\.txt' file extension."
    with pytest.warns(UserWarning, match=match):
        result = check_file_extension(filename, ".txt")

    assert str(result) == "example.txt"
    assert isinstance(result, type(filename))
