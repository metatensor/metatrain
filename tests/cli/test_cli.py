import subprocess

import pytest


class Test_parse_args(object):
    """Tests for argument parsing."""

    def test_required_args(self):
        """Test required arguments."""
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(["metatensor-models"])

    def test_wrong_module(self):
        """Test wrong module."""
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(["metatensor-models", "foo"])

    @pytest.mark.parametrize("module", tuple(["eval", "export", "train"]))
    def test_available_modules(self, module):
        """Test available modules."""
        subprocess.check_call(["metatensor-models", module, "--help"])

    @pytest.mark.parametrize("args", ("version", "help"))
    def test_extra_options(self, args):
        """Test extra options."""
        subprocess.check_call(["metatensor-models", "--" + args])

    @pytest.mark.parametrize("args", ("version", "help"))
    def test_debug_flag(self, args):
        """Test that even if debug flag is set commands run normal."""
        subprocess.check_call(["metatensor-models", "--debug", "train", "-h"])
