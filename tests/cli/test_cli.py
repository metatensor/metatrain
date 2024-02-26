"""Tests for argument parsing."""

import shutil
import subprocess
from pathlib import Path
from typing import List

import pytest


COMPFILE = (
    Path(__file__).parents[2]
    / "src/metatensor/models/share/metatensor-models-completion.bash"
)


def test_required_args():
    """Test required arguments."""
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_call(["metatensor-models"])


def test_wrong_module():
    """Test wrong module."""
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_call(["metatensor-models", "foo"])


@pytest.mark.parametrize("module", tuple(["eval", "export", "train"]))
def test_available_modules(module):
    """Test available modules."""
    subprocess.check_call(["metatensor-models", module, "--help"])


@pytest.mark.parametrize("args", ("version", "help"))
def test_extra_options(args):
    """Test extra options."""
    subprocess.check_call(["metatensor-models", "--" + args])


def test_debug_flag():
    """Test that even if debug flag is set commands run normal."""
    subprocess.check_call(["metatensor-models", "--debug", "train", "-h"])


def test_shell_completion_flag():
    """Test that path to the `shell-completion` is correct."""
    completion_path = subprocess.check_output(
        ["metatensor-models", "--shell-completion"]
    )

    assert Path(completion_path.decode("ascii")).is_file


# TODO: There seems to be an issue with zsh, Github CI and subprocesses.
@pytest.mark.parametrize(
    "shell",
    [
        "bash",
        pytest.param("zsh", marks=pytest.mark.xfail(reason="Github CI - zsh issue")),
    ],
)
def test_syntax_completion(shell):
    """Test that the completion can be sourced"""
    subprocess.check_call(
        args=[
            shutil.which(shell),
            "-i",
            "-c",
            "source $(metatensor-models --shell-completion)",
        ],
    )


def get_completion_suggestions(partial_word: str) -> List[str]:
    """Suggestions of a simulated <tab> completion of a metatensor-models subcommand.

    https://stackoverflow.com/questions/9137245/unit-test-for-bash-completion-script
    """
    cmd = ["metatensor-models", partial_word]
    cmdline = " ".join(cmd)

    out = subprocess.Popen(
        args=[
            "bash",
            "-i",
            "-c",
            r'source {compfile}; COMP_LINE="{cmdline}" COMP_WORDS=({cmdline}) '
            r"COMP_CWORD={cword} COMP_POINT={cmdlen} $(complete -p {cmd} | "
            r'sed "s/.*-F \\([^ ]*\\) .*/\\1/") && echo ${{COMPREPLY[*]}}'.format(
                compfile=str(COMPFILE),
                cmdline=cmdline,
                cmdlen=len(cmdline),
                cmd=cmd[0],
                cword=cmd.index(partial_word),
            ),
        ],
        stdout=subprocess.PIPE,
    )
    stdout, _ = out.communicate()
    return stdout.decode("ascii").split()


@pytest.mark.parametrize(
    "partial_word, expected_completion",
    [(" ", ["--debug", "--help", "--version", "-h", "eval", "export", "train"])],
)
def test_subcommand_completion(partial_word, expected_completion):
    """Test that expected subcommand completion matches."""
    assert set(get_completion_suggestions(partial_word)) == set(expected_completion)
