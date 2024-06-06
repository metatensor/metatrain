import argparse

from metatrain.cli.formatter import CustomHelpFormatter


def test_formatter(capsys):
    """Test that positonal arguments are displayed before optional in usage."""
    parser = argparse.ArgumentParser(prog="myprog", formatter_class=CustomHelpFormatter)
    parser.add_argument("required_input")
    parser.add_argument("required_input2")
    parser.add_argument("-f", "--foo", help="optional argument")
    parser.add_argument("-b", "--bar", help="optional argument 2")

    parser.print_help()

    captured = capsys.readouterr()
    assert (
        "usage: myprog required_input required_input2 [-h] [-f FOO] [-b BAR]"
        in captured.out
    )
