import logging
import re

from metatensor.models.utils.logging import setup_logging


def assert_log_entry(logtext: str, loglevel: str, message: str) -> None:
    pattern = (
        r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\["
        + re.escape(loglevel.strip())
        + r"\] - "
        + re.escape(message)
    )

    if re.match(pattern, logtext) is None:
        raise AssertionError(f"{message!r} and {loglevel!r} not found in {logtext!r}")


def test_default_log(caplog, capsys):
    """Default message only in STDOUT."""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("testing")

    with setup_logging(logger, level=logging.INFO):
        logger.info("foo")
        logger.debug("A debug message")

    stdout_log = capsys.readouterr().out

    assert "Logging to file is disabled." in caplog.text
    assert_log_entry(stdout_log, loglevel="INFO", message="foo")
    assert "A debug message" not in stdout_log


def test_info_log(caplog, monkeypatch, tmp_path, capsys):
    """Default message in STDOUT and file."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test")

    with setup_logging(logger, logfile="logfile.log", level=logging.INFO):
        logger.info("foo")
        logger.debug("A debug message")

    with open("logfile.log", "r") as f:
        file_log = f.read()

    stdout_log = capsys.readouterr().out

    assert "This log is also available in 'logfile.log'" in caplog.text

    assert file_log == stdout_log

    for logtext in [stdout_log, file_log]:
        assert_log_entry(logtext, loglevel="INFO", message="foo")
        assert "A debug message" not in logtext


def test_debug_log(caplog, monkeypatch, tmp_path, capsys):
    """Debug message in STDOUT and file."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.DEBUG)
    logger = logging.getLogger("test")

    with setup_logging(logger, logfile="logfile.log", level=logging.DEBUG):
        logger.info("foo")
        logger.debug("A debug message")

    with open("logfile.log", "r") as f:
        file_log = f.read()

    stdout_log = capsys.readouterr().out

    assert file_log == stdout_log
    assert "This log is also available in 'logfile.log'" in caplog.text

    for logtext in [stdout_log, file_log]:
        assert "foo" in logtext
        assert "A debug message" in logtext
        # Test that debug information is in output
        assert "test_logging.py:test_debug_log:67" in logtext
