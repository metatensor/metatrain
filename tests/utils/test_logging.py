import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import List

import pytest
import wandb
from metatomic.torch import ModelCapabilities, ModelOutput

from metatrain import PACKAGE_ROOT
from metatrain.utils.logging import (
    CSVFileHandler,
    CustomLogger,
    MetricLogger,
    WandbHandler,
    get_cli_input,
    human_readable,
    setup_logging,
)


def assert_log_entry(logtext: str, loglevel: str, message: str) -> None:
    pattern = (
        r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\["
        + re.escape(loglevel.strip())
        + r"\] - "
        + re.escape(message)
    )

    if not re.search(pattern, logtext, re.MULTILINE):
        raise AssertionError(f"{message!r} and {loglevel!r} not found in {logtext!r}")


def test_default_log(caplog, capsys):
    """Default message only in STDOUT."""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger()

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
    logger = logging.getLogger()

    with setup_logging(logger, log_file="logfile.log", level=logging.INFO):
        logger.info("foo")
        logger.debug("A debug message")

    with open("logfile.log", "r") as f:
        file_log = f.read()

    log_path = str((tmp_path / "logfile.log").absolute())

    assert f"This log is also available at '{log_path}'" in caplog.text
    assert f"Package directory: {PACKAGE_ROOT}" in caplog.text
    assert f"Working directory: {Path('.').absolute()}" in caplog.text
    assert f"Executed command: {get_cli_input()}" in caplog.text

    stdout_log = capsys.readouterr().out
    assert file_log == stdout_log
    assert_log_entry(stdout_log, loglevel="INFO", message="foo")
    assert "A debug message" not in stdout_log


def test_debug_log(caplog, monkeypatch, tmp_path, capsys):
    """Debug message in STDOUT and file."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.DEBUG)
    logger = logging.getLogger()

    with setup_logging(logger, log_file="logfile.log", level=logging.DEBUG):
        logger.info("foo")
        logger.debug("A debug message")

    with open("logfile.log", "r") as f:
        file_log = f.read()

    stdout_log = capsys.readouterr().out
    log_path = str((tmp_path / "logfile.log").absolute())

    assert file_log == stdout_log
    assert f"This log is also available at '{log_path}'" in caplog.text

    for logtext in [stdout_log, caplog.text]:
        assert "foo" in logtext
        assert "A debug message" in logtext
        # Test that debug information is in output
        assert "test_logging" in logtext


def read_csv(path: str) -> List[List[str]]:
    """Utility to read CSV file content into a list of rows."""
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_csv_file_handler_emit_data(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    log_file = tmp_path / "log.csv"

    handler = CSVFileHandler(filename=log_file, encoding="utf-8")

    keys = ["Time", "Value"]
    values = ["12:00", "42"]
    units = ["s", "ångstrom"]

    # First write
    handler.emit_data(keys, values, units)

    rows = read_csv(log_file)
    assert rows == [keys, units, values]

    # Second write should only append values
    more_values = ["12:01", "43"]
    handler.emit_data(keys, more_values, units)

    rows = read_csv(log_file)
    assert rows == [keys, units, values, more_values]

    handler.close()


def test_csv_file_handler_emit_does_nothing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    log_file = Path("log.csv")

    handler = CSVFileHandler(log_file)

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Test",
        args=(),
        exc_info=None,
    )

    handler.emit(record)

    assert not log_file.exists() or log_file.stat().st_size == 0

    handler.close()


def test_custom_logger_logs_to_csv_handler(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    log_file = "structured_log.csv"

    logger = CustomLogger("test_logger")
    handler = CSVFileHandler(log_file)
    logger.addHandler(handler)

    keys = ["Epoch", "Energy"]
    values = ["1", "-10.5"]
    units = ["", "kcal/mol"]

    logger.data(keys, values, units)
    logger.data(keys, values, units)

    rows = read_csv(log_file)
    assert rows == [keys, units, values, values]

    handler.close()


def test_wandb_handler_emit_data(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    run = wandb.init(mode="offline")
    handler = WandbHandler(run=run)

    keys = ["Epoch", "loss", "Value"]
    values = ["10", "0.3", "42"]
    units = ["", "", "units"]

    # First write
    handler.emit_data(keys, values, units)
    handler.close()


def test_wandb_handler_handler_emit_does_nothing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    run = wandb.init(mode="offline")
    handler = WandbHandler(run=run)

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Test",
        args=(),
        exc_info=None,
    )

    handler.emit(record)
    handler.close()


class MockWandbRun:
    """Mock class for wandb.Run to simulate logging behavior."""

    def __init__(self, log_file):
        self.log_file = log_file
        self.logs = []

    def log(self, data, step=None, commit=True):
        entry = {"step": step, "commit": commit, "data": data}
        self.logs.append(entry)
        # Also write to file for inspection
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def finish(self):
        pass  # for compatibility


@pytest.mark.parametrize("prefix", ["training", "test", "validation"])
def test_custom_logger_logs_to_wandb(monkeypatch, tmp_path, prefix):
    monkeypatch.chdir(tmp_path)

    logger = CustomLogger("test_logger")

    log_file = tmp_path / "wandb_log.jsonl"
    mock_run = MockWandbRun(log_file)
    handler = WandbHandler(run=mock_run)

    logger.addHandler(handler)

    keys = ["Epoch", f"{prefix} energy"]
    values = ["1", "-10.5"]
    units = ["", "kcal/mol"]

    logger.data(keys, values, units)
    logger.data(keys, values, units)

    for handler in logger.handlers:
        handler.close()

    # Read logged entries
    with open(log_file) as f:
        lines = f.readlines()

    assert len(lines) == 2
    for line in lines:
        entry = json.loads(line)
        assert entry["step"] == 1
        assert entry["commit"] is True

        # Check cleaned key format
        expected_key = f"{prefix}/energy [kcal per mol]"
        assert expected_key in entry["data"]
        assert entry["data"][expected_key] == -10.5


@pytest.mark.parametrize("handler_cls", [WandbHandler, CSVFileHandler])
def test_handler_different_lengths(handler_cls, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    if handler_cls is CSVFileHandler:
        log_file = tmp_path / "log.csv"
        handler = CSVFileHandler(log_file)
    elif handler_cls is WandbHandler:
        run = wandb.init(mode="offline")
        handler = WandbHandler(run=run)
    else:
        raise ValueError("Unknown handler class")

    keys = ["Epoch", "loss", "Value"]
    values = ["10", "0.3", "42"]
    units = ["", ""]  # Different lengths

    match = (
        f"keys, values and units must have the same length: "
        f"{len(keys)}, {len(values)}, {len(units)}"
    )
    with pytest.raises(ValueError, match=match):
        handler.emit_data(keys, values, units)

    handler.close()


def test_custom_logger_ignores_handlers_without_emit_data(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    class DummyHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.called = False

        def emit(self, record):
            self.called = True

    dummy = DummyHandler()
    logger = CustomLogger("test_logger_dummy")
    logger.addHandler(dummy)

    logger.data(["A"], ["1"], ["unit"])
    assert not dummy.called


def test_default_logger():
    """Test that the default logger is set up correctly."""
    logger = logging.getLogger(name="metatrain")

    assert type(logger) is CustomLogger
    assert logger.name == "metatrain"


def test_metric_logger(caplog, monkeypatch, tmp_path):
    """Tests the MetricLogger class."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO)
    logger = logging.getLogger(__name__)

    assert type(logger) is CustomLogger

    outputs = {
        "energy": ModelOutput(unit="eV", explicit_gradients=["positions"]),
        "mtt::foo": ModelOutput(unit="eV"),
        "mtt::bar": ModelOutput(unit="hartree"),
    }
    capabilities = ModelCapabilities(
        length_unit="angstrom",
        atomic_types=[1, 2, 3],
        outputs=outputs,
    )

    names = ["train"]
    train_metrics = [
        {
            "loss": 0.1,
            "baz": 1e-5,
            "energy RMSE": 1.0,
            "energy_positions_gradients MAE": 0.5,
            "mtt::foo RMSE": 1.0,
            "mtt::bar RMSE": 0.1,
        }
    ]

    # single dict to test that metrics will be converted to list
    eval_metrics = {"mtt::foo RMSE": 5.0, "mtt::bar RMSE": 10.0}

    with setup_logging(logger, log_file="logfile.log", level=logging.INFO):
        trainer_logger = MetricLogger(
            log_obj=logger,
            dataset_info=capabilities,
            initial_metrics=train_metrics,
            names=names,
        )
        trainer_logger.log(metrics=train_metrics, epoch=1)

        eval_logger = MetricLogger(
            log_obj=logger, dataset_info=capabilities, initial_metrics=eval_metrics
        )
        eval_logger.log(metrics=eval_metrics)

    # Test for correctly formatted log messages (only one space between words)
    # During training
    assert "Epoch:    1 | " in caplog.text
    assert "train loss: 1.000e-01 | " in caplog.text
    assert "train baz: 1.000e-05 | " in caplog.text
    assert "train energy RMSE: 1000.0 meV | " in caplog.text
    assert "train energy_positions_gradients MAE: 500.00 meV/A | " in caplog.text
    assert "train mtt::bar RMSE: 0.10000 hartree | " in caplog.text
    assert "train mtt::foo RMSE: 1000.0 meV" in caplog.text  # eV converted to meV

    # During evaluation
    assert "| mtt::foo RMSE: 5000.0 meV" in caplog.text

    # Test CSV file output
    rows = read_csv("logfile.csv")

    assert len(rows) == 3
    assert rows[0] == [
        "Epoch",
        "train loss",
        "train baz",
        "train energy RMSE",
        "train energy_positions_gradients MAE",
        "train mtt::bar RMSE",
        "train mtt::foo RMSE",
    ]
    assert rows[1] == ["", "", "", "meV", "meV/A", "hartree", "meV"]
    assert rows[2] == [
        "   1",
        "1.000e-01",
        "1.000e-05",
        "1000.0",
        "500.00",
        "0.10000",
        "1000.0",
    ]


def get_argv():
    argv = ["mypgroam", "option1", "-o", "optional", "--long", "extra options"]
    argv_str = 'mypgroam option1 -o optional --long "extra options"'
    return argv, argv_str


def test_get_cli_input():
    argv, argv_str = get_argv()
    assert get_cli_input(argv) == argv_str


def test_get_cli_input_sys(monkeypatch):
    argv, argv_str = get_argv()
    monkeypatch.setattr(sys, "argv", argv)
    assert get_cli_input() == argv_str


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, "0"),
        (123, "123"),
        (999, "999"),
        (1000, "1K"),
        (1049, "1K"),
        (1050, "1.1K"),
        (1234, "1.2K"),
        (9999, "10K"),
        (20454, "20.5K"),
        (99499, "99.5K"),
        (99500, "99.5K"),
        (100000, "100K"),
        # Edge case around 1 million
        (999499, "999K"),
        (999500, "1M"),
        (999999, "1M"),
        (1000000, "1M"),
        (1049999, "1M"),
        (1050000, "1.1M"),
        # Larger numbers
        (123456789, "123M"),
        (999999999999, "1T"),
        # Max suffix
        (1230000000000000, "1230T"),
        (1230000000000000000, "1230000T"),
    ],
)
def test_human_readable_parameter_counter(value, expected):
    assert human_readable(value) == expected
