import subprocess
import sys
from pathlib import Path

import pytest

from metatrain.utils import timing


BENCHMARK = Path(__file__).parents[2] / "benchmarks/benchmark_pipeline.py"
DATASET = Path(__file__).parents[1] / "resources/qm9_reduced_100.xyz"


@pytest.fixture
def enabled():
    """Enable timing for one test, then restore the default (off) state."""
    timing.reset()
    timing.ENABLED = True
    yield
    timing.ENABLED = False
    timing.reset()


def test_disabled_by_default():
    """Without the environment variable, nothing is recorded and no time is spent."""
    assert not timing.ENABLED

    with timing.timed("forward"):
        pass
    assert list(timing.timed_iter(range(3), "loader", "step")) == [0, 1, 2]

    assert timing.report() == "no timings recorded (set METATRAIN_TIMING=1 to enable)"


def test_records_stages(enabled):
    with timing.timed("forward"):
        pass
    with timing.timed("forward"):
        pass
    with timing.timed("backward"):
        pass

    report = timing.report()
    assert "forward" in report
    assert "backward" in report
    # two calls for `forward`, one for `backward`
    assert [line.split()[1] for line in report.splitlines() if "ward" in line] == [
        "2",
        "1",
    ]


def test_timed_iter_splits_wait_from_body(enabled):
    """The wait for each item and the loop body are recorded separately, and
    exhausting the iterator is not counted as a wait."""
    for _ in timing.timed_iter(range(4), "loader", "step"):
        with timing.timed("forward"):
            pass

    report = timing.report()
    assert "breakdown of `step`" in report
    for stage in ("loader", "step", "forward"):
        line = next(line for line in report.splitlines() if line.startswith(stage))
        assert line.split()[1] == "4"


def test_timed_transform_names_the_callable(enabled):
    """Each transform gets its own stage, named after the callable, so the
    aggregate `transforms` stage can be broken down."""

    def add_neighbor_lists(*args):
        return args

    class Augmenter:
        def __call__(self, *args):
            return args

    for transform in (add_neighbor_lists, Augmenter()):
        with timing.timed_transform(transform):
            pass

    report = timing.report()
    assert "transforms/add_neighbor_lists" in report
    assert "transforms/Augmenter" in report


def test_timed_transform_is_free_when_disabled():
    with timing.timed_transform(lambda *args: args):
        pass
    assert timing.report() == "no timings recorded (set METATRAIN_TIMING=1 to enable)"


def test_counters(enabled):
    class FakeSystem:
        def __len__(self):
            return 3

    timing.count_systems([FakeSystem(), FakeSystem()])
    with timing.timed("step"):
        pass

    report = timing.report()
    assert "structures/s" in report
    assert "atoms/s" in report


def test_reset(enabled):
    with timing.timed("forward"):
        pass
    timing.reset()
    assert timing.report() == "no timings recorded (set METATRAIN_TIMING=1 to enable)"


def test_benchmark_script_runs(tmp_path):
    """The benchmark script trains PET end-to-end and prints a report with the
    stages of a step."""
    result = subprocess.run(
        [
            sys.executable,
            str(BENCHMARK),
            "--dataset",
            str(DATASET),
            "--batch-size",
            "4",
            "--epochs",
            "1",
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    for expected in ("loader", "step", "forward", "backward", "serialize", "atoms/s"):
        assert expected in result.stdout
