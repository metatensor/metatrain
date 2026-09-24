"""Opt-in, per-stage timing of the training step.

Timing is off unless the ``METATRAIN_TIMING`` environment variable is set (or
:py:func:`enable` is called), in which case every :py:func:`timed` block
accumulates into a process-global table that :py:func:`report` formats. It
exists to answer one question before the data pipeline is optimized: which
fraction of a training step is spent waiting for the input pipeline rather
than running the model.

:py:func:`timed_iter` splits the training loop into the two top-level stages
``loader`` (waiting for the next batch) and ``step`` (everything the loop body
does with it); the stages in :py:data:`STEP_BREAKDOWN` break ``step`` down
further, and are therefore reported as a fraction of ``loader + step``.

The :py:data:`COLLATE_STAGES` are recorded inside ``DataLoader`` workers,
whose accumulators die with the worker process; they are only reported when
running with ``num_workers=0``. With workers, that time shows up as ``loader``
wait in the main process instead.
"""

import os
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Any, ContextManager, Dict, Iterable, Iterator, List

import torch
from metatomic.torch import System


ENABLED = bool(os.environ.get("METATRAIN_TIMING"))

TOP_STAGES = ("loader", "step")
"""The two halves of a training step: waiting for data, and using it."""

STEP_BREAKDOWN = ("unpack", "h2d", "forward", "loss", "backward", "optimizer")
"""Stages the ``step`` stage is made of, in the order they run."""

COLLATE_STAGES = ("group_and_join", "transforms", "serialize")
"""Stages inside the collate function, i.e. inside the ``loader`` wait."""

_totals: Dict[str, float] = defaultdict(float)
_calls: Dict[str, int] = defaultdict(int)
_counters: Dict[str, int] = defaultdict(int)
_EXHAUSTED = object()


def enable() -> None:
    """Turn timing on for this process, as ``METATRAIN_TIMING`` does.

    Useful for scripts that want timings without having to set the environment
    variable before importing metatrain.
    """
    global ENABLED
    ENABLED = True


def reset() -> None:
    """Forget all recorded timings and counters."""
    _totals.clear()
    _calls.clear()
    _counters.clear()


def _sync() -> None:
    """Wait for pending CUDA work, so it is not attributed to a later stage.

    Only syncs if this process has already initialized CUDA: the collate
    stages run in forked ``DataLoader`` workers, where initializing CUDA
    raises, and a CPU-only run never needs to sync at all.
    """
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


def _record(stage: str, start: float) -> None:
    """Accumulate the time elapsed since ``start`` into ``stage``.

    :param stage: Name of the stage to accumulate into.
    :param start: ``time.perf_counter()`` value from when the stage started.
    """
    _sync()
    _totals[stage] += time.perf_counter() - start
    _calls[stage] += 1


@contextmanager
def _timed(stage: str) -> Iterator[None]:
    """Time the enclosed block.

    :param stage: Name of the stage to accumulate into.
    :return: A context manager around the timed block.
    """
    _sync()
    start = time.perf_counter()
    try:
        yield
    finally:
        _record(stage, start)


def timed(stage: str) -> ContextManager[None]:
    """Time the enclosed block, or do nothing at all when timing is off.

    :param stage: Name of the stage to accumulate into.
    :return: A context manager around the timed block.
    """
    return _timed(stage) if ENABLED else nullcontext()


def _timed_iter(iterable: Iterable[Any], stage: str, body: str) -> Iterator[Any]:
    """Iterate ``iterable``, timing both the waits and the loop body.

    :param iterable: The iterable to consume.
    :param stage: Stage name for the wait before each item.
    :param body: Stage name for what the consumer does with each item.
    :yield: The items of ``iterable``.
    :ytype: Any
    """
    iterator = iter(iterable)
    while True:
        _sync()
        start = time.perf_counter()
        item = next(iterator, _EXHAUSTED)
        if item is _EXHAUSTED:  # exhaustion is not a wait for an item
            break
        _record(stage, start)

        body_start = time.perf_counter()
        try:
            yield item
        finally:
            _record(body, body_start)


def timed_iter(iterable: Iterable[Any], stage: str, body: str) -> Iterator[Any]:
    """Iterate ``iterable``, attributing waits to ``stage`` and the loop body
    to ``body``.

    :param iterable: The iterable to consume, e.g. a ``DataLoader``.
    :param stage: Stage name for the wait before each item.
    :param body: Stage name for what the consumer does with each item.
    :return: An iterator over ``iterable``.
    """
    return _timed_iter(iterable, stage, body) if ENABLED else iter(iterable)


def count_systems(systems: List[System]) -> None:
    """Accumulate the throughput counters for one batch.

    :param systems: The systems in the batch.
    """
    if ENABLED:
        _counters["structures"] += len(systems)
        _counters["atoms"] += sum(len(system) for system in systems)


def _rows(stages: Iterable[str], total: float) -> List[str]:
    """Format one table row per recorded stage.

    :param stages: Stage names to include; those never recorded are skipped.
    :param total: Time each stage's percentage share is computed against.
    :return: The formatted rows.
    """
    rows = []
    for stage in stages:
        if stage not in _totals:
            continue
        ms = 1e3 * _totals[stage] / _calls[stage]
        share = 100 * _totals[stage] / total if total else 0.0
        rows.append(f"{stage:<16}{_calls[stage]:>8}{ms:>10.2f}{share:>10.1f}%")
    return rows


def report() -> str:
    """Format the recorded timings as a table.

    :return: The formatted report, or a hint when nothing was recorded.
    """
    if not _totals:
        return "no timings recorded (set METATRAIN_TIMING=1 to enable)"

    total = sum(_totals[stage] for stage in TOP_STAGES if stage in _totals)
    lines = [f"{'stage':<16}{'calls':>8}{'ms/call':>10}{'% of step':>11}"]
    lines += _rows(TOP_STAGES, total)

    breakdown = _rows(STEP_BREAKDOWN, total)
    if breakdown:
        lines += ["", "breakdown of `step`:"] + breakdown

    collate = _rows(COLLATE_STAGES, total)
    if collate:
        lines += [
            "",
            "inside the `loader` wait (only recorded with num_workers=0):",
        ] + collate

    known = set(TOP_STAGES) | set(STEP_BREAKDOWN) | set(COLLATE_STAGES)
    other = _rows(sorted(set(_totals) - known), total)
    if other:
        lines += ["", "other:"] + other

    if total and _counters:
        lines.append("")
        lines += [
            f"{name + '/s':<16}{value / total:>18.1f}"
            for name, value in sorted(_counters.items())
        ]

    return "\n".join(lines)
