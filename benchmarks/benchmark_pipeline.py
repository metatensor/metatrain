"""Benchmark a real PET training step, stage by stage.

Runs ``Trainer.train`` on a small dataset with :py:mod:`metatrain.utils.timing`
enabled and prints where the time goes: waiting for the input pipeline
(``loader``) versus using the batch (``step``, broken down into unpack, host to
device transfer, forward, loss, backward and optimizer).

The collate stages (``group_and_join``, ``transforms``, ``serialize``) are
recorded in whichever process runs the collate function, so they are only
reported with ``--num-workers 0``. With workers they are part of the
``loader`` wait instead, which is exactly the number to compare against.
Their call count is a few higher than the number of steps, because the
(untimed) validation loop collates through the same code.

Examples::

    python benchmarks/benchmark_pipeline.py --num-workers 0
    python benchmarks/benchmark_pipeline.py --num-workers 4 --batch-size 16
"""

import argparse
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterator, Tuple

import torch
from omegaconf import OmegaConf

from metatrain.pet import PET, Trainer
from metatrain.utils import timing
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import (
    CollateFn,
    Dataset,
    DatasetInfo,
    collate_batch,
    get_atomic_types,
    unpack_batch,
)
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification


try:
    import psutil
except ModuleNotFoundError:  # measuring memory is optional
    psutil = None  # type: ignore[assignment]

DEFAULT_DATASET = (
    Path(__file__).parents[1] / "tests/resources/qm9_reduced_100.xyz"
).as_posix()


def build_dataset(path: str, key: str) -> Tuple[Dataset, Dict[str, Any]]:
    """Read an ASE-readable file into a single-energy-target dataset.

    :param path: Path to the structure file.
    :param key: Name of the per-structure energy key in that file.
    :return: The dataset and its target information.
    """
    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": path,
            "reader": "ase",
            "key": key,
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, target_info = read_targets(OmegaConf.create(conf))
    systems = read_systems(path)
    dataset = Dataset.from_dict({"system": systems, "energy": targets["energy"]})
    return dataset, target_info


def _memory_of(process: "psutil.Process") -> int:
    """Memory used by one process, counting shared pages only once.

    :param process: The process to measure.
    :return: Its proportional set size where available, otherwise its resident
        set size, in bytes.
    """
    try:
        return int(process.memory_full_info().pss)
    except (AttributeError, psutil.Error):
        # no PSS outside Linux: RSS instead, which counts the pages the workers
        # share with the parent once per process
        return int(process.memory_info().rss)


def _tree_memory() -> int:
    """Memory used by this process and its (worker) children.

    :return: The total, in bytes.
    """
    process = psutil.Process()
    total = _memory_of(process)
    for child in process.children(recursive=True):
        try:
            total += _memory_of(child)
        except psutil.Error:  # the child exited while we were looking
            pass
    return total


@contextmanager
def monitor_memory(interval: float = 0.1) -> Iterator[Dict[str, float]]:
    """Sample the memory use of the process tree while the block runs.

    Persistent workers hold their process state for the lifetime of the loader,
    so what matters is how much the run adds on top of what the parent process
    already holds.

    :param interval: Seconds between samples.
    :yield: The stats, filled in once the block is done; empty without psutil.
    :ytype: Dict[str, float]
    """
    stats: Dict[str, float] = {}
    if psutil is None:
        yield stats
        return

    baseline = peak = _tree_memory()
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak
        while True:  # sample before the first wait, so short runs get a value
            peak = max(peak, _tree_memory())
            if stop.wait(interval):
                break

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield stats
    finally:
        stop.set()
        thread.join()
        stats["baseline_gb"] = baseline / 1e9
        stats["peak_gb"] = peak / 1e9
        stats["added_gb"] = (peak - baseline) / 1e9


def compare_transport(dataset: Dataset, batch_size: int, repeats: int = 20) -> str:
    """Time the serialized transport against handing over a batch directly.

    Both paths do the same collation; the difference is the blob round trip
    that gets a batch across the worker boundary.

    :param dataset: The dataset to take samples from.
    :param batch_size: Number of samples per batch.
    :param repeats: How many times to collate the same batch.
    :return: The formatted comparison.
    """
    samples = [dataset[i] for i in range(min(batch_size, len(dataset)))]
    collate_fn = CollateFn(target_keys=["energy"])
    paths = {
        "CollateFn + unpack_batch": lambda: unpack_batch(collate_fn(samples)),
        "collate_batch (no blob)": lambda: unpack_batch(
            collate_batch(samples, {"energy"})
        ),
    }

    lines = [f"transport of a {len(samples)}-structure batch, no transformations:"]
    for name, path in paths.items():
        start = time.perf_counter()
        for _ in range(repeats):
            path()
        lines.append(
            f"{name:<28}{1e3 * (time.perf_counter() - start) / repeats:>8.2f} ms"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments.

    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--key", default="U0", help="energy key in the dataset")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    """Train PET for a few epochs and print the per-stage timing report."""
    args = parse_args()
    timing.enable()

    dataset, target_info = build_dataset(args.dataset, args.key)
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=get_atomic_types(dataset),
        targets=target_info,
    )

    # the per-target loss config is normally expanded from the `loss: mse`
    # shorthand by the yaml layer, which we bypass here
    loss = OmegaConf.create({"energy": init_with_defaults(LossSpecification)})
    OmegaConf.resolve(loss)

    hypers = get_default_hypers("pet", base_precision=32)
    hypers["training"].update(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        loss=loss,
    )
    model = PET(hypers["model"], dataset_info)

    # a disjoint split: the validation loop is not part of the timing report,
    # but it is part of an epoch, and it gets its own loader and workers
    val_size = max(1, round(args.val_fraction * len(dataset)))
    split = len(dataset) - val_size
    train_dataset = torch.utils.data.Subset(dataset, range(split))
    val_dataset = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    with TemporaryDirectory() as checkpoint_dir, monitor_memory() as stats:
        start = time.perf_counter()
        Trainer(hypers["training"]).train(
            model=model,
            dtype=torch.float32,
            devices=[torch.device(args.device)],
            train_datasets=[train_dataset],
            val_datasets=[val_dataset],
            checkpoint_dir=checkpoint_dir,
        )
        wall = time.perf_counter() - start

    memory = (
        f"memory {stats['peak_gb']:.2f} GB peak, {stats['added_gb']:.2f} GB added "
        f"over a {stats['baseline_gb']:.2f} GB baseline"
        if stats
        else "memory not measured (needs psutil)"
    )
    print(
        f"\nPET, {len(train_dataset)} train + {len(val_dataset)} validation "
        f"structures, batch_size={args.batch_size}, "
        f"num_workers={args.num_workers}, device={args.device}, "
        f"epochs={args.epochs}, {wall:.1f} s wall (incl. validation), "
        f"{memory}\n"
    )
    print(timing.report())
    # after the report, so these batches are not part of it
    print("\n" + compare_transport(dataset, args.batch_size))


if __name__ == "__main__":
    main()
