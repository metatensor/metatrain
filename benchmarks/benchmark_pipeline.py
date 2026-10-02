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
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

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

    # a small validation set: it is not part of the report, and a full second
    # pass over the data per epoch would only add wall time
    val_dataset = torch.utils.data.Subset(dataset, range(min(len(dataset), 8)))

    with TemporaryDirectory() as checkpoint_dir:
        start = time.perf_counter()
        Trainer(hypers["training"]).train(
            model=model,
            dtype=torch.float32,
            devices=[torch.device(args.device)],
            train_datasets=[dataset],
            val_datasets=[val_dataset],
            checkpoint_dir=checkpoint_dir,
        )
        wall = time.perf_counter() - start

    print(
        f"\nPET, {len(dataset)} structures, batch_size={args.batch_size}, "
        f"num_workers={args.num_workers}, device={args.device}, "
        f"epochs={args.epochs}, {wall:.1f} s wall (incl. validation)\n"
    )
    print(timing.report())
    # after the report, so these batches are not part of it
    print("\n" + compare_transport(dataset, args.batch_size))


if __name__ == "__main__":
    main()
