import pytest
import torch
import torch.multiprocessing as mp

from metatrain.composition import CompositionModel, Trainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems
from metatrain.utils.data.target_info import get_energy_target_info

from . import DATASET_PATH
from .test_regression import _build_target_tensormaps, _make_synthetic_targets


WORLD_SIZE = 2

PER_SPECIES_ENERGIES = {1: -0.5, 6: -10.0, 7: -15.0, 8: -20.0}


def _make_datasets_and_info():
    systems = read_systems(DATASET_PATH)
    # An odd number of systems: shards of unequal sizes, which a padding
    # sampler (e.g. DistributedSampler) would equalize by duplicating samples,
    # biasing the fit. A second single-sample dataset leaves one rank with an
    # empty shard for its target.
    systems_a, systems_b = systems[:7], systems[7:8]
    values_a = _make_synthetic_targets(systems_a, PER_SPECIES_ENERGIES)
    # Perturb the targets so that they are not exactly linear in the
    # composition: for an exactly-consistent least-squares problem, duplicated
    # samples would not change the solution and the comparison with the serial
    # fit could not detect them.
    values_a += 0.1 * torch.sin(
        torch.arange(len(systems_a), dtype=torch.float64)
    ).reshape(-1, 1)
    dataset_a = Dataset.from_dict(
        {
            "system": systems_a,
            "energy_a": _build_target_tensormaps(systems_a, values_a),
        }
    )
    dataset_b = Dataset.from_dict(
        {
            "system": systems_b,
            "energy_b": _build_target_tensormaps(
                systems_b, _make_synthetic_targets(systems_b, PER_SPECIES_ENERGIES)
            ),
        }
    )
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy_a": get_energy_target_info("energy_a", {"unit": "eV"}),
            "energy_b": get_energy_target_info("energy_b", {"unit": "eV"}),
        },
    )
    return [dataset_a, dataset_b], dataset_info


def _fit(is_distributed):
    datasets, dataset_info = _make_datasets_and_info()
    model = CompositionModel(hypers={}, dataset_info=dataset_info)
    trainer = Trainer(hypers={"distributed": is_distributed, "batch_size": 1})
    trainer.train(
        model=model,
        dtype=torch.float64,
        devices=[torch.device("cpu")],
        train_datasets=datasets,
        val_datasets=datasets,
        checkpoint_dir="",
    )
    return {
        target_name: model.model.weights[target_name].block().values
        for target_name in ["energy_a", "energy_b"]
    }


def _run_rank(rank, init_file, result_file):
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        world_size=WORLD_SIZE,
        rank=rank,
    )
    try:
        weights = _fit(is_distributed=True)
        if rank == 0:
            torch.save(weights, result_file)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="requires torch.distributed"
)
def test_distributed_fit_matches_serial(tmp_path):
    """The distributed fit must give the same weights as the serial one: the
    accumulated least-squares problem is identical, only sharded, so any
    discrepancy (e.g. duplicated samples from a padding sampler) is a bug."""
    init_file = tmp_path / "init"
    result_file = tmp_path / "weights.pt"
    mp.spawn(
        _run_rank,
        args=(str(init_file), str(result_file)),
        nprocs=WORLD_SIZE,
        join=True,
    )
    distributed_weights = torch.load(result_file, weights_only=True)

    serial_weights = _fit(is_distributed=False)

    for target_name, serial in serial_weights.items():
        torch.testing.assert_close(distributed_weights[target_name], serial)
