from pathlib import Path

from metatrain.experimental.graph2mat.utils.dataset import get_graph2mat_transform, graph2mat_to_tensormap
from metatrain.experimental.graph2mat.model import MetaGraph2Mat
from metatrain.experimental.graph2mat.trainer import scale_targets
import torch
import pytest

from metatrain.utils.data import DiskDataset
from torch.utils.data import DataLoader
from metatrain.utils.data import CollateFn, unpack_batch
from metatrain.utils.transfer import batch_to

from metatrain.utils.scaler import get_remove_scale_transform
from metatrain.utils.additive import get_remove_additive_transform
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists_transform,
)
from metatrain.utils.architectures import get_default_hypers

from metatensor.torch import Labels

from metatrain.utils.data import (
    DatasetInfo,
    get_atomic_types,
    get_dataset,
)

from metatrain.experimental.graph2mat.model import MetaGraph2Mat
import yaml

@pytest.fixture(params=[True, False])
def symmetric(request):
    return request.param

@pytest.fixture(params=["point_type", "max"])
def basis_grouping(request):
    return request.param

@pytest.mark.parametrize("scaler_and_composition", [False, "train", "eval"])
def test_graph2mat_transform_roundtrip(scaler_and_composition, symmetric, basis_grouping):
    """Checks that graph2mat transforms are correct by doing a round trip
    and comparing the final targets with the unprocessed targets from the dataset.
    
    :param scaler_and_composition: If False, the test is done without any scaler
      or additive composition. If "train", the scaler is applied in the way it is
      done during training. If "eval", the scaler is applied in the way it is
      done during evaluation.
    """

    dtype = torch.float32
    device = torch.device("cpu")

    with open(Path(__file__).parent / "targets.yaml", "r") as f:
        targets = yaml.safe_load(f)

    dataset, targets_info, _ = get_dataset(
        {
            "systems": {
                "read_from": "/home/febrer/COSMO_disk/COSMO/tests/mtt_pair_targets/atom/spherical/scfbench_main_100.zip",
            },
            "targets": targets,
        }
    )

    dataset_info = DatasetInfo(
        length_unit="",
        atomic_types=get_atomic_types(dataset),
        targets=targets_info,
    )

    model_hypers = get_default_hypers("experimental.graph2mat")["model"]
    model_hypers["featurizer_architecture"] = {"name": "experimental.mace"}
    model_hypers["matrices"] = {
        "hamiltonian": {
            "nodes": "mtt::matrix_nodes::ham",
            "edges": "mtt::matrix_edges::ham",
            "symmetric": symmetric,
            "edge_cutoff": 8.0,
            "basis_grouping": basis_grouping,
        }
    }
    model = MetaGraph2Mat(
        hypers=model_hypers,
        dataset_info=dataset_info,
    )

    data = dataset[0]
    unprocessed_targets = {k: data[k] for k in data._fields if k in model.graph2mat_dataset_info.targets}

    requested_neighbor_lists = get_requested_neighbor_lists(model.featurizer_model)
    preprocessing_callables = []
    if scaler_and_composition:
        preprocessing_callables = [
            get_remove_additive_transform(model.additive_models, model.graph2mat_dataset_info.targets),
            get_remove_scale_transform(model.scaler),
        ]
    collate_fn = CollateFn(
        target_keys=list(model.graph2mat_dataset_info.targets.keys()),
        callables=[
            get_system_with_neighbor_lists_transform(requested_neighbor_lists),
            *preprocessing_callables,
            get_graph2mat_transform(
                model.graph2mat_processors, model.graph2mat_nls, model.hypers["matrices"], model.graph2mat_dataset_info.targets
            ),
        ],
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        sampler=None,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=1,
    )

    batch = next(iter(loader))
    systems, targets, extra_data = unpack_batch(batch)
    systems, targets, extra_data = batch_to(
        systems, targets, extra_data, dtype=dtype, device=device
    )

    # Do a forward pass just so that model.datas is stored
    model.to(dtype=dtype, device=device)
    model(systems, model.outputs)

    def _back_to_tensormap():
        for matrix_name in model.hypers["matrices"]:
            node_target = model.hypers["matrices"][matrix_name]["nodes"]
            edge_target = model.hypers["matrices"][matrix_name]["edges"]
            
            targets.update(graph2mat_to_tensormap(
                batch=model.datas[matrix_name],
                out=targets,
                processor=model.graph2mat_processors[matrix_name],
                node_labels_name=node_target,
                edge_labels_name=edge_target,
            ))

    if scaler_and_composition == "eval":
        _back_to_tensormap()
        targets = model.scaler(
            systems,
            targets,
            selected_atoms=None,
            use_per_target_scales=True,
            use_per_property_scales=False,
        )
    elif scaler_and_composition == "train":
        targets = scale_targets(
            model.scaler,
            systems,
            targets,
            extra_data,
            per_property=False,
        )
        _back_to_tensormap()
    else:
        _back_to_tensormap()

    if scaler_and_composition:
        model.add_additive_contributions(
            targets, systems, model.outputs, None
        )

    # At this point, targets should be equal to unprocessed targets
    for key, tmap in targets.items():
        unprocessed_tmap = unprocessed_targets[key]
        for block_key, block in tmap.items():
            unprocessed_block = unprocessed_tmap.block(block_key)
            samples = Labels(
                names=block.samples.names[1:],
                values=block.samples.values[:, 1:],
            )
            unproc_samples = Labels(
                names=unprocessed_block.samples.names[1:],
                values=unprocessed_block.samples.values[:, 1:],
            )

            sample_indices = unproc_samples.select(samples)
            assert block.properties == unprocessed_block.properties, f"Properties for {key} block {block_key} do not match"
            assert block.components == unprocessed_block.components, f"Components for {key} block {block_key} do not match"
            assert torch.allclose(block.values, unprocessed_block.values[sample_indices].to(dtype), atol=1e-5), f"Values for {key} block {block_key} do not match"


