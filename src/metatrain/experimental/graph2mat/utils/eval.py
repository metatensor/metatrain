"""Little helper script to evaluate the model from a ckpt file while
it is not torchscript compatible yet."""

import argparse

import graph2mat
import sisl
import torch
from graph2mat.bindings.torch import TorchBasisMatrixData, TorchBasisMatrixDataset
from metatomic.torch import ModelOutput

from metatrain.experimental.graph2mat import MetaGraph2Mat
from metatrain.experimental.graph2mat.utils.conversions import (
    get_target_converters,
    transform_tensormap_matrix,
)
from metatrain.experimental.graph2mat.utils.dataset import (
    get_graph2mat_eval_transform,
    system_to_config,
)
from metatrain.utils.data import CollateFn, Dataset, read_systems, unpack_batch
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists_transform,
)
from metatrain.utils.transfer import batch_to


# ----------------------------------------
#          Argument parsing
# ----------------------------------------

parser = argparse.ArgumentParser(description="Evaluate Graph2Mat model from checkpoint")
parser.add_argument(
    "input_file", type=str, help="Input file containing the systems (e.g., XYZ format)"
)
parser.add_argument("model_ckpt", type=str, help="Path to the model checkpoint file")
parser.add_argument(
    "--targets",
    nargs="+",
    default=["density_matrix"],
    help="List of target properties to evaluate",
)
args = parser.parse_args()

# ----------------------------------------
#  Reading input data and preparing model
# ----------------------------------------
systems = read_systems(
    filename=args.input_file,
    reader="ase",
)
targets = {target: ModelOutput() for target in args.targets}
dataset = Dataset.from_dict({"system": systems})

ckpt = torch.load(args.model_ckpt, map_location="cpu")
model = MetaGraph2Mat.load_checkpoint(ckpt, context="export")

requested_neighbor_lists = get_requested_neighbor_lists(model.featurizer_model)
collate_fn = CollateFn(
    list(targets),
    callables=[
        get_system_with_neighbor_lists_transform(requested_neighbor_lists),
        get_graph2mat_eval_transform(
            model.graph2mat_processors, model.graph2mat_nls, outputs=list(targets)
        ),
    ],
)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, collate_fn=collate_fn, shuffle=False
)

# ---------------------------------------------------------------
#   Helpers to convert from spherical harmonics to the basis used
#   in the target data.
# ---------------------------------------------------------------

converters = {}
for target_name in targets:
    converters[target_name] = get_target_converters(
        model.graph2mat_processors[target_name].basis_table,
        in_format="spherical",
        out_format=model.graph2mat_processors[target_name].basis_table.basis_convention,
    )


def spherical_to_basis(
    data: TorchBasisMatrixData,
    converters: dict,
    data_processor: graph2mat.MatrixDataProcessor,
):
    """The metatrain graph2mat model predicts the matrices in spherical harmonics basis.
    However, the target might be in a slightly different convention
    (e.g. Y-ZX instead of YZX).

    This function (inefficiently) converts the predicted data into the right basis convention.
    """
    dm = graph2mat.conversions.torch_basismatrixdata_to_sisl_DM(data)
    config = graph2mat.conversions.sisl_to_orbitalconfiguration(dm)
    tmap = graph2mat.conversions.basisconfiguration_to_tensormap(config)
    tmap = transform_tensormap_matrix(tmap, converters=converters)
    converted_bdict = graph2mat.conversions.tensormap_to_block_dict(
        tmap, lattice=sisl.Lattice(config.cell, nsc=config.matrix.nsc)
    )
    config.matrix.block_dict = converted_bdict
    data = graph2mat.conversions.orbitalconfiguration_to_basismatrixdata(
        config, data_processor
    )
    return data


# ----------------------------------------
#           Evaluation loop
# ----------------------------------------

for batch in dataloader:
    systems, batch_targets, batch_extra_data = unpack_batch(batch)
    systems, batch_targets, batch_extra_data = batch_to(
        systems, batch_targets, batch_extra_data, dtype=torch.float32, device="cpu"
    )

    out = model(systems, outputs=targets)

    for target in args.targets:
        dm_tensormap = out[target]

        configs = [
            system_to_config(system, model.graph2mat_processors[target], None)
            for system in systems
        ]

        dataset = TorchBasisMatrixDataset(
            configs,
            data_processor=model.graph2mat_processors[target],
            data_cls=graph2mat.bindings.torch.TorchBasisMatrixData,
            load_labels=False,
        )
        data = dataset[0]

        data["point_labels"] = dm_tensormap.block(0).values.ravel()
        data["edge_labels"] = dm_tensormap.block(1).values.ravel()

        data = spherical_to_basis(
            data,
            converters=converters[target],
            data_processor=model.graph2mat_processors[target],
        )

        if target == "density_matrix":
            dm = graph2mat.conversions.torch_basismatrixdata_to_sisl_DM(data)
            dm.write("prediction.DM")
        elif target == "hamiltonian":
            hamiltonian = graph2mat.conversions.torch_basismatrixdata_to_sisl_H(data)
            hamiltonian.write("prediction.TSHS")
        elif target == "overlap_matrix":
            overlap_matrix = graph2mat.conversions.torch_basismatrixdata_to_sisl_S(data)
            overlap_matrix.write("prediction.TSHS")
        else:
            print(f"Writing for target {target} not implemented.")
