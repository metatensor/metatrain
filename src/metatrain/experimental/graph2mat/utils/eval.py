"""Little helper script to evaluate the model from a ckpt file while
it is not torchscript compatible yet."""

from typing import Optional
from functools import partial

import graph2mat
import sisl
import torch
from graph2mat.bindings.torch import TorchBasisMatrixData, TorchBasisMatrixDataset
from metatomic.torch import ModelOutput, systems_to_torch
import ase

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

from metatrain.experimental.graph2mat.utils.dataset import get_graph2mat_transform
from metatomic.torch import systems_to_torch
from metatensor.torch import TensorMap, TensorBlock, Labels


# ----------------------------------------
#          Argument parsing
# ----------------------------------------

# parser = argparse.ArgumentParser(description="Evaluate Graph2Mat model from checkpoint")
# parser.add_argument(
#     "input_file", type=str, help="Input file containing the systems (e.g., XYZ format)"
# )
# parser.add_argument("model_ckpt", type=str, help="Path to the model checkpoint file")
# parser.add_argument(
#     "--targets",
#     nargs="+",
#     default=[],
#     help="List of target properties to evaluate",
# )
# args = parser.parse_args()

# ----------------------------------------
#  Reading input data and preparing model
# ----------------------------------------
# systems = read_systems(
#     filename=args.input_file,
#     reader="ase",
# )
# dataset = Dataset.from_dict({"system": systems})

class Graph2MatCalculator:
    def __init__(self, model_ckpt):
        ckpt = torch.load(model_ckpt, map_location="cpu")
        self.model = MetaGraph2Mat.load_checkpoint(ckpt, context="export").to(torch.float64)
        self.requested_neighbor_lists = get_requested_neighbor_lists(self.model.featurizer_model)

        self.transform = get_graph2mat_transform(
            self.model.graph2mat_processors,
            self.model.graph2mat_nls,
            self.model.hypers["matrices"],
            self.model.dataset_info.targets,
            add_neighbor_lists=False
        )

    def __call__(self, atoms: ase.Atoms, properties: Optional[list[str]] = None, out_format: Optional[str] = None) -> dict[str, torch.Tensor]:

        self.model = self.model.to(torch.float32)
        self.model.eval()

        if properties is None:
            properties = list(self.model.hypers["matrices"])

        targets = {}
        for target in properties:
            matrix_spec = self.model.hypers["matrices"][target]
            targets[matrix_spec["nodes"]] = ModelOutput()
            targets[matrix_spec["edges"]] = ModelOutput()

        collate_fn = CollateFn(
            list(targets),
            callables=[
                get_system_with_neighbor_lists_transform(self.requested_neighbor_lists),
            ],
        )

        if isinstance(atoms, ase.Atoms):
            systems = systems_to_torch([atoms])
        else:
            systems = [atoms]
        systems = [system.to(torch.float64) for system in systems] # so that they can be pickled
        dataset = Dataset.from_dict({"system": systems})
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, collate_fn=collate_fn, shuffle=False
        )
        
        batch = next(iter(dataloader))
        systems, batch_targets, batch_extra_data = unpack_batch(batch)
        systems, batch_targets, batch_extra_data = batch_to(
            systems, batch_targets, batch_extra_data, dtype=torch.float32, device="cpu"
        )

        out = self.model(systems, outputs=targets)

        return self._convert_output(systems, out, out_format=out_format)

    def _convert_output(self, systems, out, out_format: Optional[str]):
        detached_out = {}
        for k, tmap in out.items():
            new_blocks = []
            for block in tmap.blocks():
                b = TensorBlock(
                    values=block.values.detach(),
                    samples=block.samples,
                    components=block.components,
                    properties=block.properties,
                )
                new_blocks.append(b)
            detached_out[k] = TensorMap(keys=tmap.keys, blocks=new_blocks)

        index = 0
        tensor_map = TensorMap(
            keys=Labels(["_"], torch.tensor([[0]])),
            blocks=[
                TensorBlock(
                    # Integer values are not supported (coming soon)
                    values=torch.tensor([[index]]).to(torch.float64),
                    samples=Labels(
                        names=["system"],
                        values=torch.tensor([[index]]),
                    ),
                    components=[],
                    properties=Labels(["_"], torch.tensor([[0]])),
                )
            ],
        )
        _, out, _ = self.transform(systems, detached_out, {'mtt::aux::system_index': tensor_map})

        results = {}
        for matrix_name, matrix_spec in self.model.hypers["matrices"].items():
            node_target = matrix_spec["nodes"]
            edge_target = matrix_spec["edges"]

            configs = [
                system_to_config(system, self.model.graph2mat_processors[matrix_name], None)
                for system in systems
            ]

            data = TorchBasisMatrixDataset(
                configs,
                data_processor=self.model.graph2mat_processors[matrix_name],
                data_cls=TorchBasisMatrixData,
                load_labels=False,
            )[0]

            preds = {
                "node_labels": out[node_target].block().values.ravel(),
                "edge_labels": out[edge_target].block().values.ravel(),
            }

            if out_format is None:
                pred = data.copy()
                pred.point_labels = preds["node_labels"]
                pred.edge_labels = preds["edge_labels"]
            else:
                processor = self.model.graph2mat_processors[matrix_name]
                pred = processor.matrix_from_data(
                    data, preds, out_format=out_format
                )

            results[matrix_name] = pred

        return results
    
    def convert_sample(self, sample, out_format: Optional[str]):
        system = sample["system"]
        target_names = [field for field in sample._fields if field not in ["system", "mtt::aux::system_index"]]
        targets = {target_name: sample[target_name] for target_name in target_names}

        return self._convert_output([system], targets, out_format=out_format)

        

# ---------------------------------------------------------------
#   Helpers to convert from spherical harmonics to the basis used
#   in the target data.
# ---------------------------------------------------------------

# converters = {}
# for target_name in targets:
#     converters[target_name] = get_target_converters(
#         model.graph2mat_processors[target_name].basis_table,
#         in_format="spherical",
#         out_format=model.graph2mat_processors[target_name].basis_table.basis_convention,
#     )


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

# for batch in dataloader:
#     systems, batch_targets, batch_extra_data = unpack_batch(batch)
#     systems, batch_targets, batch_extra_data = batch_to(
#         systems, batch_targets, batch_extra_data, dtype=torch.float32, device="cpu"
#     )

#     out = model(systems, outputs=targets, raw_out=True)

#     for target in args.targets:
#         dm_tensormap = out[target]

#         configs = [
#             system_to_config(system, model.graph2mat_processors[target], None)
#             for system in systems
#         ]

#         dataset = TorchBasisMatrixDataset(
#             configs,
#             data_processor=model.graph2mat_processors[target],
#             data_cls=graph2mat.bindings.torch.TorchBasisMatrixData,
#             load_labels=False,
#         )
#         data = dataset[0]

#         data["point_labels"] = dm_tensormap.block(0).values.ravel()
#         data["edge_labels"] = dm_tensormap.block(1).values.ravel()

        # data = spherical_to_basis(
        #     data,
        #     converters=converters[target],
        #     data_processor=model.graph2mat_processors[target],
        # )

        # if target == "density_matrix":
        #     dm = graph2mat.conversions.torch_basismatrixdata_to_sisl_DM(data)
        #     dm.write("prediction.DM")
        # elif target == "hamiltonian":
        #     hamiltonian = graph2mat.conversions.torch_basismatrixdata_to_sisl_H(data)
        #     hamiltonian.write("prediction.TSHS")
        # elif target == "overlap_matrix":
        #     overlap_matrix = graph2mat.conversions.torch_basismatrixdata_to_sisl_S(data)
        #     overlap_matrix.write("prediction.TSHS")
        # else:
        #     print(f"Writing for target {target} not implemented.")
