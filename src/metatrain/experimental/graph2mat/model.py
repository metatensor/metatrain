import logging
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
from e3nn import o3
from graph2mat import MatrixDataProcessor
from graph2mat.bindings.e3nn import E3nnGraph2Mat
from metatensor.torch import Labels, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)

from metatrain.utils.abc import ModelInterface
from metatrain.utils.architectures import get_default_hypers, import_architecture
from metatrain.utils.data import DatasetInfo
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.metadata import merge_metadata

from .documentation import ModelHypers
from .modules.edge_embedding import BesselBasis
from .utils.basis import get_basis_table_from_yaml
from .utils.mtt import g2m_labels_to_tensormap, split_dataset_info
from .utils.structures import create_batch, get_edge_vectors_and_lengths


class MetaGraph2Mat(ModelInterface[ModelHypers]):
    """Interface of MACE for metatrain."""

    __checkpoint_version__ = 1
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]
    __default_metadata__ = ModelMetadata(
        references={
            "architecture": [
                "https://iopscience.iop.org/article/10.1088/2632-2153/adc871"
            ]
        }
    )

    def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)

        # -----------------------------------------------------------
        #   Split dataset info into targets that Graph2Mat handles
        #   and those that will be handled by the featurizer itself
        # -----------------------------------------------------------
        self.featurizer_dataset_info, self.graph2mat_dataset_info = split_dataset_info(
            dataset_info=dataset_info,
            node_hidden_irreps=self.hypers["node_hidden_irreps"],
        )

        # ---------------------------
        # Initialize the featurizer
        # ---------------------------
        # We use the "featurizer_architecture" hyper to initialize a model.

        featurizer_name = self.hypers["featurizer_architecture"]["name"]
        featurizer_arch = import_architecture(featurizer_name)
        default_hypers = get_default_hypers(featurizer_name)
        model_hypers = {
            **default_hypers["model"],
            **self.hypers["featurizer_architecture"].get("model", {}),
        }
        self.featurizer_model = featurizer_arch.__model__(
            hypers=model_hypers,
            dataset_info=self.featurizer_dataset_info,
        )

        # ----------------------------------------------------
        #      Prepare things for initializing Graph2Mat
        # ----------------------------------------------------

        # Get the basis, this will likely be a different basis table
        # per target in the end, let's see
        basis_table = get_basis_table_from_yaml(self.hypers["basis_yaml"])

        # Atomic types, and helper to convert from atomic type (Z) to index
        # in the basis table.
        self.atomic_types = [atom.Z for atom in basis_table.atoms]
        self.register_buffer(
            "atomic_types_to_species_index",
            torch.zeros(max(self.atomic_types) + 1, dtype=torch.int64),
        )
        for i, atomic_type in enumerate(self.atomic_types):
            self.atomic_types_to_species_index[atomic_type] = i

        # Functions to embed edges for graph2mat.
        # Radial embedding (i.e. embedding of the edge length).
        n_basis = 8
        self.radial_embedding = BesselBasis(
            r_max=np.max(basis_table.R), num_basis=n_basis
        )

        # Embedding of the direction of the edge.
        sh_irreps = o3.Irreps.spherical_harmonics(2)
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # Irreps for all the inputs that graph2mat will take.
        graph2mat_irreps = dict(
            # One hot encoding of species
            node_attrs_irreps=o3.Irreps("0e") * len(self.atomic_types),
            # Features coming from the featurizer
            node_feats_irreps=o3.Irreps(self.hypers["node_hidden_irreps"]),
            # Embedding of the edges direction.
            edge_attrs_irreps=sh_irreps,
            # Embedding of the edges length.
            edge_feats_irreps=o3.Irreps(f"{n_basis}x0e"),
            # Internal irreps for graph2mat
            edge_hidden_irreps=o3.Irreps(self.hypers["edge_hidden_irreps"]),
        )

        # ----------------------------------------------------
        #        Initialize one Graph2Mat per target
        # ----------------------------------------------------
        self.graph2mats = torch.nn.ModuleDict()
        self.graph2mat_nls: dict[str, NeighborListOptions] = {}
        self.graph2mat_processors: dict[str, MatrixDataProcessor] = {}
        for i, target_name in enumerate(self.graph2mat_dataset_info.targets):
            # Get the matrix processor for this target.
            data_processor = MatrixDataProcessor(
                basis_table=basis_table,
                symmetric_matrix=True,
                sub_point_matrix=False,
                out_matrix=target_name,
                node_attr_getters=[],
            )
            self.graph2mat_processors[target_name] = data_processor

            # Initialize graph2mat.
            self.graph2mats[target_name] = E3nnGraph2Mat(
                unique_basis=data_processor.basis_table.basis,
                irreps=graph2mat_irreps,
                symmetric=data_processor.symmetric_matrix,
                basis_grouping=self.hypers["basis_grouping"],
            )

            # The neighbor list options are ignored, since the neighbor lists
            # are created by graph2mat according to the basis.
            # Here we just make sure we have a unique neighbor list for
            # each graph2mat.
            self.graph2mat_nls[target_name] = NeighborListOptions(
                cutoff=80.999 + i * 0.03,
                full_list=True,
                strict=True,
                requestor=f"graph2mat_{target_name}",
            )

        # ---------------------------
        #    Outputs definition
        # ---------------------------

        all_targets = {
            **self.featurizer_dataset_info.targets,
            **self.graph2mat_dataset_info.targets,
        }

        self.outputs = {
            k: ModelOutput(
                quantity=target_info.quantity,
                unit=target_info.unit,
                per_atom=True,
            )
            for k, target_info in all_targets.items()
        }

        # ---------------------------
        # Data preprocessing modules
        # ---------------------------

        # For now we don't have additive contributions or scaling.
        # self.additive_models = torch.nn.ModuleList([])

        # self.scaler = Scaler(hypers={}, dataset_info=self.featurizer_dataset_info)

        # self.finetune_config: Dict[str, Any] = {}

    def restart(self, dataset_info: DatasetInfo) -> "MetaGraph2Mat":
        # Check that the new dataset info does not contain new atomic types
        if new_atomic_types := set(dataset_info.atomic_types) - set(
            self.dataset_info.atomic_types
        ):
            raise ValueError(
                f"New atomic types found in the dataset: {new_atomic_types}. "
                "The MACE model does not support adding new atomic types."
            )

        # Merge the old dataset info with the new one
        merged_info = self.dataset_info.union(dataset_info)

        # Check if there are new targets
        new_targets = {
            key: value
            for key, value in merged_info.targets.items()
            if key not in self.dataset_info.targets
        }
        self.has_new_targets = len(new_targets) > 0

        # Add extra heads for the new targets
        for target_name, target in new_targets.items():
            self._add_output(target_name, target)

        self.dataset_info = merged_info

        # restart the composition and scaler models
        # self.additive_models[0] = self.additive_models[0].restart(
        #     dataset_info=DatasetInfo(
        #         length_unit=dataset_info.length_unit,
        #         atomic_types=self.dataset_info.atomic_types,
        #         targets={
        #             target_name: target_info
        #             for target_name, target_info in dataset_info.targets.items()
        #             if CompositionModel.is_valid_target(target_name, target_info)
        #         },
        #     ),
        # )
        # self.scaler = self.scaler.restart(dataset_info)

        return self

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if selected_atoms is not None:
            raise NotImplementedError("selected_atoms not implemented yet")

        # -------------------------------------------------------
        #  Split outputs according to whether the featurizer or
        #  graph2mat will handle them
        # -------------------------------------------------------

        featurizer_outputs = {
            k: v
            for k, v in outputs.items()
            if k in self.featurizer_dataset_info.targets
        }
        graph2mat_outputs = {
            k: v for k, v in outputs.items() if k in self.graph2mat_dataset_info.targets
        }

        # -----------------------------
        #   Featurizer forward pass
        # -----------------------------
        # We add extra outputs to the featurizer to retrieve the node
        # features that graph2mat will use.

        featurizer_return = self.featurizer_model.forward(
            systems=systems,
            outputs={
                **featurizer_outputs,
                **{
                    f"mtt::aux::graph2mat_{target_name}": ModelOutput(
                        quantity="",
                        unit="",
                        per_atom=True,
                    )
                    for target_name in graph2mat_outputs
                },
            },
            selected_atoms=selected_atoms,
        )

        # ----------------------------------------------------------------
        #   Concatenate tensormap outputs to get flat tensors (e3nn-like)
        # ----------------------------------------------------------------

        graph2mat_inputs = {}
        # Concatenate outputs to get the e3nn representations from the tensormap
        for target_name in graph2mat_outputs:
            graph2mat_inputs[target_name] = []

            tensormap = featurizer_return.pop(f"mtt::aux::graph2mat_{target_name}")

            for block in tensormap.blocks():
                # Move components dimension to last and then flatten to get (n_atoms, irreps_dim)
                block_values = block.values.transpose(1, 2)
                graph2mat_inputs[target_name].append(
                    block_values.reshape(block_values.shape[0], -1)
                )

            graph2mat_inputs[target_name] = torch.cat(
                graph2mat_inputs[target_name], dim=-1
            )

        # -----------------------------
        #      Run each Graph2Mat
        # -----------------------------

        graph2mat_returns = {}

        for target_name, graph2mat in self.graph2mats.items():
            if target_name not in graph2mat_outputs:
                continue

            # Create the batch with the graph that this graph2mat will use
            data = create_batch(
                systems=systems,
                neighbor_list_options=self.graph2mat_nls[target_name],
                atomic_types_to_species_index=self.atomic_types_to_species_index,
                n_types=len(self.atomic_types),
                data_processor=self.graph2mat_processors[target_name],
            )

            # Convert coordinates from XYZ to YZX so that the outputs are spherical
            # harmonics.
            data["positions"] = data["positions"][:, [1, 2, 0]]
            data["cell"] = data["cell"][:, [1, 2, 0]]
            data["shifts"] = data["shifts"][:, [1, 2, 0]]

            # Embed edges and add them to the batch
            vectors, lengths = get_edge_vectors_and_lengths(
                positions=data["positions"],
                edge_index=data["edge_index"],
                shifts=data["shifts"],
            )
            edge_attrs = self.spherical_harmonics(vectors)
            edge_feats = self.radial_embedding(lengths)

            data["edge_attrs"] = edge_attrs
            data["edge_feats"] = edge_feats

            # Run graph2mat and store the outputs (a tuple of tensors: node labels and edge labels)
            graph2mat_returns[target_name] = graph2mat(
                data=data, node_feats=graph2mat_inputs[target_name]
            )

        # -----------------------------------
        #   Convert outputs to TensorMaps
        # -----------------------------------

        # At this point, we have a dictionary of all outputs as normal torch tensors.
        # Now, we simply convert to TensorMaps.

        # Get the labels for the samples (system and atom of each value)

        return_dict: Dict[str, TensorMap] = {
            **featurizer_return,
            **{
                output_name: g2m_labels_to_tensormap(
                    node_labels=graph2mat_returns[output_name][0],
                    edge_labels=graph2mat_returns[output_name][1],
                )
                for output_name in graph2mat_outputs
            },
        }

        return return_dict

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def requested_neighbor_lists(
        self,
    ) -> List[NeighborListOptions]:
        return [self.requested_nl]

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "MetaGraph2Mat":
        if context == "restart":
            logging.info(f"Using latest model from epoch {checkpoint['epoch']}")
            model_state_dict = checkpoint["model_state_dict"]
        elif context in {"finetune", "export"}:
            logging.info(f"Using best model from epoch {checkpoint['best_epoch']}")
            model_state_dict = checkpoint["best_model_state_dict"]
            if model_state_dict is None:
                model_state_dict = checkpoint["model_state_dict"]
        else:
            raise ValueError("Unknown context tag for checkpoint loading!")

        # Create the model
        model_data = checkpoint["model_data"]
        model = cls(**model_data)
        # Infer dtype
        dtype = None

        # Set up composition and scaler models
        # model.additive_models[0].sync_tensor_maps()
        # model.scaler.sync_tensor_maps()
        model.load_state_dict(model_state_dict)

        # Loading the metadata from the checkpoint
        model.metadata = merge_metadata(model.metadata, checkpoint.get("metadata"))

        return model

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        raise NotImplementedError("Export not implemented yet for MetaGraph2Mat")
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {dtype} for MACE")

        # Make sure the model is all in the same dtype
        # For example, after training, the additive models could still be in
        # float64
        self.to(dtype)

        # Additionally, the composition model contains some `TensorMap`s that cannot
        # be registered correctly with Pytorch. This function moves them:
        self.additive_models[0].weights_to(torch.device("cpu"), torch.float64)

        interaction_range = self.hypers["num_interactions"] * self.cutoff

        capabilities = ModelCapabilities(
            outputs=self.outputs,
            atomic_types=self.atomic_types,
            interaction_range=interaction_range,
            length_unit=self.dataset_info.length_unit,
            supported_devices=self.__supported_devices__,
            dtype=dtype_to_str(dtype),
        )

        metadata = merge_metadata(self.metadata, metadata)

        return AtomisticModel(jit.compile(self.eval()), metadata, capabilities)

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict) -> Dict:
        if checkpoint["model_ckpt_version"] != cls.__checkpoint_version__:
            raise RuntimeError(
                f"Unable to upgrade the checkpoint: the checkpoint is using model "
                f"version {checkpoint['model_ckpt_version']}, while the current model "
                f"version is {cls.__checkpoint_version__}."
            )

        return checkpoint

    def get_checkpoint(self) -> Dict:
        model_state_dict = self.state_dict()

        # If the MACE model was passed as part of the hypers, we store it
        # again as part of the hypers.
        hypers = self.hypers.copy()

        checkpoint = {
            "architecture_name": "experimental.graph2mat",
            "model_ckpt_version": self.__checkpoint_version__,
            "metadata": self.metadata,
            "model_data": {
                "hypers": hypers,
                "dataset_info": self.dataset_info.to(device="cpu"),
            },
            "epoch": None,
            "best_epoch": None,
            "model_state_dict": model_state_dict,
            "best_model_state_dict": model_state_dict,
        }
        return checkpoint

    def get_fixed_scaling_weights(self) -> dict:
        return {}
