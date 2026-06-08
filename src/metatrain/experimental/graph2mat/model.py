import logging
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
from e3nn import o3
from graph2mat import MatrixDataProcessor
from graph2mat.bindings.e3nn import E3nnGraph2Mat, E3nnSimpleNodeBlock, E3nnSimpleEdgeBlock, E3nnEdgeMessageBlock
from metatensor.torch.operations._add import _add_block_block
from metatensor.torch import Labels, TensorMap, TensorBlock
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)

from metatrain.utils.abc import ModelInterface
from metatrain.utils.additive import CompositionModel
from metatrain.utils.scaler import Scaler
from metatrain.utils.architectures import get_default_hypers, import_architecture
from metatrain.utils.data import DatasetInfo
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.metadata import merge_metadata

from .documentation import ModelHypers
from .modules.edge_embedding import RadialEmbeddingBlock
from .modules.operations import OPERATIONS_REGISTRY
from .utils.basis import get_basis_from_layout
from .utils.mtt import g2m_labels_to_tensormap, split_dataset_info
from .utils.structures import create_batch, get_edge_vectors_and_lengths
from .utils.dataset import graph2mat_to_tensormap, add_neighbor_lists


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
            matrices=self.hypers["matrices"],
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

        # Atomic types, and helper to convert from atomic type (Z) to index
        # in the basis table.
        #self.atomic_types = [atom.Z for atom in basis_table.atoms]
        self.atomic_types = dataset_info.atomic_types
        self.register_buffer(
            "atomic_types_to_species_index",
            torch.zeros(max(self.atomic_types) + 1, dtype=torch.int64),
        )
        for i, atomic_type in enumerate(self.atomic_types):
            self.atomic_types_to_species_index[atomic_type] = i

        # Embedding of the direction of the edge.
        sh_irreps = o3.Irreps.spherical_harmonics(self.hypers["edge_max_ell"])
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # ----------------------------------------------------
        #        Initialize one Graph2Mat per target
        # ----------------------------------------------------
        self.graph2mats = torch.nn.ModuleDict()
        self.graph2mat_nls: dict[str, NeighborListOptions] = {}
        self.graph2mat_processors: dict[str, MatrixDataProcessor] = {}
        self.radial_embeddings = torch.nn.ModuleDict()
        for matrix_name, matrix_spec in self.hypers["matrices"].items():
            node_target = matrix_spec["nodes"]
            edge_target = matrix_spec["edges"]

            # Get the basis for this matrix
            basis_table = get_basis_from_layout(
                layout=self.graph2mat_dataset_info.targets[node_target].layout,
                R=matrix_spec.get("edge_cutoff") / 2
            )

            # Functions to embed edges for graph2mat.
            # Radial embedding (i.e. embedding of the edge length).
            n_basis = 10
            self.radial_embeddings[matrix_name] = RadialEmbeddingBlock(
                r_max=np.max(basis_table.R) * 2,
                num_bessel=n_basis, 
                num_polynomial_cutoff=10
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

            data_processor = MatrixDataProcessor(
                basis_table=basis_table,
                symmetric_matrix=matrix_spec.get("symmetric", False),
                sub_point_matrix=False,
                out_matrix=matrix_name,
                node_attr_getters=[],
            )

            self.graph2mat_processors[matrix_name] = data_processor

            node_operation = OPERATIONS_REGISTRY["node_operation"].get(
                matrix_spec.get("node_operation", "tsq"),
                E3nnSimpleNodeBlock
            )
            edge_operation = OPERATIONS_REGISTRY["edge_operation"].get(
                matrix_spec.get("edge_operation", "none"),
                E3nnSimpleEdgeBlock
            )
            preprocessing_edges = OPERATIONS_REGISTRY["preprocessing_edges"].get(
                matrix_spec.get("preprocessing_edges", "none"),
                E3nnEdgeMessageBlock
            )
            preprocessing_nodes = OPERATIONS_REGISTRY["preprocessing_nodes"].get(
                matrix_spec.get("preprocessing_nodes", "none"),
                None
            )

            # Initialize graph2mat.
            self.graph2mats[matrix_name] = E3nnGraph2Mat(
                unique_basis=data_processor.basis_table.basis,
                irreps=graph2mat_irreps,
                symmetric=data_processor.symmetric_matrix,
                basis_grouping=matrix_spec.get("basis_grouping", "point_type"),
                preprocessing_nodes=preprocessing_nodes,
                preprocessing_edges=preprocessing_edges,
                node_operation=node_operation,
                edge_operation=edge_operation,
                self_blocks_symmetry=matrix_spec.get("self_blocks_symmetry"),
            )

            # The neighbor list options are ignored, since the neighbor lists
            # are created by graph2mat according to the basis.
            # Here we just make sure we have a unique neighbor list for
            # each graph2mat.
            self.graph2mat_nls[matrix_name] = NeighborListOptions(
                cutoff=80.999 + i * 0.03,
                full_list=True,
                strict=True,
                requestor=f"graph2mat_{matrix_name}",
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
        # The composition model and scaler are handled by the trainer during training.
        # Their purpose is to adapt the data for optimal training.
        # At evaluation time, the model applies them on forward.
        composition_model = CompositionModel(
            hypers={},
            dataset_info=DatasetInfo(
                length_unit=dataset_info.length_unit,
                atomic_types=self.atomic_types,
                targets={
                    target_name: target_info
                    for target_name, target_info in dataset_info.targets.items()
                    if CompositionModel.is_valid_target(target_name, target_info)
                },
            ),
        )

        self.additive_models = torch.nn.ModuleList([composition_model])

        self.scaler = Scaler(hypers={}, dataset_info=self.dataset_info)

        self.finetune_config: Dict[str, Any] = {}

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
        self.additive_models[0] = self.additive_models[0].restart(
            dataset_info=DatasetInfo(
                length_unit=dataset_info.length_unit,
                atomic_types=self.dataset_info.atomic_types,
                targets={
                    target_name: target_info
                    for target_name, target_info in dataset_info.targets.items()
                    if CompositionModel.is_valid_target(target_name, target_info)
                },
            ),
        )
        self.scaler = self.scaler.restart(dataset_info)

        return self

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if selected_atoms is not None:
            raise NotImplementedError("selected_atoms not implemented yet")
        
        if not self.training:
            systems = add_neighbor_lists(systems, self.graph2mat_processors, self.graph2mat_nls)

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

        for matrix_name in self.graph2mats.keys():
            node_taget = self.hypers["matrices"][matrix_name]["nodes"]
            edge_target = self.hypers["matrices"][matrix_name]["edges"]
            if node_taget not in graph2mat_outputs and edge_target not in graph2mat_outputs:
                continue
            featurizer_outputs[f"mtt::aux::graph2mat_{matrix_name}"] = ModelOutput(
                quantity="",
                unit="",
                sample_kind="atom",
            )

        # -----------------------------
        #   Featurizer forward pass
        # -----------------------------
        # We add extra outputs to the featurizer to retrieve the node
        # features that graph2mat will use.

        featurizer_return = self.featurizer_model.forward(
            systems=systems,
            outputs=featurizer_outputs,
            selected_atoms=selected_atoms,
        )

        # ----------------------------------------------------------------
        #   Concatenate tensormap outputs to get flat tensors (e3nn-like)
        # ----------------------------------------------------------------

        graph2mat_inputs = {}
        # Concatenate outputs to get the e3nn representations from the tensormap
        for name in featurizer_outputs.keys():
            if not name.startswith("mtt::aux::graph2mat_"):
                continue
            matrix_name = name[len("mtt::aux::graph2mat_") :]
            graph2mat_inputs[matrix_name] = []

            tensormap = featurizer_return.pop(f"mtt::aux::graph2mat_{matrix_name}")

            for block in tensormap.blocks():
                # Move components dimension to last and then flatten to get (n_atoms, irreps_dim)
                block_values = block.values.transpose(1, 2)
                graph2mat_inputs[matrix_name].append(
                    block_values.reshape(block_values.shape[0], -1)
                )

            graph2mat_inputs[matrix_name] = torch.cat(
                graph2mat_inputs[matrix_name], dim=-1
            )

        # -----------------------------
        #      Run each Graph2Mat
        # -----------------------------

        graph2mat_returns = {}
        datas = {}

        for matrix_name, graph2mat in self.graph2mats.items():
            node_taget = self.hypers["matrices"][matrix_name]["nodes"]
            edge_target = self.hypers["matrices"][matrix_name]["edges"]
            if node_taget not in graph2mat_outputs and edge_target not in graph2mat_outputs:
                continue

            # Create the batch with the graph that this graph2mat will use
            data = create_batch(
                systems=systems,
                neighbor_list_options=self.graph2mat_nls[matrix_name],
                atomic_types_to_species_index=self.atomic_types_to_species_index,
                n_types=len(self.atomic_types),
                data_processor=self.graph2mat_processors[matrix_name],
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
            edge_feats = self.radial_embeddings[matrix_name](
                lengths, data["node_attrs"], data["edge_index"], self.dataset_info.atomic_types
            )

            data["edge_attrs"] = edge_attrs
            data["edge_feats"] = edge_feats

            datas[matrix_name] = data

            # Run graph2mat and store the outputs (a tuple of tensors: node labels and edge labels)
            graph2mat_returns[matrix_name] = graph2mat(
                data=data, node_feats=graph2mat_inputs[matrix_name]
            )

        self.datas = datas

        # -----------------------------------
        #   Convert outputs to TensorMaps
        # -----------------------------------

        # At this point, we have a dictionary of all outputs as normal torch tensors.
        # Now, we simply convert to TensorMaps.

        # Get the labels for the samples (system and atom of each value)

        return_dict: Dict[str, TensorMap] = {
            **featurizer_return,
        }
        
        for matrix_name, graph2mat_return in graph2mat_returns.items():
            node_target = self.hypers["matrices"][matrix_name]["nodes"]
            edge_target = self.hypers["matrices"][matrix_name]["edges"]

            return_dict[node_target], return_dict[edge_target] = g2m_labels_to_tensormap(
                node_labels=graph2mat_return[0],
                edge_labels=graph2mat_return[1],
                dtype=graph2mat_return[0].dtype,
            )
                

        # -----------------------------------------
        #   Undo data preprocessing (eval only)
        # -----------------------------------------

        # At evaluation, we also introduce the scaler and additive contributions
        if not self.training:
            for matrix_name in graph2mat_returns.keys():
                node_target = self.hypers["matrices"][matrix_name]["nodes"]
                edge_target = self.hypers["matrices"][matrix_name]["edges"]
                
                return_dict.update(graph2mat_to_tensormap(
                    batch=datas[matrix_name],
                    out=return_dict,
                    processor=self.graph2mat_processors[matrix_name],
                    node_labels_name=node_target,
                    edge_labels_name=edge_target,
                ))
            
            return_dict = self.scaler(
                systems,
                return_dict,
                selected_atoms=selected_atoms,
                use_per_target_scales=True,
                use_per_property_scales=True,
            )
            self.add_additive_contributions(
                return_dict, systems, outputs, selected_atoms
            )

        return return_dict

    def add_additive_contributions(
        self,
        values: Dict[str, TensorMap],
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> None:
        """Adds the contributions from all additive models to the passed values.

        :param values: Dictionary of TensorMaps containing the current outputs
          (without additive contributions). The additive contributions will be added
          in place to this dictionary.
        :param systems: List of systems that have been evaluated to produce the outputs.
        :param outputs: Dictionary of requested ModelOutputs.
        :param selected_atoms: Optional Labels selecting a subset of atoms.
        """
        for additive_model in self.additive_models:
            outputs_for_additive_model: Dict[str, ModelOutput] = {}
            for name, output in outputs.items():
                if name in additive_model.outputs:
                    outputs_for_additive_model[name] = output
            additive_contributions = additive_model.forward(
                systems,
                outputs_for_additive_model,
                selected_atoms,
            )
            for name in additive_contributions:
                # # TODO: uncomment this after metatensor.torch.add is updated to
                # # handle sparse sums
                # return_dict[name] = metatensor.torch.add(
                #     return_dict[name],
                #     additive_contributions[name].to(
                #         device=return_dict[name].device,
                #         dtype=return_dict[name].dtype
                #         ),
                # )

                # TODO: "manual" sparse sum: update to metatensor.torch.add after
                # sparse sum is implemented in metatensor.operations
                output_blocks: List[TensorBlock] = []
                for k, b in values[name].items():
                    if k in additive_contributions[name].keys:
                        output_blocks.append(
                            _add_block_block(
                                b,
                                additive_contributions[name]
                                .block(k)
                                .to(device=b.device, dtype=b.dtype),
                            )
                        )
                    else:
                        output_blocks.append(b.copy())
                values[name] = TensorMap(values[name].keys, output_blocks)

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def requested_neighbor_lists(
        self,
    ) -> List[NeighborListOptions]:
        return [
            *self.featurizer_model.requested_neighbor_lists(),
            #*list(self.graph2mat_nls.values())
        ]

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "MetaGraph2Mat":
        if context in {"restart", "export"}:
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

        model.load_state_dict(model_state_dict)

        # Set up composition and scaler models
        model.additive_models[0].sync_tensor_maps()
        model.scaler.sync_tensor_maps()

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

    def capabilities(self) -> ModelCapabilities:
        dtype = next(self.parameters()).dtype

        capabilities = ModelCapabilities(
            outputs=self.outputs,
            atomic_types=self.atomic_types,
            interaction_range=4.0,
            length_unit=self.dataset_info.length_unit,
            supported_devices=self.__supported_devices__,
            dtype=dtype_to_str(dtype),
        )

        return capabilities
