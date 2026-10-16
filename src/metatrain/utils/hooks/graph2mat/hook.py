from typing import Any, Optional

from metatensor.torch import Labels, TensorMap
from metatomic.torch import ModelOutput, System, NeighborListOptions
from sphericart.torch import SphericalHarmonics


from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.data.atom_pair_helpers import get_pair_sample_labels
from metatrain.utils.data.atomic_basis_helpers import get_per_atom_sample_labels
from metatrain.experimental.mace.utils.mts import get_e3nn_mts_layout, target_info_to_e3nn_irreps, e3nn_to_tensormap
from metatrain.experimental.mace.utils.structures import create_batch

from e3nn import o3
import torch

from ..abc import HookInterface
from .documentation import Hypers
from .utils.edge_embeddings import RadialEmbeddingBlock
from .utils.edge_operations import MACEEdgeState
from .utils.structures import get_edge_vectors_and_lengths

class Graph2MatHook(HookInterface[Hypers]):
    """
    Passes the inputs to the outputs without any modification.

    :param hypers: A dictionary with the hook's hyper-parameters.
    :param dataset_info: Information containing details about the dataset, such as
        target quantities and atomic types.
    """

    __checkpoint_version__ = 1

    def __init__(self, hypers: Hypers, dataset_info: DatasetInfo):
        super().__init__(hypers, dataset_info)

        # Get the information about the output targets from the dataset info
        out_names = hypers["outputs"]
        if isinstance(out_names, str):
            out_names = [out_names]
        if out_names is None:
            raise ValueError("Graph2mat hook requires at least one output target")

        # Modules to embed edges
        self.spherical_harmonics = SphericalHarmonics(
            self.hypers["edge_lmax"],
        )
        sh_irreps = o3.Irreps.spherical_harmonics(self.hypers["edge_lmax"])
        n_basis = 10
        self.radial_embeddings = RadialEmbeddingBlock(
            r_max=self.hypers["cutoff"],
            num_bessel=n_basis,
            num_polynomial_cutoff=10,
        )
        radial_irreps = o3.Irreps(f"{n_basis}x0e")

        self.requested_nl = NeighborListOptions(
            cutoff=self.hypers["cutoff"],
            full_list=True,
            strict=True,
        )

        # Build the target infos that the hook will request.
        self.out_targets = {}
        self._input_target_infos = {}
        operations = []
        targets = dataset_info.targets
        for out_name in out_names:
            target = targets[out_name]

            self.out_targets[out_name] = target
            node_irreps = self.hypers["node_hidden_irreps"]
            layout = get_e3nn_mts_layout(
                "mtt::aux::mace_features",
                {
                    "type": {"spherical": {"irreps": node_irreps }},
                    "sample_kind": "atom",
                    "properties_name": "feature",
                },
            )
            layout = TensorMap(
                keys=Labels(
                    names=[name for name in layout.keys.names if name != "i_irrep"],
                    values=layout.keys.values[:, :-1]
                ),
                blocks=[block.copy() for block in layout.blocks()]
            )
            self._input_target_infos["mtt::aux::graph2mat::node_states"] = TargetInfo(
                layout
            )
            edge_irreps = target_info_to_e3nn_irreps(target)
            operations.append(
                MACEEdgeState(
                    irreps={
                        "node_feats_irreps": o3.Irreps(node_irreps),
                        "edge_attrs_irreps": sh_irreps,
                        "edge_feats_irreps": radial_irreps,
                        "message_irreps": o3.Irreps("20x0e + 20x1o + 20x2e + 20x3o + 20x4e"),
                        "edge_targets_irreps": o3.Irreps(edge_irreps),
                    }
                )
            )

        self.operations = torch.nn.ModuleList(operations)

        torch.jit.script(self)


    def requested_target_infos(self) -> dict[str, TargetInfo]:
        """
        Returns the list of requested target infos for the hook.

        :return: A list of requested target names.
        """
        return self._input_target_infos

    def requested_hook_inputs(
        self, outputs: dict[str, ModelOutput]
    ) -> dict[str, ModelOutput]:
        """
        Returns the list of requested inputs for the hook.

        :param outputs: Dictionary of requested outputs. These can contain outputs that
            are handled by other hooks or the main model. In that case, the hook
            ignores those outputs.
        :return: A list of requested input names.
        """
        req_inputs: dict[str, ModelOutput] = {}
        for out_name, (in_name, target_info) in zip(
            self.out_targets, self._input_target_infos.items(), strict=True
        ):
            if out_name in outputs:
                req_inputs[in_name] = ModelOutput(
                    quantity=target_info.quantity,
                    unit=target_info.unit,
                    sample_kind=target_info.sample_kind,
                )

        return req_inputs

    def requested_neighbor_lists(
        self,
    ) -> list[NeighborListOptions]:
        return [self.requested_nl]

    def supported_outputs(self) -> dict[str, ModelOutput]:
        """
        Returns the supported outputs for the hook.

        :return: A list of supported output names.
        """
        return {
            out_name: ModelOutput(
                quantity=target_info.quantity,
                unit=target_info.unit,
                sample_kind=target_info.sample_kind,
            )
            for out_name, target_info in self.out_targets.items()
        }

    def forward(
        self,
        systems: list[System],
        outputs: dict[str, ModelOutput],
        inputs: dict[str, TensorMap],
        selected_atoms: Optional[Labels] = None,
    ) -> dict[str, TensorMap]:

        device = systems[0].positions.device

        if selected_atoms is not None:
            raise NotImplementedError(
                "Graph2mat hook does not support selected_atoms yet."
            )

        batch_data = create_batch(
            systems=systems,
            neighbor_list_options=self.requested_nl,
            atomic_types_to_species_index=torch.zeros(max(self.dataset_info.atomic_types) + 1, device=device, dtype=torch.int64),
            n_types=len(self.dataset_info.atomic_types),
            head_index=0,
        )

        vectors, lengths = get_edge_vectors_and_lengths(
            positions=batch_data["positions"],
            edge_index=batch_data["edge_index"],
            shifts=batch_data["shifts"],
        )

        per_atom_samples = get_per_atom_sample_labels(
            systems,
            system_ids=torch.arange(len(systems), device=device, dtype=torch.int64),
        )
        pair_samples = get_pair_sample_labels(
            sample_labels=per_atom_samples,
            centers=batch_data["edge_index"][0],
            neighbors=batch_data["edge_index"][1],
            cell_shifts=batch_data["unit_shifts"].T,
        )

        vectors_spherical = self.spherical_harmonics(vectors)
        radial_embeddings = self.radial_embeddings(
            lengths,
            batch_data["node_attrs"],
            batch_data["edge_index"],
            torch.tensor(self.dataset_info.atomic_types, device=device),
        )

        batch_data["edge_attrs"] = vectors_spherical
        batch_data["edge_feats"] = radial_embeddings

        in_names = list(self._input_target_infos.keys())
        out_names = list(self.out_targets.keys())
        return_dict: dict[str, TensorMap] = {}
        for i, operation in enumerate(self.operations):
            in_name = in_names[i]
            out_name = out_names[i]
            if out_name in outputs:
                node_states = inputs[in_name]
                n_samples = node_states.block(0).samples.values.shape[0]
                values = torch.concat(
                    [block.values.reshape(n_samples, -1) for block in node_states.blocks()],
                    dim=1
                )
                edge_values = operation(batch_data, values)

                return_dict[out_name] = e3nn_to_tensormap(
                    edge_values,
                    pair_samples,
                    self.out_targets[out_name].layout
                )

        return return_dict

    def get_checkpoint(self, best_state_dict: Optional[dict[str, Any]] = None) -> dict:
        """
        Get the checkpoint of the hook.

        :return: A dictionary containing the state of the hook.
        """
        state_dict = self.state_dict()
        return {
            "hook_name": "graph2mat",
            "hook_ckpt_version": self.__checkpoint_version__,
            "hook_data": {
                "hypers": self.hypers,
                "dataset_info": self.dataset_info,
            },
            "model_state_dict": state_dict,
        }

    @classmethod
    def load_checkpoint(cls, checkpoint: dict[str, Any], context: str) -> "Graph2MatHook":
        """
        Load the checkpoint of the hook.

        :param checkpoint: A dictionary containing the state of the hook.
        """
        if checkpoint["hook_ckpt_version"] != cls.__checkpoint_version__:
            raise ValueError(
                f"Invalid checkpoint version for the hook: "
                f"expected {cls.__checkpoint_version__}, got "
                f"{checkpoint['hook_ckpt_version']}"
            )

        hook = cls(
            hypers=checkpoint["hook_data"]["hypers"],
            dataset_info=checkpoint["hook_data"]["dataset_info"],
        )

        hook.load_state_dict(checkpoint["model_state_dict"])
        return hook
