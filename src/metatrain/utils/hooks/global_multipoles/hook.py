from typing import Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.sum_over_atoms import sum_over_atoms

from ..abc import HookInterface
from .documentation import Hypers


class GlobalMultipole(HookInterface[Hypers]):
    """
    Computes a global multipole from local predictions.

    :param hypers: A dictionary with the hook's hyper-parameters.
    :param dataset_info: Information containing details about the dataset, such as
        target quantities and atomic types.
    """

    __checkpoint_version__ = 1

    def __init__(self, hypers: Hypers, dataset_info: DatasetInfo):
        super().__init__(hypers, dataset_info)

        self.hypers = hypers

        # Origin the positions are referred to. Stored as a plain string so
        # that the forward stays TorchScript-friendly.
        origin = hypers.get("origin", "center_of_charge")
        if origin not in ("center_of_charge", "absolute"):
            raise ValueError(
                f"Invalid 'origin' for the global multipoles hook: {origin!r}. "
                f"Expected either 'center_of_charge' or 'absolute'."
            )
        self.origin: str = origin

        # Get the information about the output targets from the dataset info
        out_names = hypers["outputs"]
        if isinstance(out_names, str):
            out_names = [out_names]
        if out_names is None:
            raise ValueError("GlobalMultipole hook requires at least one output target")
        self.out_targets = {k: dataset_info.targets[k] for k in out_names}

        # Get the information for the input targets.
        input_names = hypers.get("inputs")
        if isinstance(input_names, str):
            input_names = [input_names]
        elif input_names is None:
            input_names = [
                f"mtt::aux::local_multipoles::{out_name.replace('mtt::', '')}"
                for out_name in out_names
            ]

        if len(input_names) != len(out_names):
            raise ValueError(
                f"Global multipoles hook expects the same number of input and output "
                f"targets, but got {len(input_names)} inputs and "
                f"{len(out_names)} outputs"
            )

        # Build the target infos that the hook will request.
        self._input_target_infos = {}
        for in_name, (out_name, out_target) in zip(
            input_names, self.out_targets.items(), strict=True
        ):
            if out_target.sample_kind != "system":
                raise ValueError(
                    f"Global multipoles hook only supports system-level outputs, "
                    f"but {out_name} has sample kind "
                    f"{out_target.sample_kind}"
                )
            if not out_target.is_spherical:
                raise ValueError(
                    f"Global multipoles hook only supports spherical outputs, "
                    f"but {out_name} has components "
                    f"{out_target.layout.components}"
                )

            max_degree = out_target.layout.keys["o3_lambda"].max().item()
            if max_degree > 1:
                raise ValueError(
                    f"Global multipoles hook only supports multipoles up "
                    f"to l=1 for now, but {out_name} has max degree "
                    f"{max_degree}"
                )

            # Get the input that we will request from the model, which are
            # the local multipoles up to the degree of the output multipole.
            self._input_target_infos[in_name] = get_generic_target_info(
                in_name,
                {
                    "quantity": "",
                    "unit": "",
                    "type": {
                        "spherical": {
                            "irreps": [
                                {"o3_lambda": i, "o3_sigma": 1}
                                for i in range(max_degree + 1)
                            ]
                        }
                    },
                    "num_subtargets": 1,
                    "sample_kind": "atom",
                },
            )
            out_target.layout.set_info("max_degree", str(max_degree))
            out_target.layout.set_info("input_name", in_name)

    def requested_target_infos(self) -> dict[str, TargetInfo]:
        """
        Returns the list of requested target infos for the hook.

        :return: A list of requested target names.
        """
        return self._input_target_infos

    def requested_inputs(self) -> dict[str, ModelOutput]:
        """
        Returns the list of requested inputs for the hook.

        :return: A list of requested input names.
        """
        return {
            in_name: ModelOutput(
                quantity=target_info.quantity,
                unit=target_info.unit,
                sample_kind="atom",
            )
            for in_name, target_info in self._input_target_infos.items()
        }

    def supported_outputs(self) -> dict[str, ModelOutput]:
        """
        Returns the supported outputs for the hook.

        :return: A list of supported output names.
        """
        return {
            out_name: ModelOutput(
                quantity=target.quantity,
                unit=target.unit,
                sample_kind="system",
            )
            for out_name, target in self.out_targets.items()
        }

    def forward(
        self,
        systems: list[System],
        outputs: dict[str, ModelOutput],
        inputs: dict[str, TensorMap],
        selected_atoms: Optional[Labels] = None,
    ) -> dict[str, TensorMap]:
        requested_outs: list[str] = []
        for out_name in self.out_targets:
            if out_name in outputs:
                requested_outs.append(out_name)
        if not requested_outs:
            # Quick exit if no output of this hook is requested
            return {}

        device = systems[0].positions.device

        positions = torch.cat([system.positions for system in systems], dim=0)

        if self.origin == "center_of_charge":
            # Enforce origin independence by referring the positions of each
            # system to its own centre of nuclear charge. The per-system sums
            # are done with a scatter over the concatenated batch; note that
            # the origin is computed per system and never over the whole batch,
            # which would make the prediction depend on how systems are
            # batched together.
            nuclear_charges = torch.cat(
                [system.types for system in systems], dim=0
            ).to(positions.dtype)
            sizes = torch.tensor(
                [len(system) for system in systems], device=device, dtype=torch.long
            )
            system_indices = torch.repeat_interleave(
                torch.arange(len(systems), device=device), sizes
            )
            totals = torch.zeros(
                len(systems), dtype=positions.dtype, device=device
            ).index_add_(0, system_indices, nuclear_charges)
            origins = torch.zeros(
                (len(systems), 3), dtype=positions.dtype, device=device
            ).index_add_(0, system_indices, nuclear_charges.unsqueeze(1) * positions)
            origins = origins / totals.unsqueeze(1)

            positions = positions - origins[system_indices]

        # Reorder the axes to match the spherical harmonics convention
        # (x, y, z) -> (y, z, x)
        positions = positions[:, [1, 2, 0]]
        if selected_atoms is not None:
            # Here we should filter out the positions only for the selected atoms
            raise NotImplementedError(
                "Global multipoles hook does not support selected atoms yet."
            )

        return_dict: dict[str, TensorMap] = {}
        for out_name in requested_outs:
            out_target = self.out_targets[out_name]

            layout = out_target.layout.to(device)
            info = layout.info()
            max_degree = int(info["max_degree"])

            # Get the local predictions for each degree in the multipole expansion
            input_tmap = inputs[info["input_name"]]
            local_values = [
                input_tmap.block(dict(o3_lambda=ell, o3_sigma=1)).values
                for ell in range(max_degree + 1)
            ]

            # Compute the local contributions to the global multipole for each degree
            local_contribs = []
            for ell in range(max_degree + 1):
                if ell == 0:
                    local_contribs.append(local_values[ell])
                elif ell == 1:
                    local_contribs.append(
                        torch.einsum(
                            "sp, sx -> sxp",
                            local_contribs[ell - 1].squeeze(1),
                            positions,
                        )
                        + local_values[ell]
                    )

            # Build a tensor map with the requested multipole degrees.
            local_tmap = TensorMap(
                keys=layout.keys,
                blocks=[
                    TensorBlock(
                        values=local_contribs[key["o3_lambda"]],
                        samples=input_tmap.block(0).samples,
                        components=layout_block.components,
                        properties=layout_block.properties,
                    )
                    for key, layout_block in layout.items()
                ],
            )

            return_dict[out_name] = sum_over_atoms(local_tmap)

        return return_dict
