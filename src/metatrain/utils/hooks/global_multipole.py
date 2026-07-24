import torch
from metatensor.torch import TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System
from typing_extensions import TypedDict

from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.sum_over_atoms import sum_over_atoms


class HookHypers(TypedDict):
    """
    Hyperparameters for the global multipole hook.
    """

    inputs: str

    outputs: str


class GlobalMultipole(torch.nn.Module):
    """
    Computes a global multipole from local predictions.
    """

    def __init__(self, hypers: HookHypers, dataset_info: DatasetInfo):
        super().__init__()

        self.hypers = hypers

        # Get the information about the output target from the dataset info
        self.out_name = hypers["outputs"]
        self.out_target = dataset_info.targets[self.out_name]

        self.degrees = self.out_target.layout.keys["o3_lambda"]
        self.max_degree = self.degrees.max().item()

        if self.max_degree > 1:
            raise ValueError(
                f"Global multipoles hook only supports multipoles up "
                f"to l=1 for now, but {self.out_name} has max degree "
                f"{self.max_degree}"
            )

        # Build the input target that we will request from the model,
        # which is the local multipoles
        self._input_name = "mtt::aux::local_multipoles"

        self._input_target_info = get_generic_target_info(
            self._input_name,
            {
                "quantity": "",
                "unit": "",
                "type": {
                    "spherical": {
                        "irreps": [
                            {"o3_lambda": i, "o3_sigma": 1}
                            for i in range(self.max_degree + 1)
                        ]
                    }
                },
                "num_subtargets": 1,
                "sample_kind": "atom",
            },
        )

    def requested_target_infos(self) -> dict[str, TargetInfo]:
        """
        Returns the list of requested target infos for the hook.

        :return: A list of requested target names.
        """
        return {self._input_name: self._input_target_info}

    def requested_inputs(self) -> dict[str, ModelOutput]:
        """
        Returns the list of requested inputs for the hook.

        :return: A list of requested input names.
        """
        return {
            self._input_name: ModelOutput(
                quantity="",
                unit="",
                sample_kind="atom",
            )
        }

    def forward(
        self, systems: list[System], inputs: dict[str, TensorMap]
    ) -> dict[str, TensorMap]:
        """
        Computes the global multipole from the local predictions.
        """
        device = systems[0].positions.device
        layout = self.out_target.layout.to(device)

        # Get the concatenated positions of all atoms in the systems,
        # and reorder the axes to match the spherical harmonics convention
        # (x, y, z) -> (y, z, x)
        positions = torch.cat([s.positions for s in systems], dim=0)
        positions = positions[:, [1, 2, 0]]

        # Get the local predictions for each degree in the multipole expansion
        input_tmap = inputs[self._input_name]
        local_values = [
            input_tmap.block(dict(o3_lambda=ell, o3_sigma=1)).values
            for ell in range(self.max_degree + 1)
        ]

        # Compute the local contributions to the global multipole for each degree
        local_contribs = []
        for ell in range(self.max_degree + 1):
            if ell == 0:
                local_contribs.append(local_values[ell])
            elif ell == 1:
                local_contribs.append(
                    torch.einsum(
                        "sp, sx -> sxp", local_contribs[ell - 1].squeeze(1), positions
                    )
                    + local_values[ell]
                )
            else:
                raise ValueError(
                    "Global multipoles hook only supports multipoles up to l=1 for now"
                    f", but {self.out_name} has degree {ell}"
                )

        # Build a tensor map with the requested multipole degrees.
        local_tmap = TensorMap(
            keys=self.out_target.layout.keys,
            blocks=[
                TensorBlock(
                    values=local_contribs[degree],
                    samples=input_tmap.block(0).samples,
                    components=layout.block(i).components,
                    properties=layout.block(i).properties,
                )
                for i, degree in enumerate(self.degrees)
            ],
        )

        # Return the global multipole by summing over the atoms in the system
        return {self.out_name: sum_over_atoms(local_tmap)}
