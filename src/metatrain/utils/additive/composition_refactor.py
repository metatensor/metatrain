from typing import Dict, List
import torch

from metatensor.torch import TensorMap, TensorBlock, Labels
from metatensor.torch.atomistic import System


class CompositionModel(torch.nn.Module):
    """Fits for a dict of targets"""

    def __init__(
        self,
        atomic_types: List[int],
        layouts: Dict[str, TensorMap],
    ) -> None:

        target_names = []
        sample_kinds = {}
        for target_name, layout in layouts.items():  # identify target_type

            target_names.append(target_name)
            if layout.sample_names == ["system"]:
                sample_kinds[target_name] = "per_structure"

            elif layout.sample_names == ["system", "atom"]:
                sample_kinds[target_name] = "per_atom"

            elif layout.sample_names == [
                "system",
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ]:
                sample_kinds[target_name] = "per_pair"

            else:
                raise ValueError

        self.atomic_types = atomic_types
        self.target_names = target_names
        self.sample_kinds = sample_kinds
        self.XTX: Dict[str, TensorMap] = {
            target_name: TensorMap(
                layout.keys,
                blocks=[
                    TensorBlock(
                        values=torch.zeros(
                            len(self.atomic_types),
                            len(self.atomic_types),
                            dtype=torch.float64,
                        ),
                        samples=Labels(
                            ["first_atom_type"],
                            torch.tensor(self.atomic_types, dtype=torch.int32).reshape(
                                -1, 1
                            ),
                        ),
                        components=[],
                        properties=Labels(
                            ["second_atom_type"],
                            torch.tensor(self.atomic_types, dtype=torch.int32).reshape(
                                -1, 1
                            ),
                        ),
                    )
                    for _ in layout
                ],
            )
            for target_name, layout in layouts.items()
        }
        self.XTY: Dict[str, TensorMap] = {
            target_name: TensorMap(
                layout.keys,
                blocks=[
                    TensorBlock(
                        values=torch.zeros(
                            len(self.atomic_types),
                            *[len(c) for c in block.components],
                            len(block.properties),
                            dtype=torch.float64,
                        ),
                        samples=Labels(
                            ["center_type"],
                            torch.tensor(self.atomic_types, dtype=torch.int32).reshape(
                                -1, 1
                            ),
                        ),
                        components=block.components,
                        properties=block.properties,
                    )
                    for block in layout
                ],
            )
            for target_name, layout in layouts.items()
        }
        self.weights: Dict[str, TensorMap] = {}

    def _accumulate(
        self,
        systems: List[System],
        targets: Dict[str, TensorMap],
    ):

        num_atoms = torch.tensor([len(system) for system in systems])

        for target_name, target in targets.items():

            for key, block in target.items():

                # Get the target block values
                values = block.values

                if self.sample_kinds[target_name] == "per_structure":

                    # For per-structure, divide target values by number of atoms
                    values /= num_atoms.reshape(-1, *values.shape[1:])

                    # Compute X
                    X = self._compute_X_per_structure(systems)

                elif self.sample_kinds[target_name] == "per_atom":

                    # if not (key["o3_lambda"] == 0 and key["o3_sigma"] == 1):
                    #     continue

                    # X needs to be sliced based on atom type
                    if "center_type" in key.names:
                        center_types = [key["center_type"]]

                    else:
                        center_types = self.atomic_types

                    # Compute X
                    X = self._compute_X_per_atom(systems, center_types)

                else:
                    assert self.sample_kinds[target_name] == "per_pair"

                    # TODO: assumes coupled basis, perumtationally symmetrized

                    if "o3_lambda" in key.names:  # coupled

                        if not (key["o3_lambda"] == 0 and key["o3_sigma"] == 1):
                            continue

                    # X needs to be sliced based on atom type
                    if (
                        "first_atom_type" in key.names
                        and "second_atom_type" in key.names
                    ):
                        if key["first_atom_type"] != key["second_atom_type"]:
                            continue

                        if key["s2_pi"] != 0:
                            continue

                        center_types = [key["first_atom_type"]]

                    else:
                        center_types = self.atomic_types

                    # Compute X
                    X = self._compute_X_per_atom(systems, center_types)

                # Compute a sparse XTX
                self.XTX[target_name][key].values[:] += X.T @ X

                # Compute XTY
                if len(values.shape) == 2:
                    XTY = torch.einsum("sZ,sP->ZP", X, values)
                    self.XTY[target_name][key].values[:] += XTY

                else:
                    assert len(values.shape) == 3
                    XTY = torch.einsum("sZ,sCP->ZCP", X, values)
                    self.XTY[target_name][key].values[:] += XTY

    def _compute_X_per_structure(self, systems: List[System]) -> torch.Tensor:
        X = []
        for system in systems:
            X_system = torch.tensor(
                [
                    torch.sum(system.types == atom_type)
                    for atom_type in self.atomic_types
                ],
                dtype=torch.float64,
            )
            X.append(X_system / len(system))

        return torch.vstack(X)

    def _compute_X_per_atom(
        self, systems: List[System], center_types: List[int]
    ) -> torch.Tensor:

        X = []

        # TODO: make this one hot encoding quicker

        column_idx_map = {atom_type: i for i, atom_type in enumerate(self.atomic_types)}

        for system in systems:
            for atom_type in system.types:
                if atom_type in center_types:
                    row = torch.zeros(
                        len(self.atomic_types),
                        dtype=torch.float64,
                    )
                    row[column_idx_map[atom_type.item()]] = 1.0
                    X.append(row)

        return torch.vstack(X)

    def fit(self, dataloader, sigma: float = 0.01):

        # acccumulate
        for batch in dataloader:
            self._accumulate(
                batch.systems,
                {target_name: batch[target_name] for target_name in self.target_names},
            )

        # fit
        for target_name in self.target_names:

            if self.sample_kinds[target_name] == "per_structure":

                blocks = []
                for key in self.XTX[target_name].keys:

                    XTX_block = self.XTX[target_name][key]
                    XTY_block = self.XTY[target_name][key]
                    blocks.append(
                        TensorBlock(
                            values=_solve_linear_system(
                                XTX_block.values, XTY_block.values
                            ),
                            samples=XTY_block.samples,
                            components=XTY_block.components,
                            properties=XTY_block.properties,
                        )
                    )

                self.weights[target_name] = TensorMap(
                    self.XTX[target_name].keys, blocks
                )

            elif self.sample_kinds[target_name] in ["per_atom", "per_pair"]:

                blocks = []
                for key in self.XTX[target_name].keys:

                    XTX_block = self.XTX[target_name][key]
                    XTY_block = self.XTY[target_name][key]

                    XTX_values = XTX_block.values
                    XTY_values = XTY_block.values

                    # TODO: should non-invariant keys be even present?
                    weights_are_zero = False
                    if "o3_lambda" in key.names:
                        if not (key["o3_lambda"] == 0 and key["o3_sigma"] == 1):
                            weights_are_zero = True
                        # Weights are zero for off-site blocks
                        if "second_atom_type" in key.names:
                            if not (
                                key["s2_pi"] == 0
                                and key["first_atom_type"] == key["second_atom_type"]
                            ):
                                weights_are_zero = True

                    if weights_are_zero:
                        weight_block = torch.zeros_like(XTY_values)
                    else:
                        XTY_shape = XTY_values.shape
                        if len(XTY_values.shape) != 2:
                            XTY_values = XTY_values.reshape(XTY_values.shape[0], -1)

                        weight_block = _solve_linear_system(XTX_values, XTY_values)
                        weight_block = weight_block.reshape(*XTY_shape)

                    blocks.append(
                        TensorBlock(
                            values=weight_block,
                            samples=XTY_block.samples,
                            components=XTY_block.components,
                            properties=XTY_block.properties,
                        )
                    )

                self.weights[target_name] = TensorMap(
                    self.XTX[target_name].keys, blocks
                )


def _solve_linear_system(compf_t_at_compf, compf_t_at_targets) -> torch.Tensor:
    trace_magnitude = float(torch.diag(compf_t_at_compf).abs().mean())
    regularizer = 1e-14 * trace_magnitude
    return torch.linalg.solve(
        compf_t_at_compf
        + regularizer
        * torch.eye(
            compf_t_at_compf.shape[1],
            dtype=compf_t_at_compf.dtype,
            device=compf_t_at_compf.device,
        ),
        compf_t_at_targets,
    )


# def _tmp(composition_features, all_targets) -> torch.Tensor:
#     compf_t_at_compf = composition_features.T @ composition_features
#     compf_t_at_targets = composition_features.T @ all_targets
#     trace_magnitude = float(torch.diag(compf_t_at_compf).abs().mean())
#     regularizer = 1e-14 * trace_magnitude
#     return torch.linalg.solve(
#         compf_t_at_compf
#         + regularizer
#         * torch.eye(
#             composition_features.shape[1],
#             dtype=composition_features.dtype,
#             device=composition_features.device,
#         ),
#         compf_t_at_targets,
#     )


# def forward()
