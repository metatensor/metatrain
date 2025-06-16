"""
Contains the ``BaseCompositionModel class. This is intended for eventual porting to
metatomic. The class ``MetatrainCompositionModel`` wraps this to be compatible with
metatrain-style objects.
"""

from typing import Dict, List, Optional

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, LabelsEntry, TensorBlock, TensorMap
from metatensor.torch.atomistic import System
from metatensor.torch.learn.data import DataLoader


class BaseCompositionModel(torch.nn.Module):
    """Fits a composition model for a dict of targets."""

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

            # elif layout.sample_names == [
            #     "system",
            #     "first_atom",
            #     "second_atom",
            #     "cell_shift_a",
            #     "cell_shift_b",
            #     "cell_shift_c",
            # ]:
            #     sample_kinds[target_name] = "per_pair"

            else:
                raise ValueError

        self.atomic_types = atomic_types
        self.target_names = target_names
        self.sample_kinds = sample_kinds

        # find the keys that of the blocks the composition actually applies to
        in_keys = {
            target_name: Labels(
                layout.keys.names,
                torch.vstack([key.values for key in layout.keys if _include_key(key)]),
            )
            for target_name, layout in layouts.items()
        }

        # filter the layout TensorMaps according to these sliced keys
        layouts = {
            target_name: mts.filter_blocks(layout, in_keys[target_name])
            for target_name, layout in layouts.items()
        }
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
                            ["center_type"],
                            torch.tensor(self.atomic_types, dtype=torch.int32).reshape(
                                -1, 1
                            ),
                        ),
                        components=[],
                        properties=Labels(
                            ["center_type"],
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
        self.is_fitted = False

    def _accumulate(
        self,
        systems: List[System],
        targets: Dict[str, TensorMap],
    ) -> None:
        num_atoms = torch.tensor([len(system) for system in systems])

        for target_name, target in targets.items():
            for key, block in target.items():
                if not _include_key(key):
                    continue

                # Get the target block values
                Y = block.values

                if self.sample_kinds[target_name] == "per_structure":
                    X = self._compute_X_per_structure(systems)

                    # For per-structure, divide target values by number of atoms
                    Y /= num_atoms.reshape(-1, *Y.shape[1:])

                elif self.sample_kinds[target_name] in ["per_atom", "per_pair"]:
                    X = self._compute_X_per_atom(
                        systems, self._get_sliced_atomic_types(key)
                    )

                else:
                    raise ValueError(
                        f"unknown sample kind: {self.sample_kinds[target_name]}"
                        f" for target {target_name}"
                    )

                # Compute a sparse XTX
                self.XTX[target_name][key].values[:] += X.T @ X

                # Compute XTY
                self.XTY[target_name][key].values[:] += _compute_XTY(X, Y)

    def fit(self, dataloader: DataLoader) -> None:
        """
        Iterates over the batches in ``dataloader`` and accumulates the necessary
        quantities. Then, fits the compositions for each target.

        Assumes the systems are stored in the ``system`` attribute of the batch.
        """

        # acccumulate
        for batch in dataloader:
            self._accumulate(
                batch.system,
                {target_name: batch[target_name] for target_name in self.target_names},
            )

        # fit
        for target_name in self.target_names:
            blocks = []
            if self.sample_kinds[target_name] == "per_structure":
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

            elif self.sample_kinds[target_name] in ["per_atom", "per_pair"]:  # TODO: remove per_pair
                blocks = []
                for key in self.XTX[target_name].keys:
                    XTX_block = self.XTX[target_name][key]
                    XTY_block = self.XTY[target_name][key]

                    XTX_values = XTX_block.values
                    XTY_values = XTY_block.values

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

            else:
                raise ValueError(
                    f"unknown sample kind: {self.sample_kinds[target_name]}"
                    f" for target {target_name}"
                )

            self.weights[target_name] = TensorMap(self.XTX[target_name].keys, blocks)
            self.is_fitted = True

    def forward(
        self,
        systems: List[System],
        output_names: List[str],
        selected_samples: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """Compute the targets for each system based on the composition weights.

        :param systems: List of systems to calculate the energy.
        :param output_names: List of output names to compute. These should be a subset
            of the target names used during fitting.
        :param selected_atoms: Optional selection of atoms for which to compute the
            predictions.
        :returns: A dictionary with the computed predictions for each system.

        :raises ValueError: If no weights have been computed or if `outputs` keys
            contain unsupported keys.
        """

        if not self.is_fitted:
            raise ValueError(
                "The model has not been fitted yet. Please fit the model before "
                "calling forward."
            )

        outputs = {}
        for output_name in output_names:
            if output_name not in self.target_names:
                raise ValueError(
                    f"output key {output_name} is not supported by this composition "
                    "model."
                )
            weights = self.weights[output_name]

            output_key_vals = []
            output_blocks = []
            for key, weight_block in weights.items():
                # Compute X
                if self.sample_kinds[output_name] == "per_structure":
                    sample_labels = Labels(
                        ["system"],
                        torch.arange(len(systems), dtype=torch.int32).reshape(-1, 1),
                    )
                    X = self._compute_X_per_structure(systems)

                # TODO: add support for per_pair. As compositions are only fitted for on-site
                # blocks this extension is simple, reusing the per_atom code.
                elif self.sample_kinds[output_name] in ["per_atom", "per_pair"]:  # TODO: remove per_pair
                    sample_labels = Labels(
                        ["system", "atom"],
                        torch.vstack(
                            [
                                [A, i]
                                for A, system in enumerate(systems)
                                for i in torch.arange(len(system), dtype=torch.int32)
                            ],
                            dtype=torch.int64,
                        ),
                    )
                    X = self._compute_X_per_atom(
                        systems, self._get_sliced_atomic_types(key)
                    )

                else:
                    raise ValueError(
                        f"unknown sample kind: {self.sample_kinds[output_name]}"
                        f" for target {output_name}"
                    )

                # Predict this block
                output_key_vals.append(key.values)

                out_vals = _compute_XTW(X, weight_block.values)
                output_blocks.append(
                    TensorBlock(
                        values=out_vals,
                        samples=sample_labels,
                        # samples=Labels(
                        #     ["_"], torch.arange(out_vals.shape[0]).reshape(-1, 1)
                        # ),
                        components=weight_block.components,
                        properties=weight_block.properties,
                    )
                )

            output = TensorMap(
                Labels(
                    self.weights[output_name].keys.names,
                    torch.vstack(output_key_vals),
                ),
                output_blocks,
            )

            # apply selected_atoms to the composition if needed
            if selected_samples is not None:
                output = mts.slice(output, "samples", selected_samples)

            outputs[output_name] = output

        return outputs

    def _get_sliced_atomic_types(self, key) -> List[int]:
        """
        Gets the slice of atomic types needed for the block indexed by the input ``key``
        """

        center_types = self.atomic_types
        if "center_type" in key.names:
            center_types = [key["center_type"]]

        if "first_atom_type" in key.names and "second_atom_type" in key.names:
            assert (
                key["first_atom_type"] == key["second_atom_type"] and key["s2_pi"] == 0
            )
            center_types = [key["first_atom_type"]]

        return center_types

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
        # TODO: make this one hot encoding quicker

        column_idx_map = {atom_type: i for i, atom_type in enumerate(self.atomic_types)}

        X = []
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


def _include_key(key: LabelsEntry) -> bool:
    include_key = False

    if len(key.names) == 1 and key.names[0] == "_":  # scalar
        include_key = True

    if "o3_lambda" in key.names:
        if key.names == ["o3_lambda", "o3_sigma"]:  # normal spherical target
            if key["o3_lambda"] == 0 and key["o3_sigma"] == 1:
                include_key = True

        # For one-center targets include invariant blocks
        elif "center_type" in key.names:
            if key["o3_lambda"] == 0 and key["o3_sigma"] == 1:
                include_key = True

        # For two-center targets include only invariant off-site blocks. Assumes
        # symmetrized for now
        elif "first_atom_type" in key.names and "second_atom_type" in key.names:
            if (
                key["o3_lambda"] == 0
                and key["o3_sigma"] == 1
                and key["s2_pi"] == 0
                and key["first_atom_type"] == key["second_atom_type"]
            ):
                include_key = True

        else:
            raise ValueError("unknown target type")

    return include_key


def _solve_linear_system(
    XTX_vals: torch.Tensor, XTY_vals: torch.Tensor
) -> torch.Tensor:
    trace_magnitude = float(torch.diag(XTX_vals).abs().mean())
    regularizer = 1e-14 * trace_magnitude
    return torch.linalg.solve(
        XTX_vals
        + regularizer
        * torch.eye(
            XTX_vals.shape[1],
            dtype=XTX_vals.dtype,
            device=XTX_vals.device,
        ),
        XTY_vals,
    )


def _compute_XTY(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return torch.tensordot(X, Y, dims=([0], [0]))


def _compute_XTW(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    return torch.tensordot(X, W, dims=([1], [0]))
