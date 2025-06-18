"""
Contains the ``BaseCompositionModel class. This is intended for eventual porting to
metatomic. The class ``MetatrainCompositionModel`` wraps this to be compatible with
metatrain-style objects.
"""

from typing import Dict, List, Optional

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, LabelsEntry, TensorBlock, TensorMap
from metatensor.torch.learn.data import DataLoader
from metatomic.torch import ModelOutput, System


class BaseCompositionModel(torch.nn.Module):
    """
    Fits a composition model for a dict of targets.

    A composition model is a model that predicts the target values based on the
    composition of the system, i.e., the chemical identity of atoms in the system.

    Only invariant blocks of the specified targets are fitted, i.e. those indexed by
    keys with a single name "_" (for scalars) or keys with including names "o3_lambda"
    and "o3_sigma" (for spherical targets).
    """

    def __init__(
        self,
        atomic_types: List[int],
        layouts: Dict[str, TensorMap],
    ) -> None:
        """
        Initializes the composition model with the given atomic types and layouts.

        :param atomic_types: List of atomic types to use in the composition model.
        :param layouts: Dict of zero-sample layout :py:class:`TensorMap` corresponding
            to each target. The keys of the dict are the target names, and the values
            are :py:class:`TensorMap` objects with the zero-sample layout for each
            target.
        """
        super().__init__()
        target_names = []
        sample_kinds = {}
        for target_name, layout in layouts.items():  # identify target_type
            target_names.append(target_name)
            if layout.sample_names == ["system"]:
                sample_kinds[target_name] = "per_structure"

            elif layout.sample_names == ["system", "atom"]:
                sample_kinds[target_name] = "per_atom"

            # TODO: per_pair sample kinds.
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
                raise ValueError(
                    "unknown sample kind. TensorMap has sample names"
                    f" {layout.sample_names} but expected either"
                    "['system'], or ['system', 'atom']"
                )

        self.atomic_types = atomic_types
        self.target_names = target_names
        self.sample_kinds = sample_kinds

        # Find the keys that of the blocks the composition actually applies to
        in_keys = {
            target_name: Labels(
                layout.keys.names,
                torch.vstack([key.values for key in layout.keys if _include_key(key)]),
            )
            for target_name, layout in layouts.items()
        }

        # Filter the layout TensorMaps according to these sliced keys
        layouts = {
            target_name: mts.filter_blocks(layout, in_keys[target_name])
            for target_name, layout in layouts.items()
        }

        # Initialize dict of TensorMaps for XTX and XTY:
        #
        #  - XTX is a square matrix of shape (n_atomic_types, n_atomic_types)
        #  - XTY is a matrix of shape (n_atomic_types, n_components, n_properties)
        #
        # Both are initialized with zeros, and accumulated during fitting by iterating
        # over batches in the passed dataloader.
        #
        # Then a linear system is solved for each target to obtain the
        # composition weights, which are stored in the `weights` attribute.
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
        self.weights: Dict[str, TensorMap] = {
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
        self.is_fitted: bool = False

    def _accumulate(
        self,
        systems: List[System],
        targets: Dict[str, TensorMap],
    ) -> None:
        """
        Accumulates the necessary quantities (XTX and XTY) for the composition model
        based on the provided systems and targets.
        """
        for target_name, target in targets.items():
            for key, block in target.items():
                if not _include_key(key):
                    continue

                # Get the target block values
                Y = block.values

                if self.sample_kinds[target_name] == "per_structure":
                    X = self._compute_X_per_structure(systems)

                elif self.sample_kinds[target_name] in [
                    "per_atom",
                    # "per_pair"
                ]:
                    X = self._compute_X_per_atom(
                        systems, self._get_sliced_atomic_types(key)
                    )

                else:
                    raise ValueError(
                        f"unknown sample kind: {self.sample_kinds[target_name]}"
                        f" for target {target_name}"
                    )

                # Compute "XTX", i.e. X.T @ X
                self.XTX[target_name][key].values[:] += X.T @ X

                # Compute "XTY", i.e. X.T @ Y
                self.XTY[target_name][key].values[:] += torch.tensordot(
                    X, Y, dims=([0], [0])
                )

    def fit(self, dataloader: DataLoader) -> None:
        """
        Iterates over the batches in ``dataloader`` and accumulates the necessary
        quantities (XTX and XTY). Then, fits the compositions for each target.

        Assumes the systems are stored in the ``system`` attribute of the batch.
        """

        # TODO: add an option to pass a subset of the names of the targets to fit.

        # accumulate
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

            elif self.sample_kinds[target_name] in [
                "per_atom",
                # "per_pair",
            ]:
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
        outputs: Dict[str, ModelOutput],
        selected_samples: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """
        Compute the targets for each system based on the composition weights.

        :param systems: List of systems to calculate the energy.
        :param output_names: List of output names to compute. These should be a subset
            of the target names used during fitting.
        :param selected_atoms: Optional selection of atoms for which to compute the
            predictions.
        :returns: A dictionary with the computed predictions for each system.

        :raises ValueError: If no weights have been computed or if `outputs` keys
            contain unsupported keys.
        """
        predictions: Dict[str, TensorMap] = {}
        for output_name, model_output in outputs.items():
            if output_name not in self.target_names:
                raise ValueError(
                    f"output key {output_name} is not supported by this composition "
                    "model."
                )
            weights = self.weights[output_name]

            prediction_key_vals = []
            prediction_blocks: List[TensorBlock] = []
            for key, weight_block in weights.items():
                # Compute X
                if self.sample_kinds[output_name] == "per_structure":
                    if model_output.per_atom:
                        sample_values = []
                        for A, system in enumerate(systems):
                            for i in torch.arange(len(system), dtype=torch.int32):
                                sample_values.append(
                                    torch.tensor([int(A), int(i)], dtype=torch.int32)
                                )
                        sample_labels = Labels(
                            ["system", "atom"],
                            torch.vstack(sample_values),
                        )
                        X = self._compute_X_per_atom(
                            systems, self._get_sliced_atomic_types(key)
                        )

                    else:
                        sample_labels = Labels(
                            ["system"],
                            torch.arange(len(systems), dtype=torch.int32).reshape(
                                -1, 1
                            ),
                        )
                        X = self._compute_X_per_structure(systems)

                # TODO: add support for per_pair. As compositions are only fitted for
                # on-site blocks this extension is simple, reusing the per_atom code.
                elif self.sample_kinds[output_name] in [
                    "per_atom",
                    # "per_pair",
                ]:
                    sample_values = []
                    for A, system in enumerate(systems):
                        for i in torch.arange(len(system), dtype=torch.int32):
                            sample_values.append(
                                torch.tensor([int(A), int(i)], dtype=torch.int32)
                            )
                    sample_labels = Labels(
                        ["system", "atom"],
                        torch.vstack(sample_values),
                    )
                    X = self._compute_X_per_atom(
                        systems, self._get_sliced_atomic_types(key)
                    )

                else:
                    raise ValueError(
                        f"unknown sample kind: {self.sample_kinds[output_name]}"
                        f" for target {output_name}"
                    )

                # If selected_samples is provided, slice the samples labels and the X
                # tensor
                if selected_samples is not None:
                    sample_indices = sample_labels.select(selected_samples)
                    sample_labels = Labels(
                        sample_labels.names,
                        sample_labels.values[sample_indices],
                    )
                    X = X[sample_indices]

                # Compute X.T @ W
                out_vals = torch.tensordot(X, weight_block.values, dims=([1], [0]))
                prediction_blocks.append(
                    TensorBlock(
                        values=out_vals,
                        samples=sample_labels,
                        components=weight_block.components,
                        properties=weight_block.properties,
                    )
                )
                prediction_key_vals.append(key.values)

            prediction = TensorMap(
                Labels(
                    self.weights[output_name].keys.names,
                    torch.vstack(prediction_key_vals),
                ),
                prediction_blocks,
            )
            predictions[output_name] = prediction

        return predictions

    def _get_sliced_atomic_types(self, key: LabelsEntry) -> List[int]:
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
        """
        Computes the one-hot encoding of the atomic types for the atoms in the
        provided systems.

        Returns a tensor of shape (n_systems, n_atomic_types), where each row
        corresponds to a system and each column corresponds to an atomic type. The
        value is the number of atoms of that type in the system.
        """
        X = []
        for system in systems:
            X_system = torch.tensor(
                [
                    int(torch.sum(system.types == atom_type))
                    for atom_type in self.atomic_types
                ],
                dtype=torch.float64,
            )
            X.append(X_system)

        return torch.vstack(X)

    def _compute_X_per_atom(
        self, systems: List[System], center_types: List[int]
    ) -> torch.Tensor:
        """
        Computes the one-hot encoding of the atomic types for the atoms in the provided
        systems, but only for the specified center types.

        Returns a tensor of shape (n_atoms, n_atomic_types), where each row corresponds
        to an atom in the systems and each column corresponds to an atomic type. The
        value is 1 if the atom's type matches the atomic type, and 0 otherwise.
        """
        # Create a Labels of the samples
        sample_values = []
        for A, system in enumerate(systems):
            for i in torch.arange(len(system), dtype=torch.int32):
                sample_values.append(
                    torch.tensor(
                        [int(A), int(i), int(system.types[i])], dtype=torch.int32
                    )
                )
        sample_labels = Labels(
            ["system", "atom", "center_type"],
            torch.vstack(sample_values),
        )

        # Create a Labels object of the possible center types
        center_types_labels = Labels(
            ["center_type"],
            torch.tensor(center_types, dtype=torch.int32).reshape(-1, 1),
        )

        return mts.one_hot(sample_labels, center_types_labels).to(torch.float64)


def _include_key(key: LabelsEntry) -> bool:
    """
    Determines whether a block indexed by the input ``key`` should be included in the
    composition model.

    The rules are as follows:
        - If the key has a single name "_" (indicating a scalar),
          it is included.
        - If the key has names "o3_lambda" and "o3_sigma", it is included
          if values are 0 and 1 respectively (indicating an invariant block of a
          spherical target).
    """
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

        # TODO: for two-center targets include only invariant on-site blocks, assuming
        # permutational symmetrization.
        # elif "first_atom_type" in key.names and "second_atom_type" in key.names:
        #   if (
        #     key["o3_lambda"] == 0 and
        #     key["o3_sigma"] == 1 and
        #     key["s2_pi"] == 0 and
        #     key["first_atom_type"] == key["second_atom_type"]
        # ):
        #     include_key = True

        else:
            raise ValueError("unknown target type")

    return include_key


def _solve_linear_system(
    XTX_vals: torch.Tensor, XTY_vals: torch.Tensor
) -> torch.Tensor:
    """
    Solves the linear system XTX * W = XTY for the weights W, where XTX is a square
    matrix of shape (n_atomic_types, n_atomic_types) and XTY is a matrix
    of shape (n_atomic_types, n_components, n_properties).

    :py:func:`metatensor.torch.solve` is not used due to numerical stability issues
    when the matrix is ill-conditioned. Instead, a regularization term is added to the
    diagonal of XTX to improve stability.
    """
    trace_magnitude = float(torch.diag(XTX_vals).abs().mean())
    regularizer = 1e-14 * trace_magnitude
    shape = (XTX_vals.shape[0], *XTY_vals.shape[1:])
    return torch.linalg.solve(
        XTX_vals
        + regularizer
        * torch.eye(
            XTX_vals.shape[1],
            dtype=XTX_vals.dtype,
            device=XTX_vals.device,
        ),
        XTY_vals.reshape(XTY_vals.shape[0], -1),
    ).reshape(shape)
