"""
Contains the ``BaseCompositionModel class. This is intended for eventual porting to
metatomic. The class ``CompositionModel`` wraps this to be compatible with
metatrain-style objects.
"""

from typing import Dict, List, Optional, Tuple, Union

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, LabelsEntry, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.utils.data.pad import (
    get_atom_sample_labels,
    get_pair_sample_labels_onsite,
)


class BaseCompositionModel(torch.nn.Module):
    """
    Fits a composition model for a dict of targets.

    A composition model is a model that predicts the target values based on the
    composition of the system, i.e., the chemical identity of atoms in the system.

    Only invariant blocks of the specified targets are fitted, i.e. those indexed by
    keys with a single name "_" (for scalars) or keys where "o3_lambda=0" and
    "o3_sigma=1" (for spherical targets).

    The :py:method:`accumulate` method is used to accumulate the necessary quantities
    based on the training data, and the :py:method:`fit` method is used to
    fit the model based on the accumulated quantities. These should both be called
    before the :py:method:`forward` method is called to compute the predictions.

    .. attribute :: atomic_types

        List of atomic types used in the model.

    .. attribute :: target_names

        List of target names in the model.

    .. attribute :: weights

        Dict of :py:class:`TensorMap` containing the fitted weights for each target.

    .. attribute :: sample_kinds

        Dict of sample kinds for each target. The sample kind can be either
        "per_structure" or "per_atom".

    .. attribute :: type_to_index

        Tensor mapping atomic types to their index in ``atomic_types``.

    .. attribute :: XTX

        Dict of :py:class:`TensorMap` containing the accumulated X^T * X for each
        target.

    .. attribute :: XTY

        Dict of :py:class:`TensorMap` containing the accumulated X^T * Y for each
        target.

    :param atomic_types: List of atomic types to use in the composition model.
    :param layouts: Dict of zero-sample layout :py:class:`TensorMap` corresponding
        to each target. The keys of the dict are the target names, and the values
        are :py:class:`TensorMap` objects with the zero-sample layout for each
        target.
    """

    # Needed for torchscript compatibility
    atomic_types: torch.Tensor
    target_names: List[str]
    weights: Dict[str, TensorMap]
    sample_kinds: Dict[str, str]
    type_to_index: torch.Tensor
    XTX: Dict[str, TensorMap]
    XTY: Dict[str, TensorMap]

    def __init__(
        self,
        atomic_types: Union[List[int], torch.Tensor],
        layouts: Dict[str, TensorMap],
    ) -> None:
        super().__init__()

        self.atomic_types = torch.as_tensor(atomic_types, dtype=torch.int32)
        self.target_names = []
        self.sample_kinds = {}
        self.XTX = {}
        self.XTY = {}
        self.weights = {}

        # go from an atomic type to its position in `self.atomic_types`
        self.register_buffer(
            "type_to_index", torch.empty(max(self.atomic_types) + 1, dtype=torch.long)
        )
        for i, atomic_type in enumerate(self.atomic_types):
            self.type_to_index[atomic_type] = i

        # Add targets based on provided layouts
        for target_name, layout in layouts.items():
            self.add_output(target_name, layout)

    def add_output(self, target_name: str, layout: TensorMap) -> None:
        """
        Adds a new target to the composition model.

        :param target_name: Name of the target to add.
        :param layout: Layout of the target as a :py:class:`TensorMap`.
        """
        if target_name in self.target_names:
            raise ValueError(f"target {target_name} already exists in the model.")

        self.target_names.append(target_name)
        valid_sample_names = [
            ["system"],
            [
                "system",
                "atom",
            ],
            [
                "system",
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
        ]

        if layout.sample_names == valid_sample_names[0]:
            self.sample_kinds[target_name] = "per_structure"

        elif layout.sample_names == valid_sample_names[1]:
            self.sample_kinds[target_name] = "per_atom"

        elif layout.sample_names == valid_sample_names[2]:
            self.sample_kinds[target_name] = "per_pair"

        else:
            raise ValueError(
                "unknown sample kind. TensorMap has sample names"
                f" {layout.sample_names} but expected one of "
                f"{valid_sample_names}."
            )

        # First slice the layout to only include the keys that the composition
        # model applies to. This is done by filtering the keys that have a single name
        # "_" (for scalars) or keys where "o3_lambda=0" and "o3_sigma=1"
        # (for spherical targets).
        layout = mts.filter_blocks(
            layout,
            Labels(
                layout.keys.names,
                torch.vstack([key.values for key in layout.keys if _include_key(key)]),
            ),
        )

        # Initialize TensorMaps for XTX and XTY for this target.
        #
        #  - XTX is a square matrix of shape (n_atomic_types, n_atomic_types)
        #  - XTY is a matrix of shape (n_atomic_types, n_components, n_properties)
        #
        # Both are initialized with zeros, and accumulated during fitting by iterating
        # over batches in the passed dataloader.
        #
        # The weights are also a matrix of shape (n_atomic_types, n_components,
        # n_properties), and are initialized for torchscript compatibility.
        #
        # Then a linear system is solved for each target to obtain the composition
        # weights, which are stored in the `weights` attribute.
        self.XTX[target_name] = TensorMap(
            layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.zeros(
                        len(self.atomic_types),
                        len(self.atomic_types),
                        dtype=torch.float64,
                    ),
                    samples=Labels(["center_type"], self.atomic_types.reshape(-1, 1)),
                    components=[],
                    properties=Labels(
                        ["center_type"], self.atomic_types.reshape(-1, 1)
                    ),
                )
                for _ in layout
            ],
        )
        self.XTY[target_name] = TensorMap(
            layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.zeros(
                        len(self.atomic_types),
                        *[len(c) for c in block.components],
                        len(block.properties),
                        dtype=torch.float64,
                    ),
                    samples=Labels(["center_type"], self.atomic_types.reshape(-1, 1)),
                    components=block.components,
                    properties=block.properties,
                )
                for block in layout
            ],
        )
        self.weights[target_name] = TensorMap(
            layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.zeros(
                        len(self.atomic_types),
                        *[len(c) for c in block.components],
                        len(block.properties),
                        dtype=torch.float64,
                    ),
                    samples=Labels(["center_type"], self.atomic_types.reshape(-1, 1)),
                    components=block.components,
                    properties=block.properties,
                )
                for block in layout
            ],
        )

    def accumulate(
        self,
        systems: List[System],
        targets: Dict[str, TensorMap],
    ) -> None:
        """
        Takes a batch of systems and targets, and for each target accumulates the
        necessary quantities (XTX and XTY).

        :param systems: List of systems in the batch.
        :param targets: Dict of target names to :py:class:`TensorMap` containing
            the target values for each system in the batch.
        """

        device = systems[0].positions.device
        dtype = systems[0].positions.dtype
        self._sync_device_dtype(device, dtype)

        # check that the systems contain no unexpected atom types
        for system in systems:
            if not torch.all(torch.isin(system.types, self.atomic_types)):
                raise ValueError(
                    "system contains unexpected atom types. "
                    f"Expected atomic types: {self.atomic_types}, "
                    f"found: {torch.unique(system.types)}"
                )

        # accumulate
        for target_name, target in targets.items():
            for key, block in target.items():
                if not _include_key(key):
                    continue

                # Get the target block values
                Y = block.values
                mask = ~torch.isnan(Y).any(dim=tuple(range(1, Y.ndim)))

                if self.sample_kinds[target_name] == "per_structure":
                    X = self._compute_X_per_structure(systems)

                elif self.sample_kinds[target_name] in ["per_atom", "per_pair"]:
                    X = self._compute_X_per_atom(systems, self.atomic_types)

                else:
                    raise ValueError(
                        f"unknown sample kind: {self.sample_kinds[target_name]}"
                        f" for target {target_name}"
                    )
                X = X.to(device=device, dtype=dtype)

                # Compute "XTX", i.e. X.T @ X
                # TODO: store XTX by sample kind instead, saving memory
                self.XTX[target_name][key].values[:] += X[mask].T @ X[mask]

                # Compute "XTY", i.e. X.T @ Y
                self.XTY[target_name][key].values[:] += torch.tensordot(
                    X[mask], Y[mask], dims=([0], [0])
                )

    def fit(
        self,
        fixed_weights: Optional[Dict[str, Dict[int, float]]] = None,
        targets_to_fit: Optional[List[str]] = None,
    ) -> None:
        """
        Based on the pre-accumulated quantities from the training data, fits the
        compositions for each target.

        :param fixed_weights: Optional dict of target names to dict of atomic types
            to fixed weights. If provided, the weights for the specified atomic types
            will be fixed to the provided values, and the weights for the other atomic
            types will be fitted normally.
        :param targets_to_fit: List of target names to fit. If `None`,
            all targets in the model will be fitted.
        """
        if targets_to_fit is None:
            targets_to_fit = self.target_names

        if fixed_weights is None:
            fixed_weights = {}

        # fit
        for target_name in targets_to_fit:
            blocks = []
            for key in self.XTX[target_name].keys:
                XTX_block = self.XTX[target_name][key]
                XTY_block = self.XTY[target_name][key]

                XTX_values = XTX_block.values
                XTY_values = XTY_block.values

                if target_name in fixed_weights:
                    weight_vals = torch.vstack(
                        [
                            torch.full(
                                (
                                    1,
                                    *[len(c) for c in XTY_block.components],
                                    len(XTY_block.properties),
                                ),
                                fixed_weights[target_name][int(atomic_type)],
                                dtype=XTY_values.dtype,
                                device=XTY_values.device,
                            )
                            for atomic_type in self.atomic_types
                        ]
                    )
                else:
                    XTY_shape = XTY_values.shape
                    if len(XTY_values.shape) != 2:
                        XTY_values = XTY_values.reshape(XTY_values.shape[0], -1)
                    weight_vals = _solve_linear_system(XTX_values, XTY_values)
                    weight_vals = weight_vals.reshape(*XTY_shape)

                blocks.append(
                    TensorBlock(
                        values=weight_vals.contiguous(),
                        samples=XTY_block.samples.to(device=weight_vals.device),
                        components=XTY_block.components,
                        properties=XTY_block.properties.to(device=weight_vals.device),
                    )
                )

            self.weights[target_name] = TensorMap(
                self.XTX[target_name].keys.to(device=weight_vals.device),
                blocks,
            )

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """
        Compute the targets for each system based on the composition weights.

        :param systems: List of systems to calculate the energy.
        :param outputs: Dict of named outputs to compute. These names (in the keys)
            should be a subset of the target names used during fitting, and the values
            are the corresponding :py:class:`ModelOutput` objects.
        :param selected_atoms: Optional selection of atoms for which to compute the
            predictions.
        :return: A dictionary with the computed predictions for each system.

        :raises ValueError: If no weights have been computed or if `outputs` keys
            contain unsupported keys.
        """

        device = systems[0].positions.device
        dtype = systems[0].positions.dtype
        self._sync_device_dtype(device, dtype)

        # Build the sample labels that are required
        sample_labels = _get_sample_labels(systems, self.sample_kinds, outputs, device)

        # Compute the X tensor
        X = self._compute_X_per_atom(systems, self.atomic_types)

        # Build the predictions for each output
        predictions: Dict[str, TensorMap] = {}
        for output_name in outputs:
            if output_name not in self.target_names:
                raise ValueError(
                    f"output {output_name} is not supported by this composition model."
                )
            weights = self.weights[output_name]

            prediction_key_vals = []
            prediction_blocks: List[TensorBlock] = []
            for key, weight_block in weights.items():
                sample_labels_block = sample_labels[self.sample_kinds[output_name]]

                # If selected_atoms is provided, slice the samples labels and the X
                # tensor
                if selected_atoms is None:
                    X_block = X
                else:
                    sample_indices = sample_labels_block.select(selected_atoms)
                    sample_labels_block = Labels(
                        sample_labels_block.names,
                        sample_labels_block.values[sample_indices],
                    ).to(device=device)
                    X_block = X[sample_indices]

                # Compute X.T @ W
                out_vals = torch.tensordot(
                    X_block, weight_block.values, dims=([1], [0])
                )
                prediction_blocks.append(
                    TensorBlock(
                        values=out_vals,
                        samples=sample_labels_block,
                        components=weight_block.components,
                        properties=weight_block.properties,
                    )
                )
                prediction_key_vals.append(key.values)

            prediction = TensorMap(
                Labels(
                    self.weights[output_name].keys.names,
                    torch.vstack(prediction_key_vals),
                    assume_unique=True,
                ),
                prediction_blocks,
            )

            # If a per-structure output is requested, sum over the sample dimensions
            # that aren't "system".
            if not outputs[output_name].per_atom:
                prediction = mts.sum_over_samples(prediction, "atom")
            predictions[output_name] = prediction

        return predictions

    def _compute_X_per_structure(self, systems: List[System]) -> torch.Tensor:
        """
        Computes the one-hot encoding of the atomic types for the atoms in the
        provided systems.

        :param systems: List of systems to compute the one-hot encoding for.

        :return: Tensor of shape ``(n_systems, n_atomic_types)``, where each row
            corresponds to a system and each column corresponds to an atomic type. The
            value is the number of atoms of that type in the system.
        """
        dtype = systems[0].positions.dtype

        counts = []
        for system in systems:
            bincount = torch.bincount(
                self.type_to_index[system.types], minlength=len(self.atomic_types)
            )
            counts.append(bincount.to(dtype=dtype))
        return torch.vstack(counts)

    def _compute_X_per_atom(
        self, systems: List[System], center_types: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the one-hot encoding of the atomic types for the atoms in the provided
        systems, but only for the specified center types.

        :param systems: List of systems to compute the one-hot encoding for.
        :param center_types: Tensor of atomic types to include in the one-hot encoding.

        :return: A tensor of shape ``(n_atoms, n_atomic_types)``, where each row
            corresponds to an atom in the systems and each column corresponds to an
            atomic type. The value is 1 if the atom's type matches the atomic type,
            and 0 otherwise.
        """
        dtype = systems[0].positions.dtype
        all_types = torch.concatenate([system.types for system in systems])
        all_types_as_indices = self.type_to_index[all_types]
        one_hot_encoding = torch.nn.functional.one_hot(
            all_types_as_indices, num_classes=len(center_types)
        )
        return one_hot_encoding.to(dtype)

    def _sync_device_dtype(self, device: torch.device, dtype: torch.dtype) -> None:
        # manually move the TensorMap dicts:

        self.atomic_types = self.atomic_types.to(device=device)
        self.type_to_index = self.type_to_index.to(device=device)
        self.XTX = {
            target_name: tm.to(device=device, dtype=dtype)
            for target_name, tm in self.XTX.items()
        }
        self.XTY = {
            target_name: tm.to(device=device, dtype=dtype)
            for target_name, tm in self.XTY.items()
        }
        self.weights = {
            target_name: tm.to(device=device, dtype=dtype)
            for target_name, tm in self.weights.items()
        }


def _include_key(key: LabelsEntry) -> bool:
    """
    Determines whether a block indexed by the input ``key`` should be included in the
    composition model.

    The rules are as follows:
        - If the key has a single name "_" (indicating a scalar), it is included.
        - If the key has names ["o3_lambda", "o3_sigma"] it is included if values are 0
          and 1 respectively (indicating an invariant block of a spherical target).
        - If the key has names ["o3_lambda", "o3_sigma", "n_centers"], it is included if
          values are 0, 1, 1 respectively (indicating an invariant block of a per-atom
          spherical target).

    :param key: The key to check.

    :return: Whether the key should be included in the composition model.
    """
    valid_key_names = [
        ["_"],  # scalar
        ["o3_lambda", "o3_sigma"],  # spherical
        ["o3_lambda", "o3_sigma", "n_centers"],  # spherical per-atom
    ]
    include_key = False

    if key.names == valid_key_names[0]:
        include_key = True

    elif key.names == valid_key_names[1]:
        if key["o3_lambda"] == 0 and key["o3_sigma"] == 1:
            include_key = True

    elif key.names == valid_key_names[2]:
        if key["o3_lambda"] == 0 and key["o3_sigma"] == 1 and key["n_centers"] == 1:
            include_key = True

    else:
        raise ValueError(
            f"key names {key.names} not in valid key names {valid_key_names}"
        )

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

    :param XTX_vals: Values of the XTX matrix.
    :param XTY_vals: Values of the XTY matrix.

    :return: The weights W, of shape (n_atomic_types, n_components, n_properties).
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


def _get_system_indices_and_labels(
    systems: List[System], device: torch.device
) -> Tuple[torch.Tensor, Labels]:
    system_indices = torch.concatenate(
        [
            torch.full(
                (len(system),),
                i_system,
                device=device,
            )
            for i_system, system in enumerate(systems)
        ],
    )

    sample_values = torch.stack(
        [
            system_indices,
            torch.concatenate(
                [
                    torch.arange(
                        len(system),
                        device=device,
                    )
                    for system in systems
                ],
            ),
        ],
        dim=1,
    )
    sample_labels = Labels(
        names=["system", "atom"],
        values=sample_values,
        assume_unique=True,
    )
    return system_indices, sample_labels


def _get_sample_labels(
    systems: List[System],
    sample_kinds: Dict[str, str],
    outputs: Dict[str, ModelOutput],
    device: torch.device,
) -> Dict[str, Labels]:
    """
    Returns the sample labels for per-atom and per-pair targets in a dict of labels
    indexed by the sample kind (either "per_atom", or "per_pair"). Per-atom labels are
    always returned, but per-pair is only returned if the sample kind of one of the
    outputs is "per_pair", otherwise an empty Labels is returned.

    :param systems: List of systems to compute the sample labels for.
    :param sample_kinds: Dict of target names to sample kinds.
    :param outputs: Dict of named outputs to compute.
    :param device: Device to put the labels on.
    :return: Dict of sample kinds to corresponding sample labels.
    """
    device = systems[0].positions.device
    atom_sample_labels = get_atom_sample_labels(systems, device)

    if any([k == "per_pair" for k in sample_kinds.values()]):
        pair_sample_labels_onsite = get_pair_sample_labels_onsite(
            systems, atom_sample_labels, device
        ).to(device=device)
    else:
        pair_sample_labels_onsite = Labels(["_"], torch.empty(0).reshape(-1, 1)).to(
            device=device
        )

    return {
        "per_structure": atom_sample_labels,
        "per_atom": atom_sample_labels,
        "per_pair": pair_sample_labels_onsite,
    }
