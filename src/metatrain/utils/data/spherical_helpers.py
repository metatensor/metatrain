from typing import Optional

import metatensor.torch as mts
import torch
from featomic.torch.clebsch_gordan._coefficients import (
    calculate_cg_coefficients,
    cg_couple,
)
from metatensor.torch import Labels, TensorBlock, TensorMap


def couple_tensor_blocks(
    tensor: TensorMap,
    cg_coeffs: Optional[TensorMap] = None,
) -> TensorMap:
    """
    Takes the uncoupled block representation of a per-pair target property on an
    atom-centered basis and couples the blocks.

    Copied from elearn
    """
    # Compute CG coefficients if not passed
    cg_backend = "python-sparse"  # if backend == "numpy" else "python-dense"
    if cg_coeffs is None:
        if torch.jit.is_scripting():
            raise ValueError(
                "Cannot compute CG coefficients inside torchscript. "
                "Please provide them as an argument."
            )
        else:
            max_angular = int(
                torch.max(
                    torch.tensor(
                        [
                            torch.max(tensor.keys.column("o3_lambda_1")),
                            torch.max(tensor.keys.column("o3_lambda_2")),
                        ],
                        dtype=torch.int32,
                    )
                )
            )
            cg_coeffs = calculate_cg_coefficients(
                max_angular * 2,
                cg_backend=cg_backend,
                arrays_backend="torch",
                dtype=tensor[0].values.dtype,
                device=tensor[0].values.device,
            )

    # Check key names
    assert tensor.keys.names[:4] == [
        "o3_lambda_1",
        "o3_lambda_2",
        "o3_sigma_1",
        "o3_sigma_2",
    ]

    other_keys = tensor.keys.names[4:]

    # Couple each block in turn
    key_values: list[tuple[int, int, int, int]] = []
    blocks: list[TensorBlock] = []
    for key, block in tensor.items():
        o3_lambda_1, o3_lambda_2 = int(key["o3_lambda_1"]), int(key["o3_lambda_2"])

        allowed_o3_lambdas = torch.arange(
            abs(o3_lambda_1 - o3_lambda_2),
            o3_lambda_1 + o3_lambda_2 + 1,
        )
        allowed_o3_lambdas = [int(o3_lambda) for o3_lambda in allowed_o3_lambdas]
        coupled_arrays = cg_couple(
            block.values, allowed_o3_lambdas, cg_coeffs, cg_backend
        )

        for o3_lambda, coupled_array in zip(allowed_o3_lambdas, coupled_arrays):
            # Check samples and properties dimensions have been preserved
            assert coupled_array.shape[0] == block.values.shape[0]
            assert coupled_array.shape[-1] == block.values.shape[-1]

            o3_sigma = (-1) ** (o3_lambda + o3_lambda_1 + o3_lambda_2)
            new_key = (o3_lambda, int(o3_sigma), o3_lambda_1, o3_lambda_2)
            key_values.append(new_key)

            new_block = TensorBlock(
                samples=block.samples,
                components=[
                    Labels(
                        ["o3_mu"],
                        torch.arange(
                            -o3_lambda,
                            o3_lambda + 1,
                        ).reshape(-1, 1),
                    ),
                ],
                properties=block.properties,
                values=coupled_array,
            )
            blocks.append(new_block)

    keys_values = torch.tensor(
        key_values,
        dtype=tensor.keys.values.dtype,
        device=tensor.keys.values.device,
    )

    keys_values = torch.stack([keys_values[:, :2], tensor.keys.values[:, 4:], tensor.keys.values[:, 4:]], dim=1)

    keys_names = ["o3_lambda", "o3_sigma"] + other_keys + ["l_1", "l_2"]
    # Build the new TensorMap and move the coupled l indices to properties
    tensor_coupled = TensorMap(
        Labels(keys_names, keys_values),
        blocks,
    )
    return tensor_coupled.keys_to_properties(["l_1", "l_2"])


def _uncouple_tensor_blocks(
    tensor: TensorMap,
    cg_coeffs: TensorMap,
) -> TensorMap:
    """
    Takes the coupled block representation of a per-pair target property on an
    atom-centered basis and uncouples the blocks.

    Copied from elearn
    """

    # Check key names
    assert tensor.keys.names[:2] == ["o3_lambda", "o3_sigma"]
    is_symmetrized = "s2_pi" in tensor.keys.names

    key_names = ["o3_lambda_1", "o3_lambda_2"]
    if is_symmetrized:
        key_names += ["s2_pi"]
    key_names += ["first_atom_type", "second_atom_type", "n_1", "n_2"]

    # Uncouple each block in turn. Dict keys are encoded as a single string (rather
    # than the variable-length tuple used previously) since TorchScript requires
    # concrete, fixed-arity tuple types (no `Tuple[int, ...]`) and does not support
    # tuples as Dict keys at all - str/int/float/bool/Tensor only.
    sample_labels: dict[str, Labels] = {}
    key_ints: dict[str, list[int]] = {}
    values: dict[str, torch.Tensor] = {}
    key_order: list[str] = []
    for k, b in tensor.items():
        b_samples = b.samples
        b_values = b.values

        o3_lambda = int(k["o3_lambda"])
        Z1 = int(k["first_atom_type"])
        Z2 = int(k["second_atom_type"])
        # Unused placeholder (0) when the tensor isn't symmetrized: dropped again
        # below, when building the final (correctly-shaped) keys Labels.
        s2_pi = int(k["s2_pi"]) if is_symmetrized else 0

        properties_values: list[list[int]] = b.properties.values.to(
            torch.int64
        ).tolist()
        for ip, (l1, l2, n1, n2) in enumerate(properties_values):
            o3_sigma = (-1) ** (l1 + l2 + o3_lambda)

            key_values = [l1, l2, s2_pi, Z1, Z2, n1, n2]
            key = "_".join([str(v) for v in key_values])
            if key not in sample_labels:
                key_order.append(key)
                sample_labels[key] = b_samples
                key_ints[key] = key_values
                values[key] = torch.zeros(
                    (b_values.shape[0], 2 * l1 + 1, 2 * l2 + 1, 1),
                    dtype=b_values.dtype,
                    device=b_values.device,
                )

            C = (
                o3_sigma
                * cg_coeffs.block({"l1": l1, "l2": l2, "lambda": o3_lambda}).values
            )
            # TODO: fix for the case of sparse cg coeffs
            C = C.reshape(2 * l1 + 1, 2 * l2 + 1, 2 * o3_lambda + 1)

            v = torch.einsum(
                "mnM,SM->Smn",
                C,
                b_values[..., ip],
            )
            values[key].add_(v.reshape(v.shape[0], v.shape[1], v.shape[2], 1))

    uncoupled_blocks: list[TensorBlock] = []
    all_key_values: list[list[int]] = []
    for key in key_order:
        v = values[key]
        l1, l2 = key_ints[key][0], key_ints[key][1]
        uncoupled_blocks.append(
            TensorBlock(
                samples=sample_labels[key],
                components=[
                    Labels(
                        ["o3_mu_1"],
                        torch.arange(
                            -l1, l1 + 1, dtype=torch.int64, device=v.device
                        ).unsqueeze(-1),
                    ),
                    Labels(
                        ["o3_mu_2"],
                        torch.arange(
                            -l2, l2 + 1, dtype=torch.int64, device=v.device
                        ).unsqueeze(-1),
                    ),
                ],
                properties=Labels(["_"], torch.tensor([[0]], device=v.device)),
                values=v,
            )
        )
        # (l1, l2, s2_pi, Z1, Z2, n1, n2) -> drop the s2_pi placeholder unless the
        # tensor is actually symmetrized, matching `key_names` above.
        if is_symmetrized:
            all_key_values.append(key_ints[key])
        else:
            kv = key_ints[key]
            all_key_values.append([kv[0], kv[1], kv[3], kv[4], kv[5], kv[6]])

    tensor_uncoupled = mts.remove_dimension(
        TensorMap(
            Labels(
                key_names,
                torch.tensor(all_key_values, device=uncoupled_blocks[0].values.device),
            ),
            uncoupled_blocks,
        ).keys_to_properties(["n_1", "n_2"]),
        "properties",
        "_",
    )

    return tensor_uncoupled


def uncouple_tensor_blocks(
    tensor: TensorMap,
    cg_coeffs: TensorMap,
) -> TensorMap:
    """_uncouple_tensor_blocks function does not
    add the o3_sigma names in the keys. This little
    wrapper adds them."""
    coupled = _uncouple_tensor_blocks(tensor, cg_coeffs)

    keys_names = (
        coupled.keys.names[:2] + ["o3_sigma_1", "o3_sigma_2"] + coupled.keys.names[2:]
    )
    keys_values = torch.concatenate(
        [
            coupled.keys.values[:, :2],
            torch.ones(
                (coupled.keys.values.shape[0], 2),
                dtype=coupled.keys.values.dtype,
                device=coupled.keys.values.device,
            ),
            coupled.keys.values[:, 2:],
        ],
        dim=1,
    )

    keys = Labels(names=keys_names, values=keys_values)

    return TensorMap(
        keys=keys,
        blocks=[block.copy(deep=False) for block in coupled.blocks()],
    )
