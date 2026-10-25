import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from .cg import cg_couple, cg_uncouple


def couple_tensor_blocks(
    tmap: TensorMap,
    cg_coeffs: TensorMap,
) -> TensorMap:
    """
    Takes a tensormap in the uncoupled representation and converts it
    to the coupled representation.

    :param tmap: TensorMap in the uncoupled representation, with keys
        ``["o3_lambda_1", "o3_lambda_2", "o3_sigma_1", "o3_sigma_2"]``
        and properties ``["n_1", "n_2"]``.
    :param cg_coeffs:
        Precomputed Clebsch-Gordan coefficients, as returned by
        :func:`calculate_cg_coefficients` from ``featomic.torch``.

    :return: TensorMap in the coupled representation.
    """
    # Find out in which layout the provided Clebsch-Gordan coefficients
    # are stored.
    if cg_coeffs.block(0).samples.names == ["_"]:
        cg_backend = "python-dense"
    elif cg_coeffs.block(0).samples.names == ["m1", "m2", "mu"]:
        cg_backend = "python-sparse"
    else:
        raise ValueError(
            "Can't recognize the format of the provided Clebsch-Gordan coefficients."
        )

    # Check key names
    assert tmap.keys.names[:4] == [
        "o3_lambda_1",
        "o3_lambda_2",
        "o3_sigma_1",
        "o3_sigma_2",
    ]

    other_keys = tmap.keys.names[4:]

    # Couple each block in turn
    key_values: list[tuple[int, int, int, int]] = []
    other_key_values = []
    blocks: list[TensorBlock] = []
    for key, block in tmap.items():
        o3_lambda_1, o3_lambda_2 = int(key["o3_lambda_1"]), int(key["o3_lambda_2"])

        allowed_o3_lambdas = torch.arange(
            abs(o3_lambda_1 - o3_lambda_2),
            o3_lambda_1 + o3_lambda_2 + 1,
        )
        allowed_o3_lambdas = [int(o3_lambda) for o3_lambda in allowed_o3_lambdas]
        if block.values.shape[0] > 0:
            coupled_arrays = cg_couple(
                block.values, allowed_o3_lambdas, cg_coeffs, cg_backend
            )
        else:
            # With no samples, the cg_couple function fails, so we
            # create empty arrays for each allowed o3_lambda instead.
            coupled_arrays = [
                torch.empty(
                    (0, 2 * o3_lambda + 1, block.values.shape[-1]),
                    dtype=block.values.dtype,
                    device=block.values.device,
                )
                for o3_lambda in allowed_o3_lambdas
            ]

        for o3_lambda, coupled_array in zip(
            allowed_o3_lambdas, coupled_arrays, strict=True
        ):
            # Check samples and properties dimensions have been preserved
            assert coupled_array.shape[0] == block.values.shape[0]
            assert coupled_array.shape[-1] == block.values.shape[-1]

            o3_sigma = (-1) ** (o3_lambda + o3_lambda_1 + o3_lambda_2)
            new_key = (o3_lambda, int(o3_sigma), o3_lambda_1, o3_lambda_2)
            key_values.append(new_key)
            other_key_values.append(key.values[4:])

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
        dtype=tmap.keys.values.dtype,
        device=tmap.keys.values.device,
    )

    if len(other_key_values) > 0:
        other_keys_values = torch.stack(other_key_values, dim=0)
        keys_values = torch.concat(
            [keys_values[:, :2], other_keys_values, keys_values[:, 2:4]], dim=1
        )

    keys_names = ["o3_lambda", "o3_sigma"] + other_keys + ["l_1", "l_2"]
    # Build the new TensorMap and move the coupled l indices to properties
    tensor_coupled = TensorMap(
        Labels(keys_names, keys_values),
        blocks,
    )
    return tensor_coupled.keys_to_properties(["l_1", "l_2"])


def uncouple_tensor_blocks(
    tmap: TensorMap,
    cg_coeffs: TensorMap,
) -> TensorMap:
    """
    Takes a tensormap in the coupled representation and converts it
    to the uncoupled representation.

    :param tmap: TensorMap in the coupled representation, with keys
        ``["o3_lambda", "o3_sigma"]`` and properties
        ``["l_1", "l_2", "n_1", "n_2"]``.
    :param cg_coeffs: Precomputed Clebsch-Gordan coefficients, as returned by
        :func:`calculate_cg_coefficients` from ``featomic.torch``.
        These coefficients must have been computed with the
        ``python-dense`` backend.

    :return: TensorMap in the uncoupled representation.
    """
    # Find out in which layout the provided Clebsch-Gordan coefficients
    # are stored.
    if cg_coeffs.block(0).samples.names == ["_"]:
        cg_backend = "python-dense"
    elif cg_coeffs.block(0).samples.names == ["m1", "m2", "mu"]:
        cg_backend = "python-sparse"
    else:
        raise ValueError(
            "Can't recognize the format of the provided Clebsch-Gordan coefficients."
        )

    # Check key names
    assert tmap.keys.names[:2] == ["o3_lambda", "o3_sigma"]

    key_names = ["o3_lambda_1", "o3_lambda_2", "o3_sigma_1", "o3_sigma_2"]
    extra_key_names = tmap.keys.names[2:]
    key_names += extra_key_names
    key_names += ["n_1", "n_2"]

    # Uncouple each block in turn. Dict keys are encoded as a single string (rather
    # than the variable-length tuple used previously) since TorchScript requires
    # concrete, fixed-arity tuple types (no `Tuple[int, ...]`) and does not support
    # tuples as Dict keys at all - str/int/float/bool/Tensor only.
    sample_labels: dict[str, Labels] = {}
    key_ints: dict[str, list[int]] = {}
    values: dict[str, torch.Tensor] = {}
    key_order: list[str] = []
    for k, b in tmap.items():
        b_samples = b.samples
        b_values = b.values

        o3_lambda = int(k["o3_lambda"])

        key_values: list[int] = k.values.tolist()
        extra_key_values = key_values[2:]

        properties_values: list[list[int]] = b.properties.values.to(
            torch.int64
        ).tolist()
        for ip, (l1, l2, n1, n2) in enumerate(properties_values):
            key_values = [l1, l2, 1, 1] + extra_key_values + [n1, n2]
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

            v = cg_uncouple(
                [b_values[..., ip][..., None]],  # shape (n_samples, 2*o3_lambda + 1, 1)
                l1,
                l2,
                [o3_lambda],
                cg_coeffs,
                cg_backend=cg_backend,
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
        all_key_values.append(key_ints[key])

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
