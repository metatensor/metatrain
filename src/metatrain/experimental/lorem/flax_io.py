"""Transfer Flax / lorem-jax parameter trees into a PyTorch LOREM.

Flax checkpoints are ``model/model.msgpack`` (plus ``baseline.yaml``). This
module does **not** depend on JAX: reading msgpack uses ``flax`` when
installed. Applying weights only needs a nested dict of array-like leaves.

Flax ``Dense`` kernels are ``(in, out)``; ``torch.nn.Linear.weight`` is
``(out, in)``. Embeddings and LayerNorm match when shapes agree.

TensorDense / e3x ``tensor/kernel`` leaves are skipped unless a caller
supplies an explicit ``name_map``. Radial bases and SH are parameter-free.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch


def read_flax_msgpack(path: str | Path) -> Dict[str, Any]:
    """Load a Flax msgpack parameter tree.

    :param path: ``model.msgpack`` or a checkpoint folder containing
        ``model/model.msgpack``.
    """
    path = Path(path)
    if path.is_dir():
        candidate = path / "model" / "model.msgpack"
        if not candidate.is_file():
            candidate = path / "model.msgpack"
        path = candidate
    try:
        from flax.serialization import msgpack_restore

        return msgpack_restore(path.read_bytes())
    except ImportError as err:
        raise ImportError(
            "Reading flax checkpoints needs flax installed in a JAX env. "
            "Do not install it into the shared metatrain venv."
        ) from err


def unwrap_params(tree: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the ``params`` subtree if present."""
    if "params" in tree and isinstance(tree["params"], Mapping):
        return tree["params"]
    return tree


def flatten_flax_tree(
    tree: Mapping[str, Any], prefix: str = ""
) -> Dict[str, torch.Tensor]:
    """Flatten a nested Flax tree to ``path/to/leaf`` → tensor."""
    if prefix == "":
        tree = unwrap_params(tree)
    flat: Dict[str, torch.Tensor] = {}
    for key, value in tree.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_flax_tree(value, name))
        else:
            flat[name] = torch.as_tensor(value)
    return flat


def apply_flax_params(
    module: torch.nn.Module,
    flax_tree: Mapping[str, Any],
    name_map: Optional[Mapping[str, str]] = None,
) -> List[str]:
    """Copy compatible Linear / Embedding / LayerNorm leaves into ``module``.

    :param module: Typically a ``LOREM`` instance.
    :param flax_tree: Flax parameter dict (with or without a ``params`` wrapper).
    :param name_map: Optional ``torch.name → flax.flat.name`` overrides.
    :return: Torch parameter names that were overwritten.
    """
    flax_flat = flatten_flax_tree(unwrap_params(flax_tree))
    flax_by_shape: Dict[Tuple[int, ...], List[str]] = {}
    for flax_name, tensor in flax_flat.items():
        flax_by_shape.setdefault(tuple(tensor.shape), []).append(flax_name)

    used: set[str] = set()
    transferred: List[str] = []
    explicit = dict(name_map) if name_map is not None else {}

    with torch.no_grad():
        for torch_name, parameter in module.named_parameters():
            source = None
            if torch_name in explicit:
                flax_name = explicit[torch_name]
                if flax_name not in flax_flat:
                    raise KeyError(
                        f"name_map target missing from flax tree: {flax_name}"
                    )
                source = flax_flat[flax_name]
                used.add(flax_name)
            else:
                source = _match_by_layout(
                    torch_name, parameter, flax_flat, flax_by_shape, used
                )
            if source is None:
                continue
            if source.shape != parameter.shape:
                raise ValueError(
                    f"shape mismatch for {torch_name}: "
                    f"torch {tuple(parameter.shape)} vs flax {tuple(source.shape)}"
                )
            parameter.copy_(source.to(device=parameter.device, dtype=parameter.dtype))
            transferred.append(torch_name)
    return transferred


def _match_by_layout(
    torch_name: str,
    parameter: torch.Tensor,
    flax_flat: Mapping[str, torch.Tensor],
    flax_by_shape: Mapping[Tuple[int, ...], List[str]],
    used: set[str],
) -> Optional[torch.Tensor]:
    is_weight = torch_name == "weight" or torch_name.endswith(".weight")
    if is_weight and parameter.ndim == 2:
        # Linear: flax kernel (in, out) → torch (out, in)
        kernel_shape = (parameter.shape[1], parameter.shape[0])
        candidates = [
            name
            for name in flax_by_shape.get(kernel_shape, [])
            if name not in used and name.endswith("kernel")
        ]
        if len(candidates) == 1:
            used.add(candidates[0])
            return flax_flat[candidates[0]].transpose(0, 1)
        # Embedding: flax (n, d) matches torch weight
        embed_candidates = [
            name
            for name in flax_by_shape.get(tuple(parameter.shape), [])
            if name not in used and "embedding" in name
        ]
        if len(embed_candidates) == 1:
            used.add(embed_candidates[0])
            return flax_flat[embed_candidates[0]]
        return None

    if (torch_name == "bias" or torch_name.endswith(".bias")) and parameter.ndim == 1:
        candidates = [
            name
            for name in flax_by_shape.get(tuple(parameter.shape), [])
            if name not in used and name.endswith("bias")
        ]
        if len(candidates) == 1:
            used.add(candidates[0])
            return flax_flat[candidates[0]]
        return None

    if ".weight" in torch_name and parameter.ndim == 1:
        # LayerNorm scale
        candidates = [
            name
            for name in flax_by_shape.get(tuple(parameter.shape), [])
            if name not in used and name.endswith("scale")
        ]
        if len(candidates) == 1:
            used.add(candidates[0])
            return flax_flat[candidates[0]]
    return None


def transferred_summary(names: Iterable[str]) -> str:
    """One-line report for logs."""
    names = list(names)
    return f"transferred {len(names)} flax leaves into torch"
