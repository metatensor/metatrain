from typing import Optional

import torch


def cutoff_func_bump(
    values: torch.Tensor, cutoff: torch.Tensor, width: float, eps: float = 1e-6
) -> torch.Tensor:
    """
    Bump cutoff function.

    :param values: Distances at which to evaluate the cutoff function.
    :param cutoff: Cutoff radius for each node.
    :param width: Width of the cutoff region.
    :param eps: Avoid computing at values too close to the edges.
    :return: Values of the cutoff function at the specified distances.
    """

    scaled_values = (values - (cutoff - width)) / width

    mask_smaller = scaled_values <= 0.0
    mask_active = (scaled_values > 0.0) & (scaled_values < 1.0)

    f = torch.zeros_like(scaled_values)
    f[mask_active] = 0.5 * (
        1 + torch.tanh(1 / torch.tan(torch.pi * scaled_values[mask_active]))
    )
    f[mask_smaller] = 1.0

    return f


def cutoff_func_cosine(
    values: torch.Tensor, cutoff: torch.Tensor, width: float
) -> torch.Tensor:
    """
    Cosine cutoff function.

    :param values: Distances at which to evaluate the cutoff function.
    :param cutoff: Cutoff radius for each node.
    :param width: Width of the cutoff region.
    :return: Values of the cutoff function at the specified distances.
    """

    scaled_values = (values - (cutoff - width)) / width

    mask_smaller = scaled_values <= 0.0
    mask_active = (scaled_values > 0.0) & (scaled_values < 1.0)

    f = torch.zeros_like(scaled_values)

    f[mask_active] = 0.5 + 0.5 * torch.cos(torch.pi * scaled_values[mask_active])
    f[mask_smaller] = 1.0
    return f


class DecomposedSiLU(torch.nn.Module):
    """SiLU activation implemented as ``x * sigmoid(x)``.

    Unlike ``torch.nn.SiLU``, this decomposes into primitive ops so that
    ``make_fx`` produces a backward graph without ``silu_backward`` nodes.
    This is needed for ``torch.compile(inductor)`` to differentiate through
    the inlined backward when using the FX compilation path for force training.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class DecomposedRMSNorm(torch.nn.Module):
    """RMSNorm using only primitive ops.

    Unlike ``torch.nn.RMSNorm``, this avoids ``_fused_rms_norm_backward``
    whose second derivative is not implemented in older PyTorch versions.
    This allows ``torch.compile(inductor)`` to differentiate through the
    backward graph when training on forces (which requires double backward
    through the FX-decomposed force computation).

    :param normalized_shape: Input shape from an expected input.
    :param eps: Value added to the denominator for numerical stability.
    """

    def __init__(self, normalized_shape: int, eps: Optional[float] = None) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps if eps is not None else torch.finfo(torch.float32).eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


def replace_silu_modules(module: torch.nn.Module) -> None:
    """Replace all ``torch.nn.SiLU`` instances with :class:`DecomposedSiLU`.

    Recurses through the module tree, including inside ``nn.Sequential``.

    :param module: The module to recursively modify in-place.
    """
    for name, child in module.named_children():
        if isinstance(child, torch.nn.SiLU):
            setattr(module, name, DecomposedSiLU())
        elif isinstance(child, torch.nn.Sequential):
            for i, layer in enumerate(child):
                if isinstance(layer, torch.nn.SiLU):
                    child[i] = DecomposedSiLU()
                else:
                    replace_silu_modules(layer)
        else:
            replace_silu_modules(child)


def replace_rmsnorm_modules(module: torch.nn.Module) -> None:
    """Replace all ``torch.nn.RMSNorm`` with :class:`DecomposedRMSNorm`.

    Copies the learned weight and eps from each original module.

    :param module: The module to recursively modify in-place.
    """
    for name, child in module.named_children():
        if isinstance(child, torch.nn.RMSNorm):
            decomposed = DecomposedRMSNorm(child.normalized_shape[0], eps=child.eps)
            decomposed.weight = child.weight
            setattr(module, name, decomposed)
        else:
            replace_rmsnorm_modules(child)


class DummyModule(torch.nn.Module):
    """Dummy torch module to make torchscript happy.
    This model should never be run"""

    def __init__(self) -> None:
        super(DummyModule, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("This model should never be run")
