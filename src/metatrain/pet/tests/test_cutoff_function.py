import torch

from metatrain.pet.modules.utilities import cutoff_func_bump


def _cutoff_func_bump_tanh(
    values: torch.Tensor, cutoff: torch.Tensor, width: float
) -> torch.Tensor:
    """Previous form of the bump cutoff, ``0.5 * (1 + tanh(y))``."""
    scaled_values = (values - (cutoff - width)) / width
    clamped = scaled_values.clamp(1e-6, 1.0 - 1e-6)
    return 0.5 * (1 + torch.tanh(1 / torch.tan(torch.pi * clamped)))


def _evaluate(function, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # with cutoff = width = 1, the scaled distance is exactly ``x``
    cutoff = torch.tensor(1.0, dtype=dtype)
    return function(x.to(dtype), cutoff, 1.0)


def test_bump_matches_tanh_form_fp64():
    """sigmoid(2y) is the same function as 0.5 * (1 + tanh(y))."""
    x = torch.cat(
        [
            torch.linspace(0.0, 1.0, 100_001, dtype=torch.float64),
            torch.tensor([-0.5, 1.5], dtype=torch.float64),
        ]
    )
    new = _evaluate(cutoff_func_bump, x, torch.float64)
    old = _evaluate(_cutoff_func_bump_tanh, x, torch.float64)

    torch.testing.assert_close(new, old, rtol=0.0, atol=1e-15)
    # the tanh form itself loses relative precision as f_c -> 0 (absolute error
    # ~1e-16), so only compare relative errors where f_c is not small
    large = old > 1e-3
    torch.testing.assert_close(new[large], old[large], rtol=1e-12, atol=0.0)

    torch.testing.assert_close(new[x <= 0], torch.ones_like(new[x <= 0]))
    torch.testing.assert_close(new[x >= 1], torch.zeros_like(new[x >= 1]))


def test_bump_tail_precision_fp32():
    """In fp32, the cutoff keeps relative precision in the tail (x -> 1).

    ``1 + tanh(y)`` cancels for ``y < -8.7``, so the previous form could only take
    multiples of 2^-25 near zero and then underflowed to exactly zero. Since PET uses
    ``log(f_c)`` as an attention bias, this made fp32 energies a staircase near the
    cutoff.
    """
    x = torch.linspace(0.5, 1.0 - 1e-6, 200_001, dtype=torch.float32)
    reference = _evaluate(cutoff_func_bump, x, torch.float64)
    new = _evaluate(cutoff_func_bump, x, torch.float32).to(torch.float64)
    old = _evaluate(_cutoff_func_bump_tanh, x, torch.float32).to(torch.float64)

    # no loss of absolute accuracy anywhere
    torch.testing.assert_close(new, reference, rtol=0.0, atol=1e-6)

    tail = reference > 1e-30
    # the previous form underflows to zero and is quantised to a few values
    assert (old[tail] == 0.0).any()
    assert torch.unique(old[tail & (reference < 1e-6)]).numel() < 64

    # the new form stays positive and accurate in log space down to f_c ~ 1e-30.
    # The remaining error comes from tan(pi * x) in fp32 (measured ~5e-4)
    assert (new[tail] > 0.0).all()
    log_error = (new[tail].log() - reference[tail].log()).abs()
    assert log_error.max() < 2e-3
    assert torch.unique(new[tail & (reference < 1e-6)]).numel() > 10_000
