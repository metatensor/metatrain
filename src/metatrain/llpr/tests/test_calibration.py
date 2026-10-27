import pytest
import torch

from metatrain.llpr.calibration import HeteroscedasticNLLCalibrator


@pytest.mark.parametrize("alpha", [0.3, 2.5])
def test_heteroscedastic_calibration_recovers_alpha(alpha):
    """With residuals drawn from N(0, s + alpha^2 e), the calibrated multiplier is
    close to alpha."""
    torch.manual_seed(0)

    n_samples = 200_000
    aleatoric = torch.rand(n_samples, 1, dtype=torch.float64) + 0.1
    epistemic = 2.0 * torch.rand(n_samples, 1, dtype=torch.float64)
    residuals = torch.randn(n_samples, 1, dtype=torch.float64) * torch.sqrt(
        aleatoric + alpha**2 * epistemic
    )

    calibrator = HeteroscedasticNLLCalibrator()
    # accumulated over several batches
    for batch in torch.split(torch.arange(n_samples), 30_000):
        calibrator.update(
            uncertainty_name="u",
            residuals=residuals[batch],
            aleatoric_variances=aleatoric[batch],
            epistemic_variances=epistemic[batch],
        )

    multipliers = calibrator.finalize()

    assert multipliers["u"].shape == (1,)
    torch.testing.assert_close(
        multipliers["u"], torch.tensor([alpha], dtype=torch.float64), rtol=0.03, atol=0
    )


def test_heteroscedastic_calibration_without_epistemic_error():
    """If the aleatoric variance already explains the residuals, alpha goes to 0."""
    torch.manual_seed(0)
    aleatoric = torch.rand(10_000, 1, dtype=torch.float64) + 0.1
    residuals = torch.randn(10_000, 1, dtype=torch.float64) * 0.5 * aleatoric.sqrt()

    calibrator = HeteroscedasticNLLCalibrator()
    calibrator.update(
        uncertainty_name="u",
        residuals=residuals,
        aleatoric_variances=aleatoric,
        epistemic_variances=torch.rand(10_000, 1, dtype=torch.float64),
    )
    assert calibrator.finalize()["u"].item() < 1e-6


def test_heteroscedastic_calibration_ignores_nan_residuals():
    """Targets can be NaN (they are masked in the loss), which must not affect the
    result."""
    torch.manual_seed(0)

    alpha = 0.7
    n_samples = 100_000
    aleatoric = torch.rand(n_samples, 1, dtype=torch.float64) + 0.1
    epistemic = 2.0 * torch.rand(n_samples, 1, dtype=torch.float64)
    residuals = torch.randn(n_samples, 1, dtype=torch.float64) * torch.sqrt(
        aleatoric + alpha**2 * epistemic
    )
    residuals[::10] = torch.nan

    calibrator = HeteroscedasticNLLCalibrator()
    calibrator.update(
        uncertainty_name="u",
        residuals=residuals,
        aleatoric_variances=aleatoric,
        epistemic_variances=epistemic,
    )

    torch.testing.assert_close(
        calibrator.finalize()["u"],
        torch.tensor([alpha], dtype=torch.float64),
        rtol=0.05,
        atol=0,
    )
