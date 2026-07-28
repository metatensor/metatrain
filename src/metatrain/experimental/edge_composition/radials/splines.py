import torch
import torchcurves as tc


class BSpline(torch.nn.Module):
    def __init__(self, n_props, x_min=0, x_max=12, N=120):
        super().__init__()

        self.curve = tc.BSplineCurve(
            num_curves=1, dim=n_props, knots_config=N, parameter_range=(x_min, x_max)
        )

    def forward(self, x):
        return self.curve(x.reshape(-1, 1)).squeeze(1)
