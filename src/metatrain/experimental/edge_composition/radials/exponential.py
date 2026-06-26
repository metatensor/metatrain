import torch


class Exponential(torch.nn.Module):
    def __init__(self, n_props, deg_poly=4, deg_exp_poly=4):
        super().__init__()

        self.coeffs = torch.nn.Parameter(
            torch.zeros((n_props, deg_exp_poly, deg_poly + 1))
        )
        self.poly_coeffs = torch.nn.Parameter(torch.zeros((n_props, deg_poly + 1)))

    def poly_eval(self, coeffs, x):
        return (
            coeffs[:, -4] * x**4
            + coeffs[:, -3] * x**3
            + coeffs[:, -2] * x**2
            + coeffs[:, -1] * x
        )

    def forward(self, x):
        x = x.reshape(-1, 1)
        return (
            self.poly_coeffs[:, -5]
            * x**4
            * torch.exp(self.poly_eval(self.coeffs[:, :, -5], x))
            + self.poly_coeffs[:, -4]
            * x**3
            * torch.exp(self.poly_eval(self.coeffs[:, :, -4], x))
            + self.poly_coeffs[:, -3]
            * x**2
            * torch.exp(self.poly_eval(self.coeffs[:, :, -3], x))
            + self.poly_coeffs[:, -2]
            * x
            * torch.exp(self.poly_eval(self.coeffs[:, :, -2], x))
            + self.poly_coeffs[:, -1]
            * torch.exp(self.poly_eval(self.coeffs[:, :, -1], x))
        )
