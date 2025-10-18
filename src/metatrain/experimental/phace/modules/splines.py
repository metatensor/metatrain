import numpy as np
import torch


def generate_splines(
    radial_basis,
    radial_basis_derivatives,
    max_index,
    cutoff_radius,
    requested_accuracy=1e-8,
):
    """Spline generator for tabulated radial integrals.

    Besides some self-explanatory parameters, this function takes as inputs two
    functions, namely radial_basis and radial_basis_derivatives. These must be
    able to calculate the radial basis functions by taking n, l, and r as their
    inputs, where n and l are integers and r is a numpy 1-D array that contains
    the spline points at which the radial basis function (or its derivative)
    needs to be evaluated. These functions should return a numpy 1-D array
    containing the values of the radial basis function (or its derivative)
    corresponding to the specified n and l, and evaluated at all points in the
    r array. If specified, n_spline_points determines how many spline points
    will be used for each splined radial basis function. Alternatively, the user
    can specify a requested accuracy. Spline points will be added until either
    the relative error or the absolute error fall below the requested accuracy on
    average across all radial basis functions.
    """

    def value_evaluator_2D(positions):
        values = []
        for index in range(max_index):
            value = radial_basis(index, np.array(positions))
            values.append(value)
        values = torch.tensor(np.array(values))
        values = values.T
        values = values.reshape(len(positions), max_index)
        return values

    def derivative_evaluator_2D(positions):
        derivatives = []
        for index in range(max_index):
            derivative = radial_basis_derivatives(index, np.array(positions))
            derivatives.append(derivative)
        derivatives = torch.tensor(np.array(derivatives))
        derivatives = derivatives.T
        derivatives = derivatives.reshape(len(positions), max_index)
        return derivatives

    dynamic_spliner = DynamicSpliner(
        0.0,
        cutoff_radius,
        value_evaluator_2D,
        derivative_evaluator_2D,
        requested_accuracy,
    )
    return dynamic_spliner


class DynamicSpliner(torch.nn.Module):
    def __init__(
        self, start, stop, values_fn, derivatives_fn, requested_accuracy
    ) -> None:
        super().__init__()

        self.start = start
        self.stop = stop

        # initialize spline with 11 points; the spline calculation
        # is performed in double precision
        positions = torch.linspace(start, stop, 11, dtype=torch.float64)
        self.register_buffer("spline_positions", positions)
        self.register_buffer("spline_values", values_fn(positions))
        self.register_buffer("spline_derivatives", derivatives_fn(positions))

        self.number_of_custom_dimensions = (
            len(self.spline_values.shape) - 1  # type: ignore
        )

        while True:
            n_intermediate_positions = len(self.spline_positions) - 1  # type: ignore

            if n_intermediate_positions >= 50000:
                raise ValueError(
                    "Maximum number of spline points reached. \
                    There might be a problem with the functions to be splined"
                )

            half_step = (
                self.spline_positions[1] - self.spline_positions[0]  # type: ignore
            ) / 2
            intermediate_positions = torch.linspace(
                self.start + half_step,
                self.stop - half_step,
                n_intermediate_positions,
                dtype=torch.float64,
            )

            estimated_values = self.compute(intermediate_positions)
            new_values = values_fn(intermediate_positions)

            mean_absolute_error = torch.mean(torch.abs(estimated_values - new_values))
            mean_relative_error = torch.mean(
                torch.abs((estimated_values - new_values) / new_values)
            )

            if (
                mean_absolute_error < requested_accuracy
                or mean_relative_error < requested_accuracy
            ):
                break

            new_derivatives = derivatives_fn(intermediate_positions)

            concatenated_positions = torch.cat(
                [self.spline_positions, intermediate_positions],
                dim=0,  # type: ignore
            )
            concatenated_values = torch.cat(
                [self.spline_values, new_values],
                dim=0,  # type: ignore
            )
            concatenated_derivatives = torch.cat(
                [self.spline_derivatives, new_derivatives],
                dim=0,  # type: ignore
            )

            sort_indices = torch.argsort(concatenated_positions, dim=0)

            self.spline_positions = concatenated_positions[sort_indices]
            self.spline_values = concatenated_values[sort_indices]
            self.spline_derivatives = concatenated_derivatives[sort_indices]

        self.spline_positions = self.spline_positions.to(torch.get_default_dtype())
        self.spline_values = self.spline_values.to(torch.get_default_dtype())
        self.spline_derivatives = self.spline_derivatives.to(torch.get_default_dtype())

    def compute(self, positions):
        x = positions
        delta_x = self.spline_positions[1] - self.spline_positions[0]
        n = (torch.floor(x / delta_x)).to(dtype=torch.long)

        t = (x - n * delta_x) / delta_x
        t_2 = t**2
        t_3 = t**3

        h00 = 2.0 * t_3 - 3.0 * t_2 + 1.0
        h10 = t_3 - 2.0 * t_2 + t
        h01 = -2.0 * t_3 + 3.0 * t_2
        h11 = t_3 - t_2

        p_k = torch.index_select(self.spline_values, dim=0, index=n)
        p_k_1 = torch.index_select(self.spline_values, dim=0, index=n + 1)

        m_k = torch.index_select(self.spline_derivatives, dim=0, index=n)
        m_k_1 = torch.index_select(self.spline_derivatives, dim=0, index=n + 1)

        new_shape = (-1,) + (1,) * self.number_of_custom_dimensions
        h00 = h00.reshape(new_shape)
        h10 = h10.reshape(new_shape)
        h01 = h01.reshape(new_shape)
        h11 = h11.reshape(new_shape)

        interpolated_values = (
            h00 * p_k + h10 * delta_x * m_k + h01 * p_k_1 + h11 * delta_x * m_k_1
        )

        return interpolated_values
