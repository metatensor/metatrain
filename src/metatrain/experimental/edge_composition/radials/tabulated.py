import torch

class Tabulated(torch.nn.Module):
    def __init__(self, n_props, x_min=0, x_max=12, dx=0.1):
        super().__init__()

        x_count = int((x_max - x_min) / dx) + 1
        self.x = torch.linspace(x_min, x_max, x_count)

        self.radial_tables = torch.nn.Parameter(torch.zeros((n_props, x_count)))

    def forward(self, x):
        # Clamp x to the range of the table
        x = torch.clamp(x, self.x[0], self.x[-1])

        # Interpolate the radial tables at the given x values
        x_indices = (x - self.x[0]) / (self.x[1] - self.x[0])
        x_indices_floor = torch.floor(x_indices).long()
        x_indices_ceil = torch.ceil(x_indices).long()

        # Handle edge cases where x is exactly at the boundaries
        x_indices_floor = torch.clamp(x_indices_floor, 0, len(self.x) - 1)
        x_indices_ceil = torch.clamp(x_indices_ceil, 0, len(self.x) - 1)

        # Get the values from the radial tables
        values_floor = self.radial_tables[:, x_indices_floor]
        values_ceil = self.radial_tables[:, x_indices_ceil]

        # Linear interpolation
        weights = (x - self.x[x_indices_floor]) / (self.x[x_indices_ceil] - self.x[x_indices_floor])
        interpolated_values = (1 - weights) * values_floor + weights * values_ceil

        assert interpolated_values.shape[0] == self.radial_tables.shape[0], "Interpolated values shape mismatch"
        assert interpolated_values.shape[1] == x.shape[0], "Interpolated values shape mismatch with input x"

        return interpolated_values.T
        
