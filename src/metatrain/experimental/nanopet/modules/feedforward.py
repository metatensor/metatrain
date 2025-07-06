import torch


class FeedForwardBlock(torch.nn.Module):
    """A single transformer feed forward block."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout_rate: float,
    ):
        super().__init__()

        self.mlp = torch.nn.Linear(
            in_features=hidden_size, out_features=intermediate_size, bias=False
        )
        self.output = torch.nn.Linear(
            in_features=intermediate_size, out_features=hidden_size, bias=False
        )

        self.rms_norm = torch.nn.RMSNorm(normalized_shape=(hidden_size, 3, 3))

    def forward(
        self,
        inputs: torch.Tensor,  # hidden_size
    ) -> torch.Tensor:  # hidden_size
        # Pre-layer normalization
        normed_inputs = self.rms_norm(inputs)
        # normed_inputs = inputs

        # Feed-forward
        normed_inputs = normed_inputs.permute(0, 1, 3, 4, 2)
        hidden = self.mlp(normed_inputs)
        hidden = hidden.permute(0, 1, 4, 2, 3)

        # artificial "matrix SiLU"
        hidden = hidden / torch.sum(torch.diagonal(1 + torch.matrix_exp(-hidden.contiguous()), dim1=-2, dim2=-1), dim=-1).unsqueeze(-1).unsqueeze(-2)

        # Project back to input size
        hidden = hidden.permute(0, 1, 3, 4, 2)
        outputs = self.output(hidden)
        outputs = outputs.permute(0, 1, 4, 2, 3)

        # Residual connection
        outputs = (outputs + inputs) * 0.5**0.5

        return outputs
