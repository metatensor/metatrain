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

        self.layernorm = torch.nn.LayerNorm(normalized_shape=hidden_size)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(
        self,
        inputs: torch.Tensor,  # hidden_size
    ) -> torch.Tensor:  # hidden_size
        # Pre-layer normalization
        normed_inputs = self.layernorm(inputs)

        # Feed-forward
        hidden = self.mlp(normed_inputs)
        hidden = torch.nn.functional.gelu(hidden)

        # Project back to input size
        outputs = self.output(hidden)

        # Apply dropout
        outputs = self.dropout(outputs)

        # Residual connection
        outputs = (outputs + inputs) * 0.5**0.5

        return outputs
