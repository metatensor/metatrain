import torch

from .attention import AttentionBlock
from .feedforward import FeedForwardBlock


class TransformerLayer(torch.nn.Module):
    """A single transformer layer."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
    ):
        super().__init__()

        self.attention_block = AttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
        )
        self.ff_block = FeedForwardBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        radial_mask: torch.Tensor,
    ) -> torch.Tensor:
        attention_output = self.attention_block(inputs, radial_mask)
        output = self.ff_block(attention_output)

        return output


class Transformer(torch.nn.Module):
    """A transformer model."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        attention_dropout_rate: float,
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                TransformerLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        inputs,
        radial_mask,
    ):
        x = inputs
        for layer in self.layers:
            x = layer(x, radial_mask)
        return x
