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
    ):
        super().__init__()

        self.attention_block = AttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        self.ff_block = FeedForwardBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        radial_mask: torch.Tensor,
        use_manual_attention: bool,
    ) -> torch.Tensor:
        attention_output = self.attention_block(
            inputs, radial_mask, use_manual_attention
        )
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
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                TransformerLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        inputs,
        radial_mask,
        use_manual_attention: bool,
    ):
        x = inputs
        for layer in self.layers:
            x = layer(x, radial_mask, use_manual_attention)
        return x
