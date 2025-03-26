import torch

from .attention import AttentionBlock


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
        self.ff_block = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_size, out_features=intermediate_size),
            torch.nn.SiLU(),
            torch.nn.Linear(in_features=intermediate_size, out_features=hidden_size),
        )

        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=hidden_size)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=hidden_size)

    def forward(
        self,
        inputs: torch.Tensor,
        radial_mask: torch.Tensor,
    ) -> torch.Tensor:
        attention_output = self.attention_block(inputs, radial_mask)
        output = inputs + attention_output
        output = self.layernorm1(output)
        output = output + self.ff_block(output)

        return self.layernorm2(output)


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
