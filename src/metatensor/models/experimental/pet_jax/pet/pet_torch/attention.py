import torch


class AttentionBlock(torch.nn.Module):
    """A single transformer attention block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.attention = torch.nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=attention_dropout_rate,
            bias=False,
        )
        self.layernorm = torch.nn.LayerNorm(normalized_shape=hidden_size)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(
        self,
        inputs: torch.Tensor,  # seq_len hidden_size
        radial_mask: torch.Tensor,  # seq_len
    ) -> torch.Tensor:  # seq_len hidden_size

        # Apply radial mask
        inputs = inputs * radial_mask[:, None]

        # Pre-layer normalization
        normed_inputs = self.layernorm(inputs)

        # Attention
        attention_output = self.attention(
            query=normed_inputs,
            key=normed_inputs,
            value=normed_inputs,
        )

        # Apply dropout
        output = self.dropout(attention_output)

        # Residual connection
        output += inputs

        return output
