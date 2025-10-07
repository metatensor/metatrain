import torch


class AttentionBlock(torch.nn.Module):
    """
    A single transformer attention block. We are not using the
    MultiHeadAttention module from torch.nn because we need to apply a
    radial mask to the attention weights.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.in_proj = torch.nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.layernorm = torch.nn.LayerNorm(normalized_shape=hidden_size)
        self.attention_dropout_rate = attention_dropout_rate

    def forward(
        self,
        inputs: torch.Tensor,  # seq_len hidden_size
        radial_mask: torch.Tensor,  # seq_len
    ) -> torch.Tensor:  # seq_len hidden_size
        # Pre-layer normalization
        normed_inputs = self.layernorm(inputs)

        # Input projection
        qkv = self.in_proj(normed_inputs)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Split heads
        q = q.reshape(q.size(0), q.size(1), self.num_heads, q.size(2) // self.num_heads)
        k = k.reshape(k.size(0), k.size(1), self.num_heads, k.size(2) // self.num_heads)
        v = v.reshape(v.size(0), v.size(1), self.num_heads, v.size(2) // self.num_heads)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Attention
        attention_weights = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        attention_weights = attention_weights.softmax(dim=-1)
        attention_weights = torch.nn.functional.dropout(
            attention_weights, p=self.attention_dropout_rate, training=self.training
        )

        # Radial mask
        attention_weights = attention_weights * radial_mask[:, None, None, :]
        attention_weights = attention_weights / (
            attention_weights.sum(dim=-1, keepdim=True) + 1e-6
        )

        # Attention output
        attention_output = torch.matmul(attention_weights, v)
        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.reshape(
            attention_output.size(0),
            attention_output.size(1),
            attention_output.size(2) * attention_output.size(3),
        )

        # Output projection
        outputs = self.out_proj(attention_output)

        # Residual connection
        outputs = (outputs + inputs) * 0.5**0.5

        return outputs
