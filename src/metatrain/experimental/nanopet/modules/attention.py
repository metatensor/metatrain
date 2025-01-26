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
    ):
        super().__init__()

        self.num_heads = num_heads
        self.in_proj = torch.nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.layernorm = torch.nn.LayerNorm(normalized_shape=hidden_size)

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

        # Trick to be able to use scaled_dot_product_attention
        q = q**2 + 1e-6
        k = k + torch.log(radial_mask[:, None, :, None])

        # Attention
        attention_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)

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
