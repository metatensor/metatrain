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
        self.rms_norm = torch.nn.Parameter(torch.ones(hidden_size))
        self.attention_dropout_rate = attention_dropout_rate

    def forward(
        self,
        inputs: torch.Tensor,  # [nodes, edges, d_pet, l_max+1, l_max+1]
        radial_mask: torch.Tensor,  # [nodes, edges]
    ) -> torch.Tensor:  # [nodes, edges, d_pet, l_max+1, l_max+1]
        # Pre-layer normalization
        # normed_inputs = inputs.permute(0, 1, 3, 4, 2)
        # normed_inputs = self.rms_norm(normed_inputs)
        # normed_inputs = inputs
        # normed_inputs = normed_inputs.permute(0, 1, 4, 2, 3)
        std = torch.std(torch.sum(torch.diagonal(inputs, dim1=-2, dim2=-1), dim=-1), dim=-1, keepdim=True).unsqueeze(-1).unsqueeze(-2)
        normed_inputs = inputs / (std + 1e-8)  # Normalize along the feature dimension
        normed_inputs = self.rms_norm.reshape(-1, 1, 1) * inputs / (std + 1e-8)  # Normalize along the feature dimension

        # Input projection
        normed_inputs = normed_inputs.permute(0, 1, 3, 4, 2)  # [nodes, edges, l_max+1, l_max+1, d_pet]
        qkv = self.in_proj(normed_inputs)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.permute(0, 1, 4, 2, 3)  # [nodes, edges, d_pet, l_max+1, l_max+1]
        k = k.permute(0, 1, 4, 2, 3)
        v = v.permute(0, 1, 4, 2, 3)
        # Split heads
        l_max_plus_one = q.size(-1)
        q = q.reshape(q.size(0), q.size(1), self.num_heads, q.size(2) // self.num_heads, l_max_plus_one, l_max_plus_one)
        k = k.reshape(k.size(0), k.size(1), self.num_heads, k.size(2) // self.num_heads, l_max_plus_one, l_max_plus_one)
        v = v.reshape(v.size(0), v.size(1), self.num_heads, v.size(2) // self.num_heads, l_max_plus_one, l_max_plus_one)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention
        queries_times_keys = torch.einsum("nhelab, nhflbc -> nhefac", q, k)  # contract between feature dimension l and at the same time Hartmut dimension b
        exponential = torch.matrix_exp(queries_times_keys.contiguous() / (k.size(3) ** 0.5))
        attention_weights = exponential * radial_mask[:, None, None, :, None, None] / (torch.sum(torch.diagonal(torch.sum(exponential, dim=-3, keepdim=True), dim1=-2, dim2=-1), dim=-1).unsqueeze(-1).unsqueeze(-2) + 1e-6)  # two sums, one to sum over matrices, one to get the trace

        # Attention output
        attention_output = torch.einsum("nhefab, nhflbc -> nhelac", attention_weights, v)  # contract between edge dimension f and at the same time Hartmut dimension b
        attention_output = attention_output.transpose(1, 2)  # [nodes, edges, head, features, l_max+1, l_max+1]
        attention_output = attention_output.reshape(
            attention_output.size(0),
            attention_output.size(1),
            attention_output.size(2) * attention_output.size(3),  # re-aggregate heads and features
            attention_output.size(4),
            attention_output.size(5),
        )

        # Output projection
        attention_output = attention_output.permute(0, 1, 3, 4, 2)
        outputs = self.out_proj(attention_output)
        outputs = outputs.permute(0, 1, 4, 2, 3)

        # Residual connection
        outputs = (outputs + inputs) * 0.5**0.5

        return outputs
