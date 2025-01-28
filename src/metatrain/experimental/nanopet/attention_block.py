import torch
import torch.nn as nn
import torch.nn.functional as F

def cutoff_neg_inf(x, r_cut, cutoff_delta):
    """
    Return a infinitelly differentiable function f(x) on [0, r_cut) such that:
      - f(x) = 0 for x <= r_cut - cutoff_delta
      - f(x) --> -inf as x --> r_cut^-
    and is piecewise defined with a smooth drop near r_cut.
    
    Parameters
    ----------
    x : 1D numpy array
        The points at which f is evaluated.
    r_cut : float
        The right endpoint (not inclusive) at which we want f --> -inf.
    cutoff_delta : float
        The width of the 'transition region' in which f drops from 0 to -inf.
    
    Returns
    -------
    fvals : 1D numpy array
        The values of the smoothly cutoff function at the points in `x`.
    """
    fvals = torch.zeros_like(x)

    mask = (x > r_cut - cutoff_delta) & (x < r_cut)
    x_sub = x[mask]
    
    y = (x_sub - (r_cut - cutoff_delta)) / cutoff_delta
    
    numerator = torch.exp(-1.0 / (y**2))
    denominator = (1.0 - y)
    
    frac = -numerator / denominator
    
    fvals[mask] = frac
    
    return fvals

class PETAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, r_cut: float, cutoff_delta: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.input_linear = nn.Linear(d_model, 3 * d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.r_cut = r_cut
        self.cutoff_delta = cutoff_delta
        
    def forward(
        self,
        x: torch.Tensor,               # [batch_size, sequence_length, d_model]
        displacement_vectors: torch.Tensor,  # [batch_size, sequence_length, 3]
        mask_exists: torch.Tensor # [batch_size, sequence_length]
    ) -> torch.Tensor:

        batch_size, seq_len, _ = x.size()

        x = self.input_linear(x)  

        q, k, v = x.split(self.d_model, dim=-1)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        lengths = displacement_vectors.norm(dim=-1)

        cutoffs = cutoff_neg_inf(lengths, self.r_cut, self.cutoff_delta)

        attn_mask = cutoffs.unsqueeze(-1).repeat(1, 1, seq_len)
        mask_exists_2d = mask_exists.unsqueeze(-1).repeat(1, 1, seq_len)
        attn_mask[~mask_exists_2d] = float('-inf')

        attn_mask = attn_mask.reshape(attn_mask.shape[0], 1, attn_mask.shape[1], attn_mask.shape[2])
        '''print(attn_mask.shape)
        print(q.shape)
        print(k.shape)
        print(v.shape)'''
        
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

        output = self.output_linear(attn_output)

        return output
