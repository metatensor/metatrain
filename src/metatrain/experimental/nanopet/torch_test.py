import torch


model = torch.nn.MultiheadAttention(
    embed_dim=512,
    num_heads=8,
    bias=False,
)
torch.jit.script(model)
