from typing import Dict, Optional

import torch


class LoRALayer(torch.nn.Module):
    def __init__(self, hidden_dim: int, rank: int):
        super(LoRALayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.A = torch.nn.Parameter(torch.randn(hidden_dim, rank))
        self.B = torch.nn.Parameter(torch.randn(rank, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.A)
        torch.nn.init.xavier_normal_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.A @ self.B


class AttentionBlockWithLoRA(torch.nn.Module):
    def __init__(self, original_module: torch.nn.Module, rank: int, alpha: float):
        super(AttentionBlockWithLoRA, self).__init__()
        self.original_module = original_module
        self.rank = rank
        self.alpha = alpha
        self.hidden_dim = original_module.output_linear.out_features
        self.lora = LoRALayer(self.hidden_dim, self.rank)

    def forward(
        self, x: torch.Tensor, multipliers: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.original_module(x, multipliers) + self.alpha * self.lora(x)


class LoRAWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, rank: int, alpha: float):
        super(LoRAWrapper, self).__init__()
        self.model = model
        self.hypers = model.hypers
        self.rank = rank
        self.alpha = alpha
        self.hidden_dim = model.hypers.TRANSFORMER_D_MODEL
        self.num_hidden_layers = model.hypers.N_GNN_LAYERS * model.hypers.N_TRANS_LAYERS
        for param in model.parameters():
            param.requires_grad = False
        for gnn_layer in model.gnn_layers:
            for trans_layer in gnn_layer.trans.layers:
                trans_layer.attention = AttentionBlockWithLoRA(
                    trans_layer.attention, self.rank, self.alpha
                )

    def forward(
        self,
        batch_dict: Dict[str, torch.Tensor],
        rotations: Optional[torch.Tensor] = None,
    ):
        return self.model(batch_dict, rotations)
