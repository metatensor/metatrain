from typing import List

import torch


class Linear(torch.nn.Module):

    def __init__(self, n_feat_in, n_feat_out):
        super().__init__()
        self.linear_layer = torch.nn.Linear(n_feat_in, n_feat_out, bias=False)
        self.linear_layer.weight.data.normal_(0.0, 1.0 / n_feat_in**0.5)

    def forward(self, x):
        return self.linear_layer(x)


class Normalizer(torch.nn.Module):

    def __init__(self, dim_to_be_reduced):
        super().__init__()
        self.dim_to_be_reduced = dim_to_be_reduced
        self.is_normalized = False
        self.normalization = torch.empty(())

    def forward(self, x):
        # for now, we will not normalize
        return x
        # if not self.is_normalized:
        #     self.normalization = 1.0/torch.sqrt(torch.mean((x.detach().requires_grad_(False))**2, dim=self.dim_to_be_reduced))
        #     if torch.max(self.normalization) > 1e10:
        #         # Most likely values that are all approximately zeros were sent to a huge normalization.
        #         # We send that normalization to zero.
        #         self.normalization = torch.where(self.normalization > 1e10, torch.tensor(0.0), self.normalization)
        #     self.is_normalized = True
        # return self.normalization * x
