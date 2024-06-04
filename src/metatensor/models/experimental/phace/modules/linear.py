import torch


class Linear(torch.nn.Module):

    def __init__(self, n_feat_in, n_feat_out):
        super().__init__()
        self.linear_layer = torch.nn.Linear(n_feat_in, n_feat_out, bias=False)
        self.linear_layer.weight.data.normal_(0.0, 1.0 / n_feat_in**0.5)
        # self.linear_layer.weight.data.normal_(0.0, 1.0)
        # self.n_feat_in = n_feat_in

    def forward(self, x):
        return self.linear_layer(x) # / self.n_feat_in**0.5
