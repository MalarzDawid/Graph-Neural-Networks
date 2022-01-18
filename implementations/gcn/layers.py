import torch
from torch_geometric.nn import MessagePassing


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="add")
        self.linear = torch.nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self):
        pass
