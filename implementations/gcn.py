import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.projection = nn.Linear(ch_in, ch_out)

    def forward(self, node_feats, adj_matrix):
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
        node_feats = self.projection(node_feats)
        node_feats = torch.bmm(adj_matrix, node_feats)
        node_feats = node_feats / num_neighbours
        return node_feats


if __name__ == "__main__":
    node_feats = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
    adj_matrix = torch.Tensor(
        [[[1, 1, 0, 0], [1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]]]
    )
    print(f"Node features:\n {node_feats}\n")
    print(f"Adacency matrix:\n {adj_matrix}\n")

    # Create layer
    layer = GCNLayer(ch_in=2, ch_out=2)
    layer.projection.weight.data = torch.Tensor([[1.0, 0.0], [0.0, 1.0]])
    layer.projection.bias.data = torch.Tensor([0.0, 0.0])

    with torch.no_grad():
        out_feats = layer(node_feats, adj_matrix)

    print(f"Node features:\n {node_feats}\n")
    print(f"Adacency matrix:\n {adj_matrix}\n")
    print(f"Output features:\n {out_feats}\n")
