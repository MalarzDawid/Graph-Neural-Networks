import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(
        self, ch_in, ch_out, num_heads=1, concat_heads=True, alpha=0.2
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert (
                ch_out % num_heads == 0
            ), "Number of output features must be a multiple of the count of heads"
            c_out = ch_out // num_heads

        self.projection = nn.Linear(ch_in, ch_out * num_heads)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * ch_out))
        self.leakyrelu = nn.LeakyReLU(alpha)

        # Initialization from the original implementation
        nn.init.xavier_uniform(self.projection.weight.data, gain=1.414)
        nn.init.xavier_uniform(self.a.data, gain=1.414)

    def forward(self, node_feats, adj_matrix, print_attn_probs=False):
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)
        node_feats = self.projection(node_feats)
        node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)

        edges = adj_matrix.nonzero(as_tuple=False)
        node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)
        edge_indices_row = edges[:, 0] * num_nodes + edges[:, 1]
        edge_indices_col = edges[:, 0] * num_nodes + edges[:, 2]
        a_input = torch.cat(
            [
                torch.index_select(
                    input=node_feats_flat, index=edge_indices_row, dim=0
                ),
                torch.index_select(
                    input=node_feats_flat, index=edge_indices_col, dim=0
                ),
            ],
            dim=-1,
        )

        # Calc attention
        attn_logits = torch.einsum("bhc, hc->bh", a_input, self.a)
        attn_logits = self.leakyrelu(attn_logits)

        # Map list to matrix
        attn_matrix = attn_logits.new_zeros(adj_matrix.shape + (self.num_heads,)).fill_(
            -9e15
        )
        attn_matrix[
            adj_matrix[..., None].repeat(1, 1, 1, self.num_heads) == 1
        ] = attn_logits.reshape(-1)

        attn_probs = F.softmax(attn_matrix, dim=2)
        if print_attn_probs:
            print("Attention probs:\n", attn_probs.permute(0, 3, 1, 2))
        node_feats = torch.einsum("bijh,bjhc->bihc", attn_probs, node_feats)

        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)
        return node_feats


if __name__ == "__main__":
    node_feats = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
    adj_matrix = torch.Tensor(
        [[[1, 1, 0, 0], [1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]]]
    )
    print(f"Node features:\n {node_feats}\n")
    print(f"Adacency matrix:\n {adj_matrix}\n")

    layer = GATLayer(2, 2, num_heads=2)
    layer.projection.weight.data = torch.Tensor([[1.0, 0.0], [0.0, 1.0]])
    layer.projection.bias.data = torch.Tensor([0.0, 0.0])
    layer.a.data = torch.Tensor([[-0.2, 0.3], [0.1, -0.1]])

    with torch.no_grad():
        out_feats = layer(node_feats, adj_matrix, print_attn_probs=True)

    print("Adjacency matrix", adj_matrix)
    print("Input features", node_feats)
    print("Output features", out_feats)
