# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Planetoid

# %%
# Load the dataset
dataset = Planetoid(root="../data", name="Cora")

# %%

# Dataset properties

print(dataset)
print("number of graphs:\t\t", len(dataset))
print("number of classes:\t\t", dataset.num_classes)
print("number of node features:\t", dataset.num_node_features)
print("number of edge features:\t", dataset.num_edge_features)

# %%
print(dataset.data)

# %%
data = dataset[0]

# %%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv = SAGEConv(
            dataset.num_features, dataset.num_classes, aggr="max"
        )  # max, mean, add ...)

    def forward(self):
        x = self.conv(data.x, data.edge_index)
        return F.log_softmax(x, dim=1)


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

#%%
def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data("train_mask", "val_mask", "test_mask"):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

#%%
# training 

best_val_acc = test_acc = 0
for epoch in range(1, 100):
    train()
    _, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = "Epoch: {:03d}, Val: {:.4f}, Test: {:.4f}"

    if epoch % 10 == 0:
        print(log.format(epoch, best_val_acc, test_acc))
