import torch_geometric
import torch_geometric.data as geom_data
import torch_geometric.nn as geom_nn

cora_dataset = torch_geometric.datasets.Planetoid(root="data/", name="Cora")
print(cora_dataset[0])
