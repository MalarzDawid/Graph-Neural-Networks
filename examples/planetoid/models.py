import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as geom_nn
import pytorch_lightning as pl


gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv,
}


class GNNModel(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_hidden,
        ch_out,
        num_layers: int = 2,
        layer_name: str = "GCN",
        dp_rate: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = ch_in, ch_hidden
        for _ in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels, out_channels=out_channels, **kwargs),
                nn.ReLu(inplace=True),
                nn.Dropout(dp_rate),
            ]
        layers += [gnn_layer(in_channels=in_channels, out_channels=ch_out, **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for layer in self.layers:
            if isinstance(layer, geom_nn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x


class MLPModel(nn.Module):
    def __init__(
        self, ch_in, ch_hidden, ch_out, num_layers: int = 2, dp_rate: float = 0.1
    ) -> None:
        super().__init__()
        layers = []
        in_channels, out_channels = ch_in, ch_hidden
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate),
            ]
            in_channels = ch_hidden
        layers += [nn.Linear(in_channels, ch_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        return self.layers(x)


class NodeLevelGNN(pl.LightningModule):
    def __init__(self, model_name, **model_kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        if model_name == "MLP":
            self.model = MLPModel(**model_kwargs)
        else:
            self.model = GNNModel(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, data, mode: str = "train"):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)

        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, "Unknown forward mode: %s" % mode

        loss = self.loss_module(x[mask], data.y[mask])
        acc = (x[mask].argmax(-1) == data.y[mask]).sum().float() / mask.sum()
        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-3
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log("test_acc", acc)
