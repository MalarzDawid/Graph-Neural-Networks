import os
import torch_geometric
import torch_geometric.data as geom_data
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint

from models import NodeLevelGNN


def train_node_classifier(model_name, dataset, **model_kwargs):
    node_data_loader = geom_data.DataLoader(dataset, batch_size=1)
    pl.seed_everything(42)
    node_data_loader = geom_data.DataLoader(dataset, batch_size=1)

    root_dir = os.path.join("NodeLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")
        ],
        gpus=1,
        max_epochs=200,
        progress_bar_refresh_rate=0,
    )
    pl.seed_everything()
    model = NodeLevelGNN(
        model_name=model_name,
        ch_in=dataset.num_node_features,
        ch_out=dataset.num_classes,
        **model_kwargs
    )
    trainer.fit(model, node_data_loader, node_data_loader)
    model = NodeLevelGNN.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )

    # Test best model on the test set
    test_result = trainer.test(model, test_dataloaders=node_data_loader, verbose=False)
    batch = next(iter(node_data_loader))
    batch = batch.to(model.device)
    _, train_acc = model.forward(batch, mode="train")
    _, val_acc = model.forward(batch, mode="val")
    result = {"train": train_acc, "val": val_acc, "test": test_result[0]["test_acc"]}
    return model, result


def print_results(result_dict):
    if "train" in result_dict:
        print("Train accuracy: %4.2f%%" % (100.0 * result_dict["train"]))
    if "val" in result_dict:
        print("Val accuracy:   %4.2f%%" % (100.0 * result_dict["val"]))
    print("Test accuracy:  %4.2f%%" % (100.0 * result_dict["test"]))


if __name__ == "__main__":
    cora_dataset = torch_geometric.datasets.Planetoid(root="data", name="Cora")

    node_mlp_model, node_mlp_result = train_node_classifier(
        model_name="MLP", dataset=cora_dataset, ch_hidden=16, num_layers=2, dp_rate=0.1
    )

    print_results(node_mlp_result)
