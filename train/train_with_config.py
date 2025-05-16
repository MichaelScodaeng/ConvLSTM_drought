# train_with_config.py
import argparse
import yaml
import numpy as np
import joblib
import os
from train.trainer_utils import get_trainer
from models.model import LightningConvLSTMModule, EncodingForecastingConvLSTM
from loaders.loader import create_dataloaders

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(config_path):
    config = load_config(config_path)
    target_name = "_".join(config["target_cols"])
    data_path = f"data/preprocessed/{target_name}/data_{target_name}.npz"

    # Load tensors
    data = np.load(data_path, allow_pickle=True)
    grid_X_train = data["grid_X_train_scaled"]
    grid_Y_train = data["grid_Y_train_scaled"]
    grid_X_val = data["grid_X_val_scaled"]
    grid_Y_val = data["grid_Y_val_scaled"]
    grid_X_test = data["grid_X_test_scaled"]
    grid_Y_test = data["grid_Y_test_scaled"]
    mask = data["mask"]
    timestamps_train = data["timestamps_train"].tolist()
    timestamps_val = data["timestamps_val"].tolist()
    timestamps_test = data["timestamps_test"].tolist()

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        grid_X_train, grid_Y_train,
        grid_X_val, grid_Y_val,
        grid_X_test, grid_Y_test,
        pre_seq_len=config["pre_seq_length"],
        aft_seq_len=config["aft_seq_length"],
        batch_size=config["batch_size"],
        timestamps_train=timestamps_train,
        timestamps_val=timestamps_val,
        timestamps_test=timestamps_test,
        mask=mask
    )

    model = EncodingForecastingConvLSTM(
        input_dim=grid_X_train.shape[-1],
        hidden_dim=config["hidden_dim"],
        kernel_size=config["kernel_size"],
        dropout=config["dropout"],
        num_layers=config["num_layers"],
        pre_seq_length=config["pre_seq_length"],
        aft_seq_length=config["aft_seq_length"]
    )

    lightning_model = LightningConvLSTMModule(model, lr=config["lr"], weight_decay=config["weight_decay"])
    trainer = get_trainer(max_epochs=config["epochs"])
    trainer.fit(lightning_model, train_loader, val_loader)
    trainer.test(lightning_model, test_loader)
    original_config = load_config("config/config.yaml")
    target_name = "_".join(original_config["target_cols"])
    trainer.save_checkpoint(f"outputs/final_model_{target_name}.ckpt")
    print("✅ Final model saved as outputs/final_model.ckpt")
    metrics = trainer.callback_metrics
    print("📉 Final Train Loss:", metrics.get("train_loss"))
    print("📉 Final Val Loss:", metrics.get("val_loss"))
    print("📈 Final Train RMSE:", metrics.get("train_rmse"))
    print("📈 Final Val RMSE:", metrics.get("val_rmse"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/config.yaml")
    args = parser.parse_args()
    main(args.config_path)