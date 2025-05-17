import argparse
import yaml
import numpy as np
import joblib
import os
import torch
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
    
    # Load the scaler to get min/max values for inverse transform
    scaler_path = f"data/preprocessed/{target_name}/scaler_{target_name}.joblib"
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_min = scaler.data_min_[0] if hasattr(scaler, 'data_min_') else scaler.min_[0]
        X_max = scaler.data_max_[0] if hasattr(scaler, 'data_max_') else scaler.max_[0]
        print(f"✅ Loaded scaler with X_min={X_min}, X_max={X_max}")
    else:
        # Fallback if scaler not found
        print("⚠️ Scaler not found, using estimated min/max from data")
        X_min = min(np.min(grid_Y_train), np.min(grid_Y_val), np.min(grid_Y_test))
        X_max = max(np.max(grid_Y_train), np.max(grid_Y_val), np.max(grid_Y_test))
    
    try:
        timestamps_train = data["timestamps_train"].tolist()
        timestamps_val = data["timestamps_val"].tolist()
        timestamps_test = data["timestamps_test"].tolist()
    except (KeyError, AttributeError):
        print("⚠️ Timestamps not found in data file")
        timestamps_train = None
        timestamps_val = None
        timestamps_test = None

    # Print dataset shapes for verification
    print(f"Training data shapes: X={grid_X_train.shape}, Y={grid_Y_train.shape}")
    print(f"Validation data shapes: X={grid_X_val.shape}, Y={grid_Y_val.shape}")
    print(f"Test data shapes: X={grid_X_test.shape}, Y={grid_Y_test.shape}")
    print(f"Mask shape: {mask.shape}")
    
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
    
    # Ensure kernel_size is in the correct format
    kernel_size = config["kernel_size"]
    if isinstance(kernel_size, list) and len(kernel_size) == 2 and all(isinstance(k, int) for k in kernel_size):
        kernel_size = tuple(kernel_size)
    print(f"Using kernel_size: {kernel_size}")

    model = EncodingForecastingConvLSTM(
        input_dim=grid_X_train.shape[-1],
        hidden_dim=config["hidden_dim"],
        kernel_size=kernel_size,
        dropout=config.get("dropout", 0.0),  # Default to 0 if not specified
        num_layers=config["num_layers"],
        pre_seq_length=config["pre_seq_length"],
        aft_seq_length=config["aft_seq_length"]
    )

    # Validate model architecture
    print(f"Model created with input_dim={grid_X_train.shape[-1]}, hidden_dim={config['hidden_dim']}")
    
    # When initializing the Lightning module, pass X_min and X_max
    lightning_model = LightningConvLSTMModule(
        model=model, 
        lr=config["lr"], 
        weight_decay=config["weight_decay"],
        X_min=X_min,
        X_max=X_max
    )
    
    # Create output directory
    output_dir = f"outputs/{target_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get trainer
    trainer = get_trainer(
        max_epochs=config["epochs"],
        patience=10,  # Early stopping patience
        monitor="val_loss",
        output_dir=output_dir,
        log_every_n_steps=10
    )
    
    # Train and test the model
    print("🚀 Starting training...")
    trainer.fit(lightning_model, train_loader, val_loader)
    print("🧪 Running test...")
    trainer.test(lightning_model, test_loader)
    
    # Save the final model
    model_save_path = f"{output_dir}/final_model_{target_name}.ckpt"
    trainer.save_checkpoint(model_save_path)
    print(f"✅ Final model saved as {model_save_path}")
    
    # Print final metrics
    metrics = trainer.callback_metrics
    print("\n📊 Final Metrics:")
    print(f"📉 Train Loss: {metrics.get('train_loss', 'N/A')}")
    print(f"📉 Val Loss: {metrics.get('val_loss', 'N/A')}")
    print(f"📉 Test Loss: {metrics.get('test_loss', 'N/A')}")
    print(f"📈 Train RMSE: {metrics.get('train_rmse', 'N/A')}")
    print(f"📈 Val RMSE: {metrics.get('val_rmse', 'N/A')}")
    print(f"📈 Test RMSE: {metrics.get('test_rmse', 'N/A')}")
    print(f"📈 Train R²: {metrics.get('train_r2', 'N/A')}")
    print(f"📈 Val R²: {metrics.get('val_r2', 'N/A')}")
    print(f"📈 Test R²: {metrics.get('test_r2', 'N/A')}")
    
    # Save a copy of the configuration with the model
    config_save_path = f"{output_dir}/config_{target_name}.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    print(f"✅ Configuration saved as {config_save_path}")

    return trainer, lightning_model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ConvLSTM model with configuration")
    parser.add_argument("--config_path", type=str, default="config/config.yaml",
                       help="Path to the configuration YAML file")
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
    
    main(args.config_path)