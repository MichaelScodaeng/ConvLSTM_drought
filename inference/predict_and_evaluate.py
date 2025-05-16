import argparse
import os
import torch
import numpy as np
import yaml
import joblib
import matplotlib.pyplot as plt

from models.model import EncodingForecastingConvLSTM
from loaders.loader import create_dataloaders  # ← YOUR version
from utils.load_config import load_config
from utils.metrics import masked_rmse, masked_r2
from utils.visualization import plot_prediction_vs_actual


def load_model_from_checkpoint(config, grid_X_train,checkpoint_path, device):
    model = EncodingForecastingConvLSTM(
        input_dim=grid_X_train.shape[-1],
        hidden_dim=config["hidden_dim"],
        kernel_size=config["kernel_size"],
        dropout=config["dropout"],
        num_layers=config["num_layers"],
        pre_seq_length=config["pre_seq_length"],
        aft_seq_length=config["aft_seq_length"]
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
@torch.no_grad()
def evaluate_model(model, dataloader, scaler_y, device):
    preds_all, targets_all = [], []
    output = None
    for x, y, mask_batch in dataloader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        if torch.isnan(x).any():
            print("⚠️ Input to model contains NaNs")

        output = model(x)

        if torch.isnan(output).any():
            print("🚨 Model output contains NaNs")
        preds_all.append(output.cpu())
        targets_all.append(y.cpu())
    print("Sample output:", output[0].cpu().numpy())
    preds = torch.cat(preds_all, dim=0).numpy()     # (B, T, C, H, W)
    targets = torch.cat(targets_all, dim=0).numpy() # (B, T, C, H, W)

    B, T, C, H, W = preds.shape

    # Make sure C == 1
    assert C == 1, f"Expected C=1, got {C}"
    #load npz mask
    config = load_config("config/config.yaml")
    target_name = "_".join(config["target_cols"])
    mask = np.load(f"data/preprocessed/{target_name}/data_{target_name}.npz")["mask"]

    # Reshape to (B * H * W, T)
    preds_flat = preds.reshape(B, T, C, H, W).transpose(0, 3, 4, 2, 1).reshape(-1, T)  # (B*H*W, T)
    targets_flat = targets.reshape(B, T, C, H, W).transpose(0, 3, 4, 2, 1).reshape(-1, T)

    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"Mask valid count: {np.sum(mask)}, total: {mask.size}")
    print(f"Mask preview:\n{mask.astype(int)}")


    print(f"preds_flat min/max before inverse: {np.min(preds_flat)}, {np.max(preds_flat)}")
    print(f"targets_flat min/max before inverse: {np.min(targets_flat)}, {np.max(targets_flat)}")
    print(f"preds_flat NaN count: {np.isnan(preds_flat).sum()}")
    print(f"targets_flat NaN count: {np.isnan(targets_flat).sum()}")
    

    preds_inv = scaler_y.inverse_transform(preds_flat).reshape(B, H, W, T)
    targets_inv = scaler_y.inverse_transform(targets_flat).reshape(B, H, W, T)



    # Transpose to (B, T, H, W) for metric consistency
    preds_inv = preds_inv.transpose(0, 3, 1, 2)
    targets_inv = targets_inv.transpose(0, 3, 1, 2)
    print("🔍 DEBUG SHAPES:")
    print(f"preds_inv.shape: {preds_inv.shape}")     # Expect: (B, T, H, W)
    print(f"targets_inv.shape: {targets_inv.shape}")

    print("🔍 DEBUG STATS:")
    print(f"preds_inv min/max: {np.nanmin(preds_inv)} / {np.nanmax(preds_inv)}")
    print(f"targets_inv min/max: {np.nanmin(targets_inv)} / {np.nanmax(targets_inv)}")

    print(f"preds_inv NaN count: {np.isnan(preds_inv).sum()}")
    print(f"targets_inv NaN count: {np.isnan(targets_inv).sum()}")

    rmse = masked_rmse(targets_inv, preds_inv)
    r2 = masked_r2(targets_inv, preds_inv)



    return preds_inv, targets_inv, rmse, r2




def save_results(preds, targets, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{name}_pred.npy"), preds)
    np.save(os.path.join(output_dir, f"{name}_true.npy"), targets)


def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_name = "_".join(config["target_cols"])
    data_path = f"data/preprocessed/{target_name}/scaler_Y_{target_name}.pkl"

    scaler_y = joblib.load(data_path)
    data = np.load(f"data/preprocessed/{target_name}/data_{target_name}.npz", allow_pickle=True)

    # Load tensors
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
    model_path = f"outputs/final_model_{target_name}.ckpt"

    model = load_model_from_checkpoint(config,grid_X_train, model_path, device)
    
    mask_indices = np.argwhere(mask == 1)
    lat_idx, lon_idx = mask_indices[0]  # First valid cell
    print(f"Plotting valid location: lat={lat_idx}, lon={lon_idx}")
    print("🔍 Train Set")
    train_preds, train_targets, train_rmse, train_r2 = evaluate_model(model, train_loader, scaler_y, device)
    print(f"✅ RMSE: {train_rmse:.4f} | R²: {train_r2:.4f}")
    train_save_path = os.path.join("outputs/train", f"{target_name}")
    save_results(train_preds, train_targets, train_save_path, "train")


    print("🔍 Validation Set")
    val_preds, val_targets, val_rmse, val_r2 = evaluate_model(model, val_loader, scaler_y, device)
    print(f"✅ RMSE: {val_rmse:.4f} | R²: {val_r2:.4f}")
    val_save_path = os.path.join("outputs/val", f"{target_name}")
    save_results(val_preds, val_targets, val_save_path, "val")

    print("🧪 Test Set")
    test_preds, test_targets, test_rmse, test_r2 = evaluate_model(model, test_loader, scaler_y, device)
    test_save_path = os.path.join("outputs/test", f"{target_name}")
    print(f"✅ RMSE: {test_rmse:.4f} | R²: {test_r2:.4f}")
    save_results(test_preds, test_targets, test_save_path, "test")

    print("val_preds[0] shape:", val_preds[0].shape)
    print("val_targets[0] shape:", val_targets[0].shape)
    print("val_preds[0] values:", val_preds[0])
    print("val_targets[0] values:", val_targets[0])
    print("Any NaN in preds?", np.isnan(val_preds[0]).any())
    print("Any NaN in targets?", np.isnan(val_targets[0]).any())
    lat_idx, lon_idx = 0, 7
    plt.plot(val_targets[0, :, lat_idx, lon_idx], label="Ground Truth")
    plt.plot(val_preds[0, :, lat_idx, lon_idx], label="Prediction", linestyle="--")
    plt.legend()
    plt.show()

    print("📊 Plotting one validation sample...")
    plot_prediction_vs_actual(val_preds[0], val_targets[0], save_path=os.path.join(val_save_path, "val_sample0.png"))
    save_pred_vs_actual_csv(
    val_preds, val_targets,
    h=0, w=7,
    save_path=os.path.join(val_save_path, "val_pred_vs_actual_lat15_lon8.csv")
)
import pandas as pd

def save_pred_vs_actual_csv(preds, targets, h, w, save_path):
    """
    Save prediction vs actual for a specific spatial location to CSV.
    Args:
        preds: [B, T, H, W]
        targets: [B, T, H, W]
        h, w: spatial location to extract
    """
    pred_series = preds[:, :, h, w].squeeze()    # shape: [B, T] → flatten to 1D
    true_series = targets[:, :, h, w].squeeze()

    time_steps = list(range(pred_series.shape[0]))
    df = pd.DataFrame({
        "time_step": time_steps,
        "prediction": pred_series,
        "ground_truth": true_series
    })
    df.to_csv(save_path, index=False)
    print(f"✅ Saved prediction vs ground truth to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()
    main(args.config_path)
