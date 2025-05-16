import matplotlib.pyplot as plt
import numpy as np
import os

def plot_prediction_vs_actual(pred, true, save_path=None):
    """
    Plot predicted vs actual values over time for a single grid cell
    pred: (T, H, W)
    true: (T, H, W)
    """
    T, H, W = pred.shape

    # Pick center location
    h, w = H // 2, W // 2

    pred_series = pred[:, h, w]
    true_series = true[:, h, w]

    plt.figure(figsize=(10, 4))
    plt.plot(true_series, label="Ground Truth", linewidth=2)
    plt.plot(pred_series, label="Prediction", linestyle='--')
    plt.title(f"Prediction vs Ground Truth at (lat={h}, lon={w})")
    plt.xlabel("Time Step")
    plt.ylabel("SPEI")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
