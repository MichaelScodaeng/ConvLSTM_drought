import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def masked_rmse(y_true, y_pred, mask_value=np.nan):
    """
    Compute RMSE while ignoring values equal to mask_value (default: NaN)
    """
    mask = ~np.isnan(y_true) if np.isnan(mask_value) else y_true != mask_value
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    return np.sqrt(mean_squared_error(y_true_masked, y_pred_masked))

def masked_r2(y_true, y_pred, mask_value=np.nan):
    """
    Compute R² while ignoring values equal to mask_value (default: NaN)
    """
    mask = ~np.isnan(y_true) if np.isnan(mask_value) else y_true != mask_value
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    return r2_score(y_true_masked, y_pred_masked)
