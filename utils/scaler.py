from sklearn.preprocessing import MinMaxScaler
import numpy as np

def scale_gridded_tensor(grid, scaler=None, fit=False, feature_range=(-1, 1)):
    """
    Reshape a 4D tensor [T, H, W, C] into 2D for scaling and reshape back.
    """
    T, H, W, C = grid.shape
    flat = grid.reshape(-1, C)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=feature_range)

    if fit:
        flat_scaled = scaler.fit_transform(flat)
    else:
        flat_scaled = scaler.transform(flat)

    return flat_scaled.reshape(T, H, W, C), scaler

def inverse_scale_gridded_tensor(grid_scaled, scaler):
    """
    Inverse transform a scaled 4D grid tensor using the fitted scaler.
    """
    T, H, W, C = grid_scaled.shape
    flat_scaled = grid_scaled.reshape(-1, C)
    flat_orig = scaler.inverse_transform(flat_scaled)
    return flat_orig.reshape(T, H, W, C)

def scale_all_grids(grid_X_train, grid_X_val, grid_X_test,
                     grid_Y_train, grid_Y_val, grid_Y_test,
                     feature_range=(-1, 1)):
    """
    Scale feature and target grids using only training data statistics.
    """
    grid_X_train_scaled, scaler_X = scale_gridded_tensor(grid_X_train, fit=True, feature_range=feature_range)
    grid_X_val_scaled, _ = scale_gridded_tensor(grid_X_val, scaler_X)
    grid_X_test_scaled, _ = scale_gridded_tensor(grid_X_test, scaler_X)

    grid_Y_train_scaled, scaler_Y = scale_gridded_tensor(grid_Y_train, fit=True, feature_range=feature_range)
    grid_Y_val_scaled, _ = scale_gridded_tensor(grid_Y_val, scaler_Y)
    grid_Y_test_scaled, _ = scale_gridded_tensor(grid_Y_test, scaler_Y)

    return (grid_X_train_scaled, grid_X_val_scaled, grid_X_test_scaled,
            grid_Y_train_scaled, grid_Y_val_scaled, grid_Y_test_scaled,
            scaler_X, scaler_Y)

def check_inverse_scaling(original, restored, tolerance=1e-4):
    if original.shape != restored.shape:
        print(f"Shape mismatch: {original.shape} vs {restored.shape}")
        return False, None, None

    # Create mask where both are not NaN
    mask = ~np.isnan(original) & ~np.isnan(restored)
    valid_count = np.sum(mask)

    if valid_count == 0:
        print("⚠️ No valid (non-NaN) elements to compare.")
        return False, np.nan, 0

    diff = np.abs(original[mask] - restored[mask])
    max_diff = np.max(diff)
    mismatch_count = np.sum(diff > tolerance)

    if mismatch_count == 0:
        print(f"✅ Inverse scaling successful: all {valid_count} values match within tolerance = {tolerance}.")
        return True, max_diff, mismatch_count
    else:
        print(f"❌ Inverse scaling mismatch: {mismatch_count} of {valid_count} values differ (max diff = {max_diff:.6f})")
        return False, max_diff, mismatch_count

