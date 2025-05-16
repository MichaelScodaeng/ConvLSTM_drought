# preprocess_data.py
import os
import yaml
import numpy as np
import pandas as pd
import joblib
from dataset.SpatialGridder import SpatialGridder
from sklearn.preprocessing import MinMaxScaler


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_from_splits(train_df, val_df, test_df, config, output_dir):
    target_col = config['target_cols']
    feature_cols = config['feature_cols']
    all_cols = feature_cols + target_col

    gridder = SpatialGridder(feature_cols=all_cols)

    grid_XY_train = gridder.fit_transform(train_df)
    grid_XY_val = gridder.transform(val_df)
    grid_XY_test = gridder.transform(test_df)

    C = len(feature_cols)
    grid_X_train, grid_Y_train = grid_XY_train[..., :C], grid_XY_train[..., C:]
    grid_X_val, grid_Y_val = grid_XY_val[..., :C], grid_XY_val[..., C:]
    grid_X_test, grid_Y_test = grid_XY_test[..., :C], grid_XY_test[..., C:]

    # Scale
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()

    grid_X_train_flat = grid_X_train.reshape(-1, C)
    grid_Y_train_flat = grid_Y_train.reshape(-1, 1)

    grid_X_train = scaler_X.fit_transform(grid_X_train_flat).reshape(grid_X_train.shape)
    grid_Y_train = scaler_Y.fit_transform(grid_Y_train_flat).reshape(grid_Y_train.shape)

    grid_X_val = scaler_X.transform(grid_X_val.reshape(-1, C)).reshape(grid_X_val.shape)
    grid_Y_val = scaler_Y.transform(grid_Y_val.reshape(-1, 1)).reshape(grid_Y_val.shape)
    grid_X_test = scaler_X.transform(grid_X_test.reshape(-1, C)).reshape(grid_X_test.shape)
    grid_Y_test = scaler_Y.transform(grid_Y_test.reshape(-1, 1)).reshape(grid_Y_test.shape)

    timestamps = list(gridder.timestamps)
    T_train, T_val, T_test = len(grid_X_train), len(grid_X_val), len(grid_X_test)
    timestamps_train = timestamps[:T_train]
    timestamps_val = timestamps[T_train:T_train + T_val]
    timestamps_test = timestamps[-T_test:]

    mask = ~np.isnan(grid_XY_train[0]).any(axis=-1).astype(np.float32)

    # Save all tensors
    tensors = {
        'grid_X_train': grid_X_train,
        'grid_Y_train': grid_Y_train,
        'grid_X_val': grid_X_val,
        'grid_Y_val': grid_Y_val,
        'grid_X_test': grid_X_test,
        'grid_Y_test': grid_Y_test,
        'timestamps_train': timestamps_train,
        'timestamps_val': timestamps_val,
        'timestamps_test': timestamps_test,
        'mask': mask,
    }
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(output_dir, 'data.npz'), **tensors)

    # Save scalers and gridder
    joblib.dump(scaler_X, os.path.join(output_dir, 'scaler_X.pkl'))
    joblib.dump(scaler_Y, os.path.join(output_dir, 'scaler_Y.pkl'))
    joblib.dump(gridder, os.path.join(output_dir, 'spatial_gridder.pkl'))
    print(f"Saved processed tensors and scalers to {output_dir}/")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_pkl', type=str, default='data/drought_train_df.pkl')
    parser.add_argument('--val_pkl', type=str, default='data/drought_val_df.pkl')
    parser.add_argument('--test_pkl', type=str, default='data/drought_test_df.pkl')
    parser.add_argument('--config_path', type=str, default='config/config.yaml')
    parser.add_argument('--output_dir', type=str, default='data/processed')
    args = parser.parse_args()

    
    # Load data pickle files
    train_df = pd.read_pickle(args.train_pkl)
    val_df = pd.read_pickle(args.val_pkl)
    test_df = pd.read_pickle(args.test_pkl)
    config = load_config(args.config_path)
    save_name = "_".join(config["target_cols"])
    args.output_dir = os.path.join(args.output_dir, save_name)
    preprocess_from_splits(train_df, val_df, test_df, config, args.output_dir)
