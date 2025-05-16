import pickle
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import pandas as pd
import yaml
from dataset.SpatialGridder import SpatialGridder
#Load Config

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

cfg = load_config()
print("Loaded config:", cfg)
with open("data/drought_train_df.pkl", "rb") as f:
    df_train = pickle.load(f)
with open("data/drought_val_df.pkl", "rb") as f:
    df_val = pickle.load(f)
with open("data/drought_test_df.pkl", "rb") as f:
    df_test = pickle.load(f)

shared_lats = sorted(df_train['lat'].unique())
shared_lons = sorted(df_train['lon'].unique())

# Train
feature_cols = cfg['feature_cols']
target_cols = cfg['target_cols']
lat_key = cfg['lat_key']
lon_key = cfg['lon_key']
time_key = cfg['time_key']
gridder_X_train = SpatialGridder(lat_key=lat_key, lon_key=lon_key, time_key=time_key, feature_cols=feature_cols, lat_vals=shared_lats, lon_vals=shared_lons)
gridder_Y_train = SpatialGridder(lat_key=lat_key, lon_key=lon_key, time_key=time_key, feature_cols=target_cols, lat_vals=shared_lats, lon_vals=shared_lons)
# Val
gridder_X_val = SpatialGridder(lat_key=lat_key, lon_key=lon_key, time_key=time_key, feature_cols=feature_cols, lat_vals=shared_lats, lon_vals=shared_lons)
gridder_Y_val = SpatialGridder(lat_key=lat_key, lon_key=lon_key, time_key=time_key, feature_cols=target_cols, lat_vals=shared_lats, lon_vals=shared_lons)
# Test
gridder_X_test = SpatialGridder(lat_key=lat_key, lon_key=lon_key, time_key=time_key, feature_cols=feature_cols, lat_vals=shared_lats, lon_vals=shared_lons)
gridder_Y_test = SpatialGridder(lat_key=lat_key, lon_key=lon_key, time_key=time_key, feature_cols=target_cols, lat_vals=shared_lats, lon_vals=shared_lons)
# Fit
gridder_X_train.fit(df_train)
gridder_Y_train.fit(df_train)
gridder_X_val.fit(df_val)
gridder_Y_val.fit(df_val)
gridder_X_test.fit(df_test)
gridder_Y_test.fit(df_test)
# Transform
train_grid_data = gridder_X_train.transform(df_train)
val_grid_data = gridder_X_val.transform(df_val)
test_grid_data = gridder_X_test.transform(df_test)
print("train_grid_data shape:", train_grid_data.shape)
print("val_grid_data shape:", val_grid_data.shape)  
print("test_grid_data shape:", test_grid_data.shape)
train_target_data = gridder_Y_train.transform(df_train)
val_target_data = gridder_Y_val.transform(df_val)
test_target_data = gridder_Y_test.transform(df_test)
print("train_target_data shape:", train_target_data.shape)
print("val_target_data shape:", val_target_data.shape)
print("test_target_data shape:", test_target_data.shape)
# Inverse Transform
train_grid_data_inv = gridder_X_train.inverse_transform(train_grid_data)
train_target_data_inv = gridder_Y_train.inverse_transform(train_target_data)
print("Original train_features_ex:", df_train.head())
print("Inverse train_features_ex:", train_grid_data_inv.head())
print("Original train_target_ex:", df_train[target_cols].head())
print("Inverse train_target_ex:", train_target_data_inv.head())

print("Original train_features shape:", df_train[feature_cols].shape)
print("Original train_target shape:", df_train[target_cols].shape)
print("train_grid_data_inv shape:", train_grid_data_inv.shape)
print("train_target_data_inv shape:", train_target_data_inv.shape)

from utils.scaler import scale_all_grids