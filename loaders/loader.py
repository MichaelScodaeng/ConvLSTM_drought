from torch.utils.data import DataLoader
from dataset.GriddedSeq2SeqDataset import GriddedSeq2SeqDataset
import numpy as np
import torch
def create_dataloaders(grid_X_train, grid_Y_train,
                        grid_X_val, grid_Y_val,
                        grid_X_test, grid_Y_test,
                        pre_seq_len, aft_seq_len,
                        batch_size=16, timestamps_train=None,
                        timestamps_val=None, timestamps_test=None,
                        mask=None, num_workers=2, pin_memory=True):
    """
    Create DataLoaders for train, val, test splits.
    
    Args:
        grid_X_train, grid_X_val, grid_X_test: Input feature grids [T, H, W, F]
        grid_Y_train, grid_Y_val, grid_Y_test: Target grids [T, H, W, C]
        pre_seq_len: Number of time steps for input sequence
        aft_seq_len: Number of time steps for target prediction
        batch_size: Batch size for DataLoader
        timestamps_train, timestamps_val, timestamps_test: Optional lists of datetime objects
        mask: Optional [H, W] mask to restrict loss/evaluation region
        num_workers: Number of worker processes for DataLoader
        pin_memory: Whether to pin memory (set to True when using CUDA)
        
    Returns:
        train_loader, val_loader, test_loader: DataLoaders for each split
    """
    # Print shapes before creating datasets
    print(f"\n📊 Data shapes:")
    print(f"  Training: X={grid_X_train.shape}, Y={grid_Y_train.shape}")
    print(f"  Validation: X={grid_X_val.shape}, Y={grid_Y_val.shape}")
    print(f"  Test: X={grid_X_test.shape}, Y={grid_Y_test.shape}")
    if mask is not None:
        print(f"  Mask shape: {mask.shape}")
    
    # Check for NaN values
    train_x_nans = np.isnan(grid_X_train).sum()
    train_y_nans = np.isnan(grid_Y_train).sum()
    if train_x_nans > 0 or train_y_nans > 0:
        print(f"⚠️ Found NaN values - X: {train_x_nans}, Y: {train_y_nans}")
        print("  NaNs will be replaced with zeros during dataset creation")
    
    # Create datasets
    dataset_train = GriddedSeq2SeqDataset(
        grid_X_train, grid_Y_train,
        pre_seq_len, aft_seq_len,
        timestamps=timestamps_train,
        mask=mask
    )

    dataset_val = GriddedSeq2SeqDataset(
        grid_X_val, grid_Y_val,
        pre_seq_len, aft_seq_len,
        timestamps=timestamps_val,
        mask=mask
    )

    dataset_test = GriddedSeq2SeqDataset(
        grid_X_test, grid_Y_test,
        pre_seq_len, aft_seq_len,
        timestamps=timestamps_test,
        mask=mask
    )

    # Create DataLoaders with optimal settings based on data size
    train_loader = DataLoader(
        dataset_train, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=pin_memory and torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )

    val_loader = DataLoader(
        dataset_val, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_memory and torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )

    test_loader = DataLoader(
        dataset_test, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_memory and torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )
    
    print(f"\n📊 Created DataLoaders:")
    print(f"  Training: {len(dataset_train)} samples, {len(train_loader)} batches")
    print(f"  Validation: {len(dataset_val)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(dataset_test)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader