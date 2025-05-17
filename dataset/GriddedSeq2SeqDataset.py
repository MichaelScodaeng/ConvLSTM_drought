import torch
from torch.utils.data import Dataset
import numpy as np

class GriddedSeq2SeqDataset(Dataset):
    def __init__(self, grid_X, grid_Y, pre_seq_len, aft_seq_len, timestamps=None, mask=None):
        """
        Args:
            grid_X: [T, H, W, F] input features
            grid_Y: [T, H, W, C] target values
            pre_seq_len: number of time steps for input sequence
            aft_seq_len: number of time steps for target prediction
            timestamps: optional list of datetime objects for each time step
            mask: optional [H, W] mask to restrict loss/evaluation region
        """
        assert grid_X.shape[0] == grid_Y.shape[0], "Time dimension mismatch"
        self.grid_X = grid_X
        self.grid_Y = grid_Y
        self.pre_seq_len = pre_seq_len
        self.aft_seq_len = aft_seq_len
        self.timestamps = timestamps
        
        # Process mask to ensure right shape
        if mask is not None:
            if len(mask.shape) == 2:
                # Ensure [H, W] -> [1, H, W]
                self.mask = mask.reshape(1, *mask.shape)
            elif len(mask.shape) == 3 and mask.shape[0] > 1:
                # If temporal mask, take first frame
                print(f"Warning: Mask has shape {mask.shape}, using first frame only")
                self.mask = mask[0:1]
            else:
                self.mask = mask
        else:
            self.mask = None

        self.num_samples = grid_X.shape[0] - pre_seq_len - aft_seq_len + 1
        assert self.num_samples > 0, "Not enough time steps to create samples"
        
        # Print dataset info
        print(f"Created dataset with {self.num_samples} samples:")
        print(f"  Input shape: {grid_X.shape}, Output shape: {grid_Y.shape}")
        if self.mask is not None:
            mask_coverage = np.mean(self.mask) * 100
            print(f"  Mask coverage: {mask_coverage:.2f}% of grid points")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x_seq = self.grid_X[idx : idx + self.pre_seq_len]         # [T, H, W, F]
        y_seq = self.grid_Y[idx + self.pre_seq_len : idx + self.pre_seq_len + self.aft_seq_len]  # [T, H, W, C]

        # Clean NaNs at dataset level before converting to tensor
        x_seq = np.nan_to_num(x_seq, nan=0.0, posinf=1e5, neginf=-1e5)
        y_seq = np.nan_to_num(y_seq, nan=0.0, posinf=1e5, neginf=-1e5)

        # Convert to torch tensors with channel dimension first
        x_seq = torch.tensor(np.transpose(x_seq, (0, 3, 1, 2)), dtype=torch.float32)  # [T, F, H, W]
        y_seq = torch.tensor(np.transpose(y_seq, (0, 3, 1, 2)), dtype=torch.float32)  # [T, C, H, W]
        
        # Process mask
        if self.mask is not None:
            mask_tensor = torch.tensor(self.mask, dtype=torch.float32)  # Already [1, H, W]
        else:
            # Create default mask of all ones if none provided
            H, W = self.grid_X.shape[1:3]
            mask_tensor = torch.ones(1, H, W, dtype=torch.float32)
          
        return x_seq, y_seq, mask_tensor