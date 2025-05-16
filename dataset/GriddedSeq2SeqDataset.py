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
        self.mask = mask  # Optional [H, W] mask for evaluation or filtering

        self.num_samples = grid_X.shape[0] - pre_seq_len - aft_seq_len + 1
        assert self.num_samples > 0, "Not enough time steps to create samples"

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x_seq = self.grid_X[idx : idx + self.pre_seq_len]         # [T, H, W, F]
        y_seq = self.grid_Y[idx + self.pre_seq_len : idx + self.pre_seq_len + self.aft_seq_len]  # [T, H, W, C]

        # ✅ Clean NaNs at dataset level before converting to tensor
        x_seq = np.nan_to_num(x_seq, nan=0.0, posinf=1e5, neginf=-1e5)
        y_seq = np.nan_to_num(y_seq, nan=0.0, posinf=1e5, neginf=-1e5)

        x_seq = torch.tensor(np.transpose(x_seq, (0, 3, 1, 2)), dtype=torch.float32)  # [T, F, H, W]
        y_seq = torch.tensor(np.transpose(y_seq, (0, 3, 1, 2)), dtype=torch.float32)  # [T, C, H, W]

        output = (x_seq, y_seq)

        if self.mask is not None:
            mask_tensor = torch.tensor(self.mask, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
            output += (mask_tensor,)
        elif self.timestamps is not None:
            time_idx = idx + self.pre_seq_len
            output += (self.timestamps[time_idx],)

        return output
