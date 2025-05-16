from torch.utils.data import DataLoader
from dataset.GriddedSeq2SeqDataset import GriddedSeq2SeqDataset

def create_dataloaders(grid_X_train, grid_Y_train,
                        grid_X_val, grid_Y_val,
                        grid_X_test, grid_Y_test,
                        pre_seq_len, aft_seq_len,
                        batch_size=16, timestamps_train=None,
                        timestamps_val=None, timestamps_test=None,
                        mask=None, num_workers=2, pin_memory=True):
    """
    Create DataLoaders for train, val, test splits.
    """

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

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)

    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
