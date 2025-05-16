# train/trainer_utils.py
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

def get_trainer(
    output_dir="logs",
    log_name="ConvLSTM",
    monitor_metric="val_loss",
    mode="min",
    max_epochs=50,
    patience=10
):
    """
    Creates a PyTorch Lightning Trainer with common callbacks and loggers.
    Logs to TensorBoard and CSV.
    """
    logger_tb = TensorBoardLogger(save_dir=output_dir, name=log_name)
    logger_csv = CSVLogger(save_dir=output_dir, name=log_name)

    checkpoint_cb = ModelCheckpoint(
        monitor=monitor_metric,
        save_top_k=1,
        mode=mode,
        filename="best-checkpoint"
    )

    early_stop_cb = EarlyStopping(
        monitor=monitor_metric,
        patience=patience,
        mode=mode
    )

    trainer = pl.Trainer(
        logger=[logger_tb, logger_csv],
        callbacks=[checkpoint_cb, early_stop_cb],
        max_epochs=max_epochs,
        accelerator="auto",
         gradient_clip_val=1.0,
        devices=1
    )

    return trainer
