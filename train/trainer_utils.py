import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

def get_trainer(
    max_epochs=100,
    patience=10,
    monitor="val_loss",
    output_dir="outputs",
    log_every_n_steps=50
):
    """
    Configure and return a PyTorch Lightning Trainer with appropriate callbacks.
    
    Args:
        max_epochs: Maximum number of training epochs
        patience: Number of epochs to wait for improvement before early stopping
        monitor: Metric to monitor for early stopping and model checkpointing
        output_dir: Directory to save outputs
        log_every_n_steps: How often to log metrics (batches)
        
    Returns:
        PyTorch Lightning Trainer
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    # Set up callbacks
    callbacks = []
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor=monitor,
        min_delta=0.00,
        patience=patience,
        verbose=True,
        mode="min"
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        verbose=True,
        monitor=monitor,
        mode="min",
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, "logs"),
        name="convlstm"
    )
    
    # Create trainer with correct arguments based on PyTorch Lightning version
    try:
        # Try importing version from pytorch_lightning
        pl_version = pl.__version__
    except (AttributeError, ImportError):
        # Fallback if version not available directly
        try:
            import pkg_resources
            pl_version = pkg_resources.get_distribution("pytorch-lightning").version
        except:
            # If all else fails, assume older version
            pl_version = "1.0.0"  
    
    # Configure trainer based on PyTorch Lightning version
    if pl_version >= "2.0.0":
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            precision="16-mixed" if torch.cuda.is_available() else "32-true",
            accelerator="auto",
            devices="auto"
        )
    else:
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            gpus=-1 if torch.cuda.is_available() else None,
            precision=16 if torch.cuda.is_available() else 32
        )
    
    return trainer