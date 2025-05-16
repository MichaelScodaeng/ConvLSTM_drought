# optuna_tuner.py
import optuna
import numpy as np
import yaml
import joblib
from train.trainer_utils import get_trainer
from models.model import LightningConvLSTMModule, EncodingForecastingConvLSTM
from loaders.loader import create_dataloaders

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def objective(trial):
    base_config = load_config("config/config.yaml")
    target_name = "_".join(base_config["target_cols"])
    data = np.load(f"data/preprocessed/{target_name}/data_{target_name}.npz", allow_pickle=True)

    # Load tensors
    grid_X_train = data["grid_X_train_scaled"]
    grid_Y_train = data["grid_Y_train_scaled"]
    grid_X_val = data["grid_X_val_scaled"]
    grid_Y_val = data["grid_Y_val_scaled"]
    grid_X_test = data["grid_X_test_scaled"]
    grid_Y_test = data["grid_Y_test_scaled"]
    mask = data["mask"]
    timestamps_train = data["timestamps_train"].tolist()
    timestamps_val = data["timestamps_val"].tolist()
    timestamps_test = data["timestamps_test"].tolist()

    # Hyperparameters to tune
    config = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 32, 64]),
        "kernel_size": trial.suggest_categorical("kernel_size", [(3, 3), (5, 5)]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.2,step=0.1),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "lr": trial.suggest_float("lr", 1e-6, 1e-5, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "epochs": trial.suggest_int("epochs", 2, 3,step=1),
        "pre_seq_length": trial.suggest_int("pre_seq_length", 10, 30,step=5),
        "aft_seq_length": trial.suggest_int("aft_seq_length", 1,2,step=5),
    }

    # Create loaders
    train_loader, val_loader, _ = create_dataloaders(
        grid_X_train, grid_Y_train,
        grid_X_val, grid_Y_val,
        grid_X_test, grid_Y_test,
        pre_seq_len=config["pre_seq_length"],
        aft_seq_len=config["aft_seq_length"],
        batch_size=config["batch_size"],
        timestamps_train=timestamps_train,
        timestamps_val=timestamps_val,
        timestamps_test=timestamps_test,
        mask=mask
    )
    #Check shapes of tensors
    print("grid_X_train_shape: ", grid_X_train.shape)
    print("grid_Y_train_shape: ", grid_Y_train.shape)
    print("grid_X_val_shape: ", grid_X_val.shape)
    print("grid_Y_val_shape: ", grid_Y_val.shape)
    print("grid_X_test_shape: ", grid_X_test.shape)
    print("grid_Y_test_shape: ", grid_Y_test.shape)
    print("timestamps_train_shape: ", len(timestamps_train))
    print("timestamps_val_shape: ", len(timestamps_val))
    print("timestamps_test_shape: ", len(timestamps_test))
    print("mask_shape: ", mask.shape)
    #print(mask)
    # Check loaders
    for batch in train_loader:
        x, y, mask = batch
        print("x shape: ", x.shape)
        print("y shape: ", y.shape)
        print("mask batch shape: ", mask.shape)
        break


    model = EncodingForecastingConvLSTM(
        input_dim=grid_X_train.shape[-1],
        hidden_dim=config["hidden_dim"],
        kernel_size=config["kernel_size"],
        dropout=config["dropout"],
        num_layers=config["num_layers"],
        pre_seq_length=config["pre_seq_length"],
        aft_seq_length=config["aft_seq_length"]
    )

    lightning_model = LightningConvLSTMModule(model, lr=config["lr"], weight_decay=config["weight_decay"])
    trainer = get_trainer(max_epochs=config["epochs"])
    trainer.fit(lightning_model, train_loader, val_loader)

    val_loss = trainer.callback_metrics["val_loss"].item()
    trial.set_user_attr("config", config)
    return val_loss

def run_study(n_trials=30):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:", study.best_trial)
    best_config = study.best_trial.user_attrs["config"]
    if isinstance(best_config["kernel_size"], tuple):
        best_config["kernel_size"] = list(best_config["kernel_size"])
    original_config = load_config("config/config.yaml")
    target_name = "_".join(original_config["target_cols"])

    for k in ["feature_cols", "target_cols", "lat_key", "lon_key", "time_key", "scale_range"]:
        best_config[k] = original_config[k]
    with open(f"config/best_config_{target_name}.yaml", "w") as f:
        yaml.dump(best_config, f)
    

    print("✅ Best config saved to config/best_config.yaml")
    return study

if __name__ == "__main__":
    run_study(n_trials=2)
