import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import gc


def inverse_transform(scaled_data, X_min, X_max):
    """
    Inverse transform data from scaled [-1,1] range back to original range.
    
    Args:
        scaled_data: Tensor with scaled values
        X_min: Minimum value in original scale
        X_max: Maximum value in original scale
    
    Returns:
        Tensor with values in original scale
    """
    X_min =  [-6.762784]
    X_min = torch.tensor(X_min, dtype=scaled_data.dtype, device=scaled_data.device)
    X_max = [3.886915]
    X_max = torch.tensor(X_max, dtype=scaled_data.dtype, device=scaled_data.device)
    X_min_tensor = torch.full_like(scaled_data, X_min.item(), dtype=scaled_data.dtype, device=scaled_data.device)
    X_max_tensor = torch.full_like(scaled_data, X_max.item(), dtype=scaled_data.dtype, device=scaled_data.device)
    # Perform the inverse transformation
    inv_data = torch.add(
        torch.mul(
            scaled_data, 
            torch.div(torch.sub(X_max_tensor, X_min_tensor), 2)
        ),
        torch.div(torch.add(X_max_tensor, X_min_tensor), 2)
    )
    return inv_data


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False, dropout=0.0):
        super().__init__()
        self._check_kernel_size_consistency(kernel_size)

        # Properly handle kernel_size and hidden_dim for multi-layer setup
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.dropout = dropout

        self.cell_list = nn.ModuleList([
            ConvLSTMCell(input_dim=self.input_dim if i == 0 else hidden_dim[i - 1],
                         hidden_dim=hidden_dim[i],
                         kernel_size=kernel_size[i],
                         bias=self.bias)
            for i in range(self.num_layers)
        ])
        self.dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()
        
        # Initialize hidden state if not provided or incorrect size
        if hidden_state is None:
            hidden_state = self._init_hidden(b, (h, w))
        
        # Ensure hidden_state has the correct number of layers
        if len(hidden_state) != self.num_layers:
            hidden_state = self._init_hidden(b, (h, w))

        layer_output_list = []
        last_state_list = []
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(input_tensor.size(1)):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t], cur_state=[h, c])
                h = self.dropout_layer(h)
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        return [self.cell_list[i].init_hidden(batch_size, image_size) for i in range(self.num_layers)]

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or (isinstance(kernel_size, list) and all(isinstance(k, tuple) for k in kernel_size))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class EncodingForecastingConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 pre_seq_length, aft_seq_length, dropout=0.0, batch_first=True):
        super().__init__()

        # Proper handling of hidden_dim and kernel_size
        if not isinstance(hidden_dim, list):
            hidden_dim = [hidden_dim] * num_layers
            
        # Handle different kernel_size formats
        if isinstance(kernel_size, list) and all(isinstance(k, int) for k in kernel_size):
            # kernel_size: [3, 3] → [(3, 3), (3, 3), ...]
            kernel_size = [tuple(kernel_size)] * num_layers
        elif isinstance(kernel_size, tuple):
            # kernel_size: (3, 3) → [(3, 3), (3, 3), (3, 3)]
            kernel_size = [kernel_size] * num_layers
        elif isinstance(kernel_size, list) and all(isinstance(k, list) and len(k) == 2 for k in kernel_size):
            # kernel_size: [[3, 3], [3, 3], ...] → [(3, 3), (3, 3), ...]
            kernel_size = [tuple(k) for k in kernel_size]
        
        # Initialize encoder and forecaster
        self.encoder = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=batch_first,
            return_all_layers=False,
            dropout=dropout
        )

        self.forecaster = ConvLSTM(
            input_dim=hidden_dim[-1],
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=batch_first,
            return_all_layers=False,
            dropout=dropout
        )

        self.conv_last = nn.Conv2d(in_channels=hidden_dim[-1], out_channels=1, kernel_size=1)

        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: 5-D Tensor of shape [b, t, c, h, w]
        Returns:
            output tensor of shape [b, t_out, 1, h, w]
        """
        assert input_tensor.dim() == 5, f"Expected input_tensor to be 5D [B, T, C, H, W], got {input_tensor.shape}"

        # Encoder phase
        encoder_outputs, encoder_states = self.encoder(input_tensor)

        # Verify encoder states
        if not encoder_states or not encoder_states[-1]:
            raise RuntimeError("Encoder returned empty states. Check input shape or model definition.")

        # Initialize forecaster with encoder states
        forecaster_states = encoder_states
        
        # First input to forecaster is the last hidden state from encoder
        next_input = encoder_states[-1][0].unsqueeze(1)  # Shape: [B, 1, hidden_dim, H, W]

        # Generate predictions one time step at a time
        predictions = []
        for t in range(self.aft_seq_length):
            forecaster_output, forecaster_states = self.forecaster(next_input, forecaster_states)
            hidden_state = forecaster_output[-1][:, -1]  # [B, hidden_dim, H, W]
            pred = self.conv_last(hidden_state)  # [B, 1, H, W]
            predictions.append(pred)
            
            # Use latest prediction as next input
            next_input = hidden_state.unsqueeze(1)  # Add time dimension

        # Stack along time dimension
        return torch.stack(predictions, dim=1)  # [B, T_out, 1, H, W]


class LightningConvLSTMModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3, weight_decay=0.0, X_min=None, X_max=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.MSELoss()
        self.save_hyperparameters(ignore=["model"])

        # Store min and max values for inverse transformation
        self.register_buffer("X_min", torch.tensor(X_min) if X_min is not None else torch.tensor(0.0))
        self.register_buffer("X_max", torch.tensor(X_max) if X_max is not None else torch.tensor(1.0))

        # Metrics
        self.train_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.test_rmse = torchmetrics.MeanSquaredError(squared=False)
        
        # Additional metrics for more comprehensive evaluation
        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()
        self.test_r2 = torchmetrics.R2Score()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        x, y, mask = batch  # x: [B, T_in, C, H, W], y: [B, T_out, C, H, W], mask: [B, 1, H, W]

        # Replace NaNs to prevent propagation
        x = torch.nan_to_num(x, nan=0.0)
        y = torch.nan_to_num(y, nan=0.0)
        mask = torch.nan_to_num(mask, nan=0.0)

        # Forward pass
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            y_hat = self(x)  # [B, T_out, 1, H, W]

            # Check for NaNs in output
            if torch.isnan(y_hat).any():
                print(f"[{stage}] ❌ Model output contains NaNs — zeroing out this batch.")
                return torch.tensor(0.0, requires_grad=True, device=self.device)

            # Apply inverse transform to predictions and targets
            y_hat_original = inverse_transform(y_hat, self.X_min, self.X_max) 
            y_original = inverse_transform(y, self.X_min, self.X_max)

            # Expand mask to match prediction shape
            expanded_mask = mask.unsqueeze(1).expand_as(y_hat)  # [B, T, 1, H, W]
            valid_pixel_count = expanded_mask.sum()

            if valid_pixel_count == 0:
                print(f"[{stage}] ⚠️ Skipping batch with no valid mask.")
                return torch.tensor(0.0, requires_grad=True, device=self.device)

            # Compute masked loss on ORIGINAL scale
            masked_mse = ((y_hat_original - y_original) ** 2) * expanded_mask
            loss = masked_mse.sum() / (valid_pixel_count + 1e-8)

            # Also compute metrics on scaled data for comparison
            scaled_masked_mse = ((y_hat - y) ** 2) * expanded_mask
            scaled_loss = scaled_masked_mse.sum() / (valid_pixel_count + 1e-8)

            # Extract valid predictions and targets for metrics
            pred_valid = y_hat_original[expanded_mask.bool()].detach()
            true_valid = y_original[expanded_mask.bool()].detach()

            # Update metrics
            rmse_metric = getattr(self, f"{stage}_rmse")
            r2_metric = getattr(self, f"{stage}_r2")
            
            rmse_metric.update(pred_valid, true_valid)
            r2_metric.update(pred_valid, true_valid)

            # Log sample stats for first batch of training
            if self.global_step == 0 and stage == "train":
                print("\n📊 Sample stats:")
                print(f"x shape: {x.shape}, y shape: {y.shape}, mask shape: {mask.shape}")
                print(f"y_hat shape: {y_hat.shape}")
                print(f"x min/max: {x.min().item():.4f}/{x.max().item():.4f}")
                print(f"y min/max: {y.min().item():.4f}/{y.max().item():.4f}")
                print(f"y_hat min/max: {y_hat.min().item():.4f}/{y_hat.max().item():.4f}")
                print(f"y_original min/max: {y_original.min().item():.4f}/{y_original.max().item():.4f}")
                print(f"y_hat_original min/max: {y_hat_original.min().item():.4f}/{y_hat_original.max().item():.4f}")
                print(f"mask valid %: {mask.sum().item() / mask.numel() * 100:.2f}%")
                print(f"Original scale loss: {loss.item():.6f}")
                print(f"Scaled loss: {scaled_loss.item():.6f}")

            # Log metrics
            self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True)
            self.log(f"{stage}_rmse", rmse_metric.compute(), prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{stage}_r2", r2_metric.compute(), prog_bar=True, on_step=False, on_epoch=True)
            
            # Additional logging
            self.log(f"{stage}_scaled_loss", scaled_loss, on_step=False, on_epoch=True)

        # Clear memory
        if stage != "train":
            torch.cuda.empty_cache()
            gc.collect()

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "train")
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            patience=5, 
            factor=0.5,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }

    def on_epoch_end(self):
        # Print current metrics at the end of each epoch
        print(f"\nEpoch {self.current_epoch} completed")
        print(f"Train RMSE: {self.train_rmse.compute():.4f}, R²: {self.train_r2.compute():.4f}")
        print(f"Val RMSE: {self.val_rmse.compute():.4f}, R²: {self.val_r2.compute():.4f}")
        print(f"Learning rate: {self.optimizers().param_groups[0]['lr']:.6f}")