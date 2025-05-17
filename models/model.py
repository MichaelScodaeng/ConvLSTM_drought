import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import gc

def weights_init(m):
    """
    More conservative initialization for model weights to help with training.
    Apply this to model with: model.apply(weights_init)
    """
    if isinstance(m, nn.Conv2d):
        # Use more conservative Kaiming initialization for Conv2d layers
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu', a=0.1)
        if m.bias is not None:
            # Initialize bias with zeros to prevent instability
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.LSTM) or isinstance(m, nn.LSTMCell):
        # More conservative LSTM initialization
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param, gain=0.5)  # Lower gain for stability
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param, gain=0.5)  # Lower gain for stability
            elif 'bias' in name:
                nn.init.zeros_(param)  # Start with zeros

# Add this method to EncodingForecastingConvLSTM
def initialize_skip_connections(self):
    """Initialize skip connections with small weights to prevent large initial outputs"""
    nn.init.normal_(self.skip_conv.weight, mean=0.0, std=0.01)
    if self.skip_conv.bias is not None:
        nn.init.zeros_(self.skip_conv.bias)
    
    nn.init.normal_(self.conv_last.weight, mean=0.0, std=0.01)
    if self.conv_last.bias is not None:
        nn.init.zeros_(self.conv_last.bias)

# Add this to the LightningConvLSTMModule class
def configure_optimizers(self):
    # Use AdamW with more stable settings
    optimizer = torch.optim.AdamW(
        self.parameters(), 
        lr=self.lr / 10,  # Start with lower learning rate
        weight_decay=self.weight_decay,
        eps=1e-8  # Larger epsilon for stability
    )
    
    # Calculate number of training steps
    if self.trainer is not None and hasattr(self.trainer, 'estimated_stepping_batches'):
        steps = self.trainer.estimated_stepping_batches
    else:
        steps = 1000  # Default fallback
    
    # Use OneCycleLR scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=self.lr,
        total_steps=steps,
        pct_start=0.3,  # 30% warmup
        div_factor=25,  # initial_lr = max_lr/25
        final_div_factor=1000,  # final_lr = max_lr/1000
        anneal_strategy='cos'
    )
    
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step"
        }
    }

# Add this to your training code
def train_with_gradient_clipping(self):
    self.trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=10)],
        gradient_clip_val=1.0,  # Add gradient clipping
        gradient_clip_algorithm="norm"
    )

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
    X_min= -6.762784
    X_max= 3.886915
    # Ensure X_min and X_max are tensors
    if not isinstance(X_min, torch.Tensor):
        X_min = torch.tensor(X_min, dtype=scaled_data.dtype, device=scaled_data.device)
    if not isinstance(X_max, torch.Tensor):
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


def weights_init(m):
    """
    Custom initialization for model weights to help with training.
    Apply this to model with: model.apply(weights_init)
    """
    if isinstance(m, nn.Conv2d):
        # Use Kaiming initialization for Conv2d layers
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            # Initialize bias with small positive values to prevent dead neurons
            nn.init.constant_(m.bias, 0.01)
    elif isinstance(m, nn.LSTM) or isinstance(m, nn.LSTMCell):
        # Special initialization for LSTM weights
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)


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
        
        # Apply custom initialization
        self.reset_parameters()
        
    def reset_parameters(self):
        # Custom initialization for the conv layer
        nn.init.xavier_uniform_(self.conv.weight)
        if self.bias:
            nn.init.zeros_(self.conv.bias)

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
        
        # Use BatchNorm to help with training stability
        self.batchnorm = nn.ModuleList([
            nn.BatchNorm2d(hidden_dim[i]) for i in range(self.num_layers)
        ])
        
        self.dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()
        
        # Initialize hidden state if not provided
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
                
                # Apply BatchNorm and dropout for training stability
                if self.training:
                    h = self.batchnorm[layer_idx](h)
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
        
        # Add skip connection for better gradient flow
        self.skip_conv = nn.Conv2d(
            in_channels=input_dim,
            out_channels=1,
            kernel_size=1
        )

        self.conv_last = nn.Conv2d(
            in_channels=hidden_dim[-1], 
            out_channels=1, 
            kernel_size=1
        )

        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        
        # Apply weight initialization
        self.apply(weights_init)

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: 5-D Tensor of shape [b, t, c, h, w]
        Returns:
            output tensor of shape [b, t_out, 1, h, w]
        """
        assert input_tensor.dim() == 5, f"Expected input_tensor to be 5D [B, T, C, H, W], got {input_tensor.shape}"

        # Save first input frame for skip connection
        first_input = input_tensor[:, 0]
        
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
            
            # Apply skip connection from first input frame
            pred = self.conv_last(hidden_state) + self.skip_conv(first_input)
            
            # Add tanh activation to constrain output to [-1, 1] range
            pred = torch.tanh(pred)
            
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
        x, y = batch  # x: [B, T_in, C, H, W], y: [B, T_out, C, H, W]
        # Note: We've removed mask from the batch unpacking

        # Replace NaNs to prevent propagation
        x = torch.nan_to_num(x, nan=0.0)
        y = torch.nan_to_num(y, nan=0.0)

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

            # Compute loss directly without masking
            # For original space (for metrics)
            loss_original = torch.mean((y_hat_original - y_original) ** 2)

            # Also compute loss in SCALED space (for backprop stability)
            loss_scaled = torch.mean((y_hat - y) ** 2)

            # We'll log the original loss but use the scaled loss for backprop
            self.log(f"{stage}_loss", loss_original, on_step=False, on_epoch=True)
            self.log(f"{stage}_scaled_loss", loss_scaled, on_step=False, on_epoch=True)

            # Update metrics with all predictions (no masking)
            rmse_metric = getattr(self, f"{stage}_rmse")
            r2_metric = getattr(self, f"{stage}_r2")
            
            rmse_metric.update(y_hat_original, y_original)
            r2_metric.update(y_hat_original, y_original)

            # Log sample stats for first batch of training
            if self.global_step == 0 and stage == "train":
                print("\n📊 Sample stats:")
                print(f"x shape: {x.shape}, y shape: {y.shape}")
                print(f"y_hat shape: {y_hat.shape}")
                print(f"x min/max: {x.min().item():.4f}/{x.max().item():.4f}")
                print(f"y min/max: {y.min().item():.4f}/{y.max().item():.4f}")
                print(f"y_hat min/max: {y_hat.min().item():.4f}/{y_hat.max().item():.4f}")
                print(f"y_original min/max: {y_original.min().item():.4f}/{y_original.max().item():.4f}")
                print(f"y_hat_original min/max: {y_hat_original.min().item():.4f}/{y_hat_original.max().item():.4f}")
                print(f"Original scale loss: {loss_original.item():.6f}")
                print(f"Scaled loss: {loss_scaled.item():.6f}")

            # Log metrics
            self.log(f"{stage}_rmse", rmse_metric.compute(), prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{stage}_r2", r2_metric.compute(), prog_bar=True, on_step=False, on_epoch=True)

        # Clear memory
        if stage != "train":
            torch.cuda.empty_cache()
            gc.collect()

        # Return the SCALED loss for backprop stability (not the original loss)
        return loss_scaled if stage == "train" else None

        # Clear memory
        if stage != "train":
            torch.cuda.empty_cache()
            gc.collect()

        # Return the SCALED loss for backprop stability (not the original loss)
        return loss_scaled if stage == "train" else None

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "train")
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        # Use OneCycleLR for better convergence
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # Calculate number of training steps
        if self.trainer is not None and hasattr(self.trainer, 'estimated_stepping_batches'):
            steps = self.trainer.estimated_stepping_batches
        else:
            steps = 1000  # Default fallback
        
        # Use OneCycleLR scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=steps,
            pct_start=0.3,  # 30% warmup
            div_factor=25,  # initial_lr = max_lr/25
            final_div_factor=1000,  # final_lr = max_lr/1000
            anneal_strategy='cos'
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    def on_epoch_end(self):
        # Print current metrics at the end of each epoch
        print(f"\nEpoch {self.current_epoch} completed")
        print(f"Train RMSE: {self.train_rmse.compute():.4f}, R²: {self.train_r2.compute():.4f}")
        print(f"Val RMSE: {self.val_rmse.compute():.4f}, R²: {self.val_r2.compute():.4f}")
        print(f"Learning rate: {self.optimizers().param_groups[0]['lr']:.6f}")