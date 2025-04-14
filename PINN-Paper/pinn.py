# pinn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- QuadrotorPINN class remains unchanged ---
class QuadrotorPINN(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, num_layers=6, output_dim=4):
        super(QuadrotorPINN, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.batch_norm_input = nn.BatchNorm1d(hidden_dim)
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.input_layer.weight, gain=gain)
        nn.init.zeros_(self.input_layer.bias)
        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, state_input):
        x = self.input_layer(state_input)
        # Handle potential batch size 1 during evaluation/inference if BN used
        if x.shape[0] > 1:
            x = self.batch_norm_input(x)
        x = F.relu(x)
        for layer, bn in zip(self.hidden_layers, self.batch_norms):
            x = layer(x)
            if x.shape[0] > 1:
                x = bn(x)
            x = F.relu(x)
        control_output = self.output_layer(x)
        return control_output

# --- NEW LocalMonotonicityLoss Class (Based on Gu et al. paper) ---
class LocalMonotonicityLoss:
    def __init__(self,
                 input_scaler,
                 # Annealing parameters (based on Fig 3 / Eq 6 in paper)
                 use_annealing=True,
                 annealing_cycles=5, # M in paper
                 annealing_ratio=0.5, # R in paper
                 lambda_max=0.1, # Max physics weight (λ_max in paper)
                 total_epochs=None # Needed for annealing calculation
                 ):
        """
        Loss based on local monotonicity between angular acceleration changes
        and predicted control input changes (approximating Gu et al. LLM).

        Args:
            input_scaler: Fitted StandardScaler for control inputs (needed to unscale predictions).
            use_annealing (bool): Whether to use cyclical annealing for lambda.
            annealing_cycles (int): Number of cycles (M).
            annealing_ratio (float): Proportion of cycle to hold lambda_max (R).
            lambda_max (float): Maximum weight for the physics loss term.
            total_epochs (int): Total number of training epochs (required if use_annealing=True).
        """
        if input_scaler is None:
             raise ValueError("input_scaler must be provided for LocalMonotonicityLoss")
        self.input_scaler = input_scaler
        self.epsilon = 1e-8 # For safe division

        # Annealing parameters
        self.use_annealing = use_annealing
        if self.use_annealing and total_epochs is None:
            raise ValueError("total_epochs must be provided if use_annealing=True")
        self.annealing_cycles = annealing_cycles
        self.annealing_ratio = annealing_ratio
        self.lambda_max = lambda_max
        self.total_epochs = total_epochs

        # Precompute scaler attributes as tensors (do once)
        self.input_mean = torch.tensor(self.input_scaler.mean_, dtype=torch.float32)
        self.input_scale = torch.tensor(self.input_scaler.scale_, dtype=torch.float32)
        if self.input_scale.shape[0] != 4:
             print(f"Warning: Expected input_scaler for 4 outputs, but scale shape is {self.input_scale.shape}")

    def _unscale_predictions(self, predictions_scaled, device):
        """Unscale network output (thrust, torques)"""
        mean = self.input_mean.to(device)
        scale = self.input_scale.to(device)
        # Ensure broadcasting works if scale/mean are not exactly (4,)
        mean = mean.view(1, -1) # Shape (1, num_outputs)
        scale = scale.view(1, -1) # Shape (1, num_outputs)
        predictions_unscaled = predictions_scaled * scale + mean
        # Extract torques/angular controls (assuming indices 1, 2, 3)
        # Adjust indices if your control output order is different
        if predictions_unscaled.shape[1] >= 4:
            controls_angular_unscaled = predictions_unscaled[:, 1:4] # Shape: (B, 3)
        else:
            # Handle case where output dim < 4 - maybe only torque? Adjust as needed.
            print(f"Warning: predictions_unscaled shape {predictions_unscaled.shape} unexpected. Assuming last {min(3, predictions_unscaled.shape[1])} are angular controls.")
            controls_angular_unscaled = predictions_unscaled[:, -min(3, predictions_unscaled.shape[1]):]

        return controls_angular_unscaled

    def _calculate_annealing_lambda(self, epoch):
        """Calculates lambda_physics based on cyclical annealing schedule."""
        if not self.use_annealing:
            return self.lambda_max # Use fixed max lambda if annealing is off

        # Calculate T_M = period of one cycle
        epochs_per_cycle_float = self.total_epochs / self.annealing_cycles
        epochs_per_cycle = int(np.ceil(epochs_per_cycle_float)) # T/M rounded up

        # Calculate current position within the current cycle (0 to T_M - 1)
        current_cycle_epoch = epoch % epochs_per_cycle

        # Calculate beta (fraction through the cycle, 0 to 1)
        # Ensure epochs_per_cycle is not zero
        beta = current_cycle_epoch / max(1, epochs_per_cycle - 1) # Avoid division by zero for single epoch cycle

        # Calculate lambda based on Eq. 6 (inverse schedule)
        if beta < self.annealing_ratio: # Start high, decrease
            lambda_val = self.lambda_max
        else: # beta >= R: Anneal down
            # Fraction through the annealing down phase (0 to 1)
            annealing_phase_fraction = (beta - self.annealing_ratio) / (1.0 - self.annealing_ratio)
            lambda_val = self.lambda_max * (1.0 - annealing_phase_fraction)

        # Ensure lambda is non-negative
        lambda_val = max(0.0, lambda_val)

        # Debug print
        # print(f"Epoch {epoch}, Cycle Epoch {current_cycle_epoch}, Beta {beta:.3f}, Lambda {lambda_val:.4f}")

        return lambda_val

    def __call__(self, predictions_scaled, targets_scaled, physics_info_batch, epoch):
        """
        Calculates combined MSE + Local Monotonicity loss.

        Args:
            predictions_scaled (Tensor): Network output (scaled control inputs), shape (B, 4).
            targets_scaled (Tensor): True scaled control inputs, shape (B, 4).
            physics_info_batch (Tensor): Batch of unscaled [t, wx, wy, wz], shape (B, 4).
            epoch (int): Current epoch (used for annealing schedule).

        Returns:
            tuple: (total_loss, mse_loss_item, physics_loss_item)
        """
        current_device = predictions_scaled.device
        batch_size = predictions_scaled.shape[0]

        if batch_size < 2:
            mse_loss = F.mse_loss(predictions_scaled, targets_scaled)
            return mse_loss, mse_loss.item(), 0.0 # No physics loss possible

        # 1. MSE Loss (calculated on the first B-1 samples to match physics diff length)
        # We compare prediction at t with target at t.
        mse_loss = F.mse_loss(predictions_scaled[:-1], targets_scaled[:-1])

        # 2. Local Monotonicity Physics Loss (LLM)
        # Unscale predicted angular controls/torques
        controls_angular_pred_unscaled = self._unscale_predictions(predictions_scaled, current_device) # Shape (B, 3)

        # Extract unscaled time and omega from physics_info
        t = physics_info_batch[:, 0]        # Shape (B,)
        omega = physics_info_batch[:, 1:4]  # Shape (B, 3) unscaled angular velocities

        # --- Calculate Finite Differences for LLM ---
        # We need changes between t and t+1
        delta_t = t[1:] - t[:-1]                             # Shape (B-1,)
        delta_omega_actual = omega[1:] - omega[:-1]          # Shape (B-1, 3), Change in actual omega

        # Predicted controls at time t and t+1
        controls_angular_t = controls_angular_pred_unscaled[:-1] # Shape (B-1, 3)
        controls_angular_tplus1 = controls_angular_pred_unscaled[1:] # Shape (B-1, 3)
        delta_controls_angular_pred = controls_angular_tplus1 - controls_angular_t # Shape (B-1, 3)

        # --- Calculate LLM based on sign consistency ---
        # Use tanh as differentiable sign approximation: sign(x) ≈ tanh(k*x)
        # A high k makes it closer to sign, but can cause gradient issues. Start with k=1.
        k_tanh = 1.0

        # Compare signs component-wise (roll, pitch, yaw)
        sign_delta_omega = torch.tanh(k_tanh * delta_omega_actual) # Shape (B-1, 3)
        sign_delta_controls = torch.tanh(k_tanh * delta_controls_angular_pred) # Shape (B-1, 3)

        # Monotonicity loss: Penalize if signs don't match
        # Loss = 0.5 * (1 - sign(a)*sign(b)) -> 0 if signs match, 1 if they mismatch
        monotonicity_term = 0.5 * (1.0 - sign_delta_omega * sign_delta_controls) # Shape (B-1, 3)

        # Average over the 3 axes and the batch dimension
        physics_loss = torch.mean(monotonicity_term)

        # 3. Get current lambda from annealing schedule
        lambda_physics = self._calculate_annealing_lambda(epoch)

        # 4. Total Loss
        total_loss = mse_loss + lambda_physics * physics_loss

        # Detach losses for history logging
        return total_loss, mse_loss.item(), physics_loss.item()