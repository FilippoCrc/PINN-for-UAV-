# pinn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- QuadrotorPINN class remains unchanged ---
class QuadrotorPINN(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=32, num_layers=3, output_dim=4):
        super(QuadrotorPINN, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.batch_norm_input = nn.BatchNorm1d(hidden_dim)
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout = nn.Dropout(0.2) # Dropout layer with 20% dropout rate
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
        x = self.dropout(x) # Apply dropout if needed
        for layer, bn in zip(self.hidden_layers, self.batch_norms):
            x = layer(x)
            if x.shape[0] > 1:
                x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
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
        mean = self.input_mean.to(device)
        scale = self.input_scale.to(device)
        predictions_unscaled = predictions_scaled * scale + mean  # Gradients flow here
        return predictions_unscaled[:, 1:4]  # Extract angular controls

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
        # print("Time steps:", physics_info_batch[:, 0])
        # print("Angular velocities:", physics_info_batch[:, 1:4])
        current_device = predictions_scaled.device
        batch_size = predictions_scaled.shape[0]

        if batch_size < 2:
            mse_loss = F.mse_loss(predictions_scaled, targets_scaled)
            return mse_loss, mse_loss.item(), 0.0 # No physics loss possible

        thrust_pred = predictions_scaled[:-1, 0]  # Scaled thrust (B-1,)
        thrust_target = targets_scaled[:-1, 0]
        tau_pred = predictions_scaled[:-1, 1:4]   # Scaled tau (B-1, 3)
        tau_target = targets_scaled[:-1, 1:4]

        # Compute separate MSE losses
        mse_thrust = F.mse_loss(thrust_pred, thrust_target)
        mse_tau = F.mse_loss(tau_pred, tau_target)
        print("MSE thrust loss:", mse_thrust.item())
        print("MSE tau loss:", mse_tau.item())
        # Combine with optional weights (adjust weights empirically)
        mse_loss = mse_thrust + 1.0 * mse_tau  # Equal weights if scaled properly

        # 2. Local Monotonicity Physics Loss (LLM)
        # Unscale predicted angular controls/torques
        controls_angular_pred_unscaled = self._unscale_predictions(predictions_scaled, current_device)

        # # Check gradients
        # assert controls_angular_pred_unscaled.requires_grad, "Gradients not attached to controls_angular_pred_unscaled!"
        # print("Gradients for controls_angular_pred_unscaled:", controls_angular_pred_unscaled.grad)

        # Extract unscaled time and omega from physics_info
        t = physics_info_batch[:, 0]        # Shape (B,)
        omega = physics_info_batch[:, 1:4]  # Shape (B, 3)

        # --- Calculate Finite Differences for Angular Acceleration ---
        delta_t = t[1:] - t[:-1]                             # Shape (B-1,)
        delta_omega_actual = (omega[1:] - omega[:-1]) / delta_t.unsqueeze(-1)  # Shape (B-1, 3)

        # --- Get Predicted Controls (Exclude Last Sample to Match delta_omega) ---
        controls_angular_pred_unscaled = self._unscale_predictions(predictions_scaled, current_device)  # Shape (B, 3)
        controls_angular_pred_unscaled = controls_angular_pred_unscaled[:-1]  # Shape (B-1, 3) <-- FIX HERE

        # --- Calculate Sign Terms ---
        k_tanh = 1000.0  # Increased from 10.0 to sharpen sign
        sign_delta_omega = torch.tanh(k_tanh * delta_omega_actual)  # Shape (B-1, 3)
        sign_controls = torch.tanh(k_tanh * controls_angular_pred_unscaled)  # Shape (B-1, 3)

        # --- Compute Loss ---
        monotonicity_term = 0.5 * (1.0 - sign_delta_omega * sign_controls)  # Shape (B-1, 3)
        physics_loss = torch.mean(monotonicity_term)

        # 3. Get current lambda from annealing schedule
        lambda_physics = self._calculate_annealing_lambda(epoch)

        # 4. Total Loss
        total_loss = mse_loss + lambda_physics * physics_loss
        # print("Angular acceleration (mean):", delta_omega_actual.mean().item())
        # print("Control changes (mean):", delta_controls_angular_pred.mean().item())
        # print("Gradients for angular controls:", controls_angular_pred_unscaled.grad)
        # Detach losses for history logging
        return total_loss, mse_loss.item(), physics_loss.item()