import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# QuadrotorPINN class remains the same as in the previous 'State -> Control Input' version
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
        x = self.batch_norm_input(x)
        x = F.relu(x)
        for layer, bn in zip(self.hidden_layers, self.batch_norms):
            x = layer(x)
            x = bn(x)
            x = F.relu(x)
        control_output = self.output_layer(x)
        return control_output

# --- NEW PhysicsInformedLoss_Control Class ---
class PhysicsInformedLoss_Control:
    def __init__(self, input_scaler, lambda_physics=1.0):
        """
        Loss for State -> Control Input prediction with physics regularization.

        Args:
            input_scaler: Fitted StandardScaler for control inputs (needed to unscale predictions).
            lambda_physics: Weight for the physics loss term.
        """
        if input_scaler is None:
             raise ValueError("input_scaler must be provided for PhysicsInformedLoss_Control")
        self.input_scaler = input_scaler
        self.lambda_physics = lambda_physics

        # Quadrotor physical parameters (ensure these match your system)
        self.m = 1.5 # Mass (kg) - Not directly used in rotational dynamics loss
        self.g = 9.81 # Gravity (m/s^2) - Not used here
        # Inertia Tensor J (kg*m^2) - IMPORTANT: Use correct values!
        self.J = torch.diag(torch.tensor([0.0146, 0.0168, 0.0309], dtype=torch.float32))
        # self.J = torch.diag(torch.tensor([0.0820, 0.0845, 0.1377])) # Example alternative

        # Precompute scaler attributes as tensors (do once)
        self.input_mean = torch.tensor(self.input_scaler.mean_, dtype=torch.float32)
        self.input_scale = torch.tensor(self.input_scaler.scale_, dtype=torch.float32)
        self.epsilon = 1e-8 # For safe division

    def _unscale_predictions(self, predictions_scaled, device):
        """Unscale network output (thrust, torques)"""
        mean = self.input_mean.to(device)
        scale = self.input_scale.to(device)
        # predictions_scaled shape: (B, 4)
        # mean/scale shape: (4,)
        predictions_unscaled = predictions_scaled * scale.unsqueeze(0) + mean.unsqueeze(0)
        # Extract torques (assuming indices 1, 2, 3)
        tau_pred_unscaled = predictions_unscaled[:, 1:] # Shape: (B, 3)
        return tau_pred_unscaled

    def __call__(self, predictions_scaled, targets_scaled, physics_info_batch, epoch=0):
        """
        Calculates combined MSE + Physics loss.

        Args:
            predictions_scaled (Tensor): Network output (scaled control inputs), shape (B, 4).
            targets_scaled (Tensor): True scaled control inputs, shape (B, 4).
            physics_info_batch (Tensor): Batch of unscaled [t, wx, wy, wz], shape (B, 4).
            epoch (int): Current epoch (optional, not used here).

        Returns:
            tuple: (total_loss, mse_loss_item, physics_loss_item)
        """
        current_device = predictions_scaled.device
        batch_size = predictions_scaled.shape[0]

        if batch_size < 2:
            # Cannot compute finite difference, return only MSE on the single sample if B=1
            # Or handle appropriately (e.g., return 0 physics loss, warn)
            # Let's calculate MSE on the available samples
            mse_loss = F.mse_loss(predictions_scaled, targets_scaled)
            return mse_loss, mse_loss.item(), 0.0 # No physics loss possible

        # 1. MSE Loss (Calculated on the first B-1 samples to match physics loss length)
        mse_loss = F.mse_loss(predictions_scaled[:-1], targets_scaled[:-1])

        # 2. Physics Loss Calculation
        # Unscale predicted torques
        tau_pred_unscaled = self._unscale_predictions(predictions_scaled, current_device) # Shape (B, 3)

        # Extract unscaled time and omega from physics_info
        t = physics_info_batch[:, 0]        # Shape (B,)
        omega = physics_info_batch[:, 1:4]  # Shape (B, 3)

        # Move inertia tensor to correct device
        J_dev = self.J.to(current_device)

        # Calculate finite differences (on B-1 pairs)
        delta_t = t[1:] - t[:-1]                  # Shape (B-1,)
        delta_omega = omega[1:] - omega[:-1]      # Shape (B-1, 3)

        # Avoid division by zero if delta_t is too small (e.g., duplicate data points)
        safe_delta_t = delta_t.unsqueeze(1) + self.epsilon # Shape (B-1, 1)
        omega_dot = delta_omega / safe_delta_t             # Shape (B-1, 3)

        # Get omega and predicted tau at the start of each interval (time t)
        omega_t = omega[:-1]              # Shape (B-1, 3)
        tau_pred_t = tau_pred_unscaled[:-1] # Shape (B-1, 3)

        # Calculate physics-based torque: J*omega_dot + omega x (J*omega)
        # Use torch.einsum for clarity with matrix-vector products per batch element
        J_omega_dot = torch.einsum('ij,bj->bi', J_dev, omega_dot) # Shape (B-1, 3)
        J_omega = torch.einsum('ij,bj->bi', J_dev, omega_t)       # Shape (B-1, 3)
        cross_term = torch.cross(omega_t, J_omega, dim=1)         # Shape (B-1, 3)

        tau_physics = J_omega_dot + cross_term # Shape (B-1, 3)

        # Calculate the physics residual: tau_pred(t) - tau_physics(t)
        physics_residual = tau_pred_t - tau_physics # Shape (B-1, 3)
        physics_loss = torch.mean(physics_residual ** 2)

        # 3. Total Loss
        total_loss = mse_loss + self.lambda_physics * physics_loss

        # Detach losses for history logging
        return total_loss, mse_loss.item(), physics_loss.item()