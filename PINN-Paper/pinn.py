import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import chi2

class QuadrotorPINN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=45, num_layers=10, output_dim=12):
        """
        Physics-Informed Neural Network for quadrotor dynamics modeling.
        
        Args:
            input_dim: Dimension of input vector (thrust and torque)
            hidden_dim: Number of neurons in each hidden layer
            num_layers: Number of hidden layers
            output_dim: Dimension of output state vector
        """
        super(QuadrotorPINN, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.batch_norm_input = nn.BatchNorm1d(hidden_dim)
        
        # Create hidden layers with batch normalization
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for layer in [self.input_layer] + list(self.hidden_layers) + [self.output_layer]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Predicted state vector
        """
        # Input layer with batch normalization and ReLU
        x = self.input_layer(x)
        x = self.batch_norm_input(x)
        x = F.relu(x)
        
        # Hidden layers with batch normalization and ReLU
        for layer, batch_norm in zip(self.hidden_layers, self.batch_norms):
            x = layer(x)
            x = batch_norm(x)
            x = F.relu(x)
        
        # Output layer (no activation - direct state prediction)
        x = self.output_layer(x)
        return x
    
class PhysicsInformedLoss:
    # Modifica __init__ per accettare lo state_scaler
    def __init__(self, state_scaler, lambda_physics=0.1): # Rimosso lambda_max etc se non usati
        """
        Args:
            state_scaler: Lo scaler StandardScaler fittato sugli stati (x..wz).
                          Serve per denormalizzare le velocità angolari predette.
            lambda_physics: Peso per il termine di loss fisica.
        """
        if state_scaler is None:
             raise ValueError("state_scaler must be provided to PhysicsInformedLoss")
        self.state_scaler = state_scaler
        self.lambda_physics = lambda_physics

        # Parametri fisici
        self.m = 1.5
        self.g = torch.tensor([0, 0, -9.81]) # Non usato in questa loss specifica
        self.J = torch.diag(torch.tensor([0.0146, 0.0168, 0.0309]))

        # Estrai media e scala per le velocità angolari (indici 9, 10, 11 nello stato x..wz)
        # E convertili subito a tensori (fallo una sola volta qui)
        omega_mean_np = self.state_scaler.mean_[9:12]
        omega_scale_np = self.state_scaler.scale_[9:12]
        self.omega_mean = torch.tensor(omega_mean_np, dtype=torch.float32)
        self.omega_scale = torch.tensor(omega_scale_np, dtype=torch.float32)


    # Modifica __call__ per accettare physics_info e usare lo scaler
    def __call__(self, predictions, targets, physics_info, epoch):
        """
        Calcola la loss combinata MSE + Physics.

        Args:
            predictions: Output del modello (stati scalati, x..wz), shape (B, 12).
            targets: Stati target reali (scalati, x..wz), shape (B, 12).
            physics_info: Tensore contenente [tempo, tau_x, tau_y, tau_z] NON scalati, shape (B, 4).
            epoch: Numero epoca attuale (non usato qui ma mantenuto per interfaccia).

        Returns:
            total_loss, mse_loss_item, physics_loss_item
        """
        # 1. MSE Loss (tra predizioni scalate e target scalati)
        mse_loss = F.mse_loss(predictions, targets)

        # Sposta media e scala sul device corretto (una volta per batch)
        current_device = predictions.device
        omega_mean = self.omega_mean.to(current_device)
        omega_scale = self.omega_scale.to(current_device)
        J = self.J.to(current_device)

        # 2. Physics Loss
        # Estrai tempo e coppie NON SCALATE da physics_info
        t = physics_info[:, 0]         # Tempo, shape (B,)
        tau = physics_info[:, 1:4]     # Coppie tau_x,y,z, shape (B, 3)

        # Estrai omega SCALATO dalle predizioni (indici 9, 10, 11)
        omega_scaled = predictions[:, 9:12] # Shape (B, 3)

        # Denormalizza omega
        # Assicurati che le operazioni siano broadcastable
        omega = omega_scaled * omega_scale.unsqueeze(0) + omega_mean.unsqueeze(0) # Shape (B, 3)

        #aggiunta equazione completa

        delta_t = t[1:] - t[:-1]
        delta_omega = omega[1:] - omega[:-1]
        epsilon = 1e-6
        omega_dot = delta_omega / (delta_t.unsqueeze(1) + epsilon)
        omega_trimmed = omega[:-1]
        tau_trimmed = tau[:-1]
        I_omega_dot = torch.einsum('ij,bj->bi', J, omega_dot)

        J_omega = torch.einsum('ij,bj->bi', J, omega_trimmed) # (3,3) x (B,3) -> (B,3)

        cross_term = torch.cross(omega_trimmed, J_omega, dim=1) # omega x (I*omega), shape (B, 3)

        # Calcola il residuo fisico: cross_term - tau
        physics_residual = I_omega_dot + cross_term - tau_trimmed # Shape (B, 3)
        physics_loss = torch.mean(physics_residual ** 2)

        mse_loss_trimmed = F.mse_loss(predictions[:-1], targets[:-1])

        # 3. Loss Totale
        total_loss = mse_loss_trimmed + self.lambda_physics * physics_loss

        return total_loss, mse_loss_trimmed.item(), physics_loss.item()

    def compute_cce(predictions: torch.Tensor, states: torch.Tensor):
        angular_accels = states[:, 3:6].detach().cpu().numpy()
        predictions = predictions.detach().cpu().numpy()
        covariance = np.cov(angular_accels.T, predictions.T)
        eigenvals, eigenvecs = np.linalg.eigh(covariance)
        return eigenvals, eigenvecs
