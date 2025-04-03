import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import chi2

class QuadrotorPINN(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=45, num_layers=10, output_dim=4):
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
    def __init__(self, lambda_max=0.1, num_cycles=5, total_epochs=300, maintain_ratio=0.5):
        self.lambda_max = lambda_max
        self.num_cycles = num_cycles
        self.total_epochs = total_epochs
        self.maintain_ratio = maintain_ratio
        
    """ def local_monotonicity_loss(self, predicted_states):
        #Calculate physics loss using PREDICTED STATES#
        # Extract angular accelerations (omega_dot) from predicted states [indices 3-6]
        angular_accels = predicted_states[:, 3:6]  # Shape: (batch_size, 3)
        
        # Calculate consecutive differences in predictions and angular accelerations
        state_diff = predicted_states[1:] - predicted_states[:-1]  # Δstate
        accel_diff = angular_accels[1:] - angular_accels[:-1]      # Δomega_dot
        
        # Direction consistency using tanh
        state_direction = torch.tanh(state_diff[:, 3:6])  # Match angular acceleration indices
        accel_direction = torch.tanh(accel_diff)
        
        # Consistency loss for roll, pitch, yaw
        loss_roll = torch.mean(1 - state_direction[:, 0] * accel_direction[:, 0])
        loss_pitch = torch.mean(1 - state_direction[:, 1] * accel_direction[:, 1])
        loss_yaw = torch.mean(1 - state_direction[:, 2] * accel_direction[:, 2])
        
        return loss_roll + loss_pitch + loss_yaw """
    
    def local_monotonicity_loss(self, predicted_states):
        # Verify these indices match your state vector's angular acceleration positions
        angular_accels = predicted_states[:, 3:6]  # Adjust indices if needed
        
        # Add numerical stability
        state_diff = predicted_states[1:] - predicted_states[:-1] + 1e-7
        accel_diff = angular_accels[1:] - angular_accels[:-1] + 1e-7
        
        # Use smoother directional measure
        state_direction = torch.atan(state_diff[:, 3:6])  # More stable than tanh
        accel_direction = torch.atan(accel_diff)
        
        # Component-wise cosine similarity
        cos_sim = F.cosine_similarity(state_direction, accel_direction, dim=1)
        return torch.mean(1 - cos_sim)

    def get_lambda(self, epoch):
        cycle_length = self.total_epochs // self.num_cycles
        cycle_position = (epoch % cycle_length) / cycle_length
        return self.lambda_max * (1 - abs(cycle_position - self.maintain_ratio)/self.maintain_ratio)
    
    def __call__(self, predictions, targets, epoch):
        """Updated call signature (no need for input states)"""
        mse_loss = F.mse_loss(predictions, targets)
        lambda_lm = self.get_lambda(epoch)
        #physics_loss = lambda_lm * self.local_monotonicity_loss(predictions)
        #return mse_loss + physics_loss, mse_loss.item(), physics_loss.item()
        return mse_loss, mse_loss.item(), 0.0
    
    def compute_cce(predictions: torch.Tensor, states: torch.Tensor):

        # Extract angular accelerations
        angular_accels = states[:, 3:6].detach().cpu().numpy()
        predictions = predictions.detach().cpu().numpy()
            
        # Compute covariance matrix
        covariance = np.cov(angular_accels.T, predictions.T)
            
        # Compute eigenvalues and eigenvectors for visualization
        eigenvals, eigenvecs = np.linalg.eigh(covariance)
            
        return eigenvals, eigenvecs