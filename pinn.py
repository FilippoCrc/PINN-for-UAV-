import torch
import torch.nn as nn
import torch.nn.functional as F

class QuadrotorPINN(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=25, num_layers=10, output_dim=4):
        """
        Physics-Informed Neural Network for quadrotor dynamics modeling.
        
        Args:
            input_dim: Dimension of input state vector (dimension: 15)
                      [v_dot(3), omega_dot(3), v(3), omega(3), phi, theta, sin(psi), cos(psi)]
            hidden_dim: Number of neurons in each hidden layer (dimension: 25)
            num_layers: Number of hidden layers (dimension: 10)
            output_dim: Dimension of output control vector (dimension: 4)
                       [PWM signals for each motor]
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
            Predicted PWM signals for each motor
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
        
        # Output layer (no activation - direct PWM prediction)
        x = self.output_layer(x)
        return x
    
class PhysicsInformedLoss:
    def __init__(self, lambda_max=0.1, num_cycles=5, total_epochs=300, maintain_ratio=0.5):
        """
        Physics-informed loss function with cyclical annealing.
        
        Args:
            lambda_max: Maximum weight for local monotonicity loss
            num_cycles: Number of annealing cycles in the paper is M
            total_epochs: Total number of training epochs in the paper is T
            maintain_ratio: Proportion of cycle to maintain lambda_max in the paper is R
        """
        self.lambda_max = lambda_max
        self.num_cycles = num_cycles
        self.total_epochs = total_epochs
        self.maintain_ratio = maintain_ratio
        
    def local_monotonicity_loss(self, predictions, states):
        """
        Calculate local monotonicity loss to enforce conservation of momentum.
        
        Args:
            predictions: Predicted PWM signals (batch_size, 4)
            states: Input state vector (batch_size, 15)
        """
        # Extract omega from states
        angular_accels = states[:, 3:6]  # omega_dot
        
        # Calculate consecutive differences
        pred_diff = predictions[1:] - predictions[:-1]
        accel_diff = angular_accels[1:] - angular_accels[:-1]
        
        # Apply tanh to get direction of change
        pred_direction = torch.tanh(pred_diff)
        accel_direction = torch.tanh(accel_diff)
        
        # Calculate consistency loss for each rotation axis
        loss_roll = torch.mean(1 - pred_direction[:, 0] * accel_direction[:, 0])
        loss_pitch = torch.mean(1 - pred_direction[:, 1] * accel_direction[:, 1])
        loss_yaw = torch.mean(1 - pred_direction[:, 2] * accel_direction[:, 2])
        
        return loss_roll + loss_pitch + loss_yaw
    
    def get_lambda(self, epoch):
        """Calculate annealing weight for current epoch."""
        cycle_length = self.total_epochs // self.num_cycles
        cycle_position = (epoch % cycle_length) / cycle_length # in the paper is beta
        
        if cycle_position <= self.maintain_ratio:
            return self.lambda_max
        else:
            return self.lambda_max * (1 - cycle_position) / (1 - self.maintain_ratio)
    
    def __call__(self, predictions, targets, states, epoch):
        """
        Calculate total loss combining MSE and physics-informed components.
        
        Args:
            predictions: Network predictions
            targets: Ground truth values
            states: Input state vector
            epoch: Current training epoch
        """
        # Calculate standard MSE loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # Calculate physics-informed loss with current lambda
        lambda_lm = self.get_lambda(epoch)
        physics_loss = lambda_lm * self.local_monotonicity_loss(predictions, states)
        
        return mse_loss + physics_loss, mse_loss.item(), physics_loss.item()