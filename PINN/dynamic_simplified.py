import torch
import torch.nn as nn

class SimplifiedQuadrotorDynamics:
    """
    Implements the simplified quadrotor dynamics from Equation (3) of the paper.
    This model is linearized around hovering conditions.
    """
    def __init__(self):
        # Parameters from Table 1
        self.m = 1.5  # Mass (kg)
        self.g = 9.81  # Gravity (m/s^2)
        self.Jx = 1.469e-2  # Inertia (kg⋅m²)
        self.Jy = 1.686e-2
        self.Jz = 3.093e-2

    def get_state_derivative(self, state, T, tau):
        """
        Computes the state derivatives using the simplified model from Equation (3)
        
        Args:
            state: tensor containing [vx, vy, vz, p, q, r, φ, θ]
                  shape: [batch_size, 8]
            T: total thrust
               shape: [batch_size, 1]
            tau: body torques [τx, τy, τz]
                 shape: [batch_size, 3]
        
        Returns:
            state_derivative: tensor containing derivatives according to Eq. (3)
                            shape: [batch_size, 6]
        """
        # Instead of using .T which changes shapes, we'll slice properly
        # Get angles (keeping the second dimension)
        phi = state[:, 6:7]    # shape: [batch_size, 1]
        theta = state[:, 7:8]  # shape: [batch_size, 1]
        
        # Unpack torques (keeping the second dimension)
        tau_x = tau[:, 0:1]  # shape: [batch_size, 1]
        tau_y = tau[:, 1:2]  # shape: [batch_size, 1]
        tau_z = tau[:, 2:3]  # shape: [batch_size, 1]
        
        # Compute derivatives according to Equation (3)
        # Now all terms will have shape [batch_size, 1]
        v_dot_x = -(1/self.m) * theta * T
        v_dot_y = (1/self.m) * phi * T
        v_dot_z = -(1/self.m) * T + self.g
        
        p_dot = (1/self.Jx) * tau_x
        q_dot = (1/self.Jy) * tau_y
        r_dot = (1/self.Jz) * tau_z
        
        # Concatenate all derivatives
        # Since each component is [batch_size, 1], we use cat instead of stack
        derivatives = torch.cat([
            v_dot_x, v_dot_y, v_dot_z,
            p_dot, q_dot, r_dot
        ], dim=1)
        
        return derivatives

class SimplifiedQuadrotorPINN(nn.Module):
    """
    Physics-Informed Neural Network using the simplified quadrotor dynamics
    """
    def __init__(self):
        super().__init__()
        self.dynamics = SimplifiedQuadrotorDynamics()
        
        # Neural network to predict thrust and torques from state
        self.net = nn.Sequential(
            nn.Linear(8, 32),  # Input: [vx, vy, vz, p, q, r, φ, θ]
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 4)   # Output: [T, τx, τy, τz]
        )
    
    def forward(self, state):
        """
        Predicts thrust and torques given state
        Returns total thrust T and torque vector τ
        """
        outputs = self.net(state)
        T = outputs[:, 0:1]    # Total thrust
        tau = outputs[:, 1:]   # Torques [τx, τy, τz]
        return T, tau
    
    def physics_loss(self, state, desired_derivative):
        """
        Computes physics-informed loss using simplified dynamics
        """
        # Predict thrust and torques
        T, tau = self.forward(state)
        
        # Compute state derivatives using simplified dynamics
        predicted_derivative = self.dynamics.get_state_derivative(state, T, tau)
        
        # Loss is MSE between predicted and desired derivatives
        loss = torch.mean((predicted_derivative - desired_derivative)**2)
        
        return loss