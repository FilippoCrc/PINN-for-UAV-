import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Full quadrotor dynamics implementation
class FullQuadrotorDynamics:
    def __init__(self, device='cpu'):
        self.device = device
        # System parameters (from Table 1)
        self.m = 1.5          # Mass [kg]
        self.g = 9.81         # Gravity [m/s²]
        self.J = torch.diag(torch.tensor([1.469e-2, 1.686e-2, 3.093e-2], device=device))  # Inertia tensor [kg·m²]
        
        # Rotor parameters
        self.c_T = 1.113e-5   # Thrust coefficient
        self.c_Q = 1.779e-7   # Torque coefficient
        self.l = 0.175        # Moment arm [m]

    def euler_to_rotation(self, phi, theta, psi):
        """Converts Euler angles to rotation matrix (FRD to NED) using batch-friendly operations"""
        # Compute trigonometric functions
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        cos_psi = torch.cos(psi)
        sin_psi = torch.sin(psi)

        # Build rotation matrices using element-wise operations
        R_x = torch.stack([
            torch.ones_like(phi), torch.zeros_like(phi), torch.zeros_like(phi),
            torch.zeros_like(phi), cos_phi, sin_phi,
            torch.zeros_like(phi), -sin_phi, cos_phi
        ], dim=-1).view(-1, 3, 3)

        R_y = torch.stack([
            cos_theta, torch.zeros_like(theta), -sin_theta,
            torch.zeros_like(theta), torch.ones_like(theta), torch.zeros_like(theta),
            sin_theta, torch.zeros_like(theta), cos_theta
        ], dim=-1).view(-1, 3, 3)

        R_z = torch.stack([
            cos_psi, sin_psi, torch.zeros_like(psi),
            -sin_psi, cos_psi, torch.zeros_like(psi),
            torch.zeros_like(psi), torch.zeros_like(psi), torch.ones_like(psi)
        ], dim=-1).view(-1, 3, 3)

        return torch.bmm(R_z, torch.bmm(R_y, R_x))


    def euler_derivative_matrix(self, phi, theta):
        """Batch-compatible transformation matrix"""
        tan_theta = torch.tan(theta)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        sec_theta = 1.0 / torch.cos(theta)

        row1 = torch.stack([torch.ones_like(phi),
                            sin_phi * tan_theta,
                            cos_phi * tan_theta], dim=-1)
        
        row2 = torch.stack([torch.zeros_like(phi),
                            cos_phi,
                            -sin_phi], dim=-1)
        
        row3 = torch.stack([torch.zeros_like(phi),
                            sin_phi * sec_theta,
                            cos_phi * sec_theta], dim=-1)

        return torch.stack([row1, row2, row3], dim=-2)

    def compute_derivatives(self, state, u):
        """
        Computes full nonlinear derivatives from Equation (1)
        Args:
            state: [vx, vy, vz, p, q, r, phi, theta, psi] (body frame)
            u: [n1^2, n2^2, n3^2, n4^2] rotor speeds squared
        Returns:
            derivatives: [v_dot, omega_dot, eta_dot]
        """
        # Unpack state components
        v = state[:, :3]    # Linear velocity (body frame)
        omega = state[:, 3:6]  # Angular velocity (body frame)
        eta = state[:, 6:9]    # Euler angles [phi, theta, psi]
        
        # Compute rotation matrix
        R = self.euler_to_rotation(eta[:,0], eta[:,1], eta[:,2])
        
        # Compute control forces and torques (Equation 2)
        rotor_matrix = torch.tensor([
            [self.c_T, self.c_T, self.c_T, self.c_T],
            [-self.c_T*self.l, self.c_T*self.l, self.c_T*self.l, -self.c_T*self.l],
            [self.c_T*self.l, -self.c_T*self.l, self.c_T*self.l, -self.c_T*self.l],
            [self.c_Q, self.c_Q, -self.c_Q, -self.c_Q]
        ], device=self.device)
        
        w_sq = u.unsqueeze(-1)
        forces_torques = torch.matmul(rotor_matrix, w_sq).squeeze(-1)
        
        # Split into forces and torques
        T = forces_torques[:, 0].unsqueeze(-1)
        tau = forces_torques[:, 1:4]
        
        # Compute linear acceleration (Equation 1b)
        gravity_force = self.m * self.g * torch.tensor([0, 0, 1], 
            dtype=torch.float32, device=self.device).repeat(state.shape[0], 1).unsqueeze(-1)
        
        v_dot = (-torch.cross(omega, v, dim=1)
                + torch.bmm(R.mT, gravity_force).squeeze(-1)
                + T/self.m)
        
        # Compute angular acceleration (Equation 1d)
        J_inv = torch.inverse(self.J)
        omega_dot = torch.matmul(J_inv, (torch.cross(omega, torch.matmul(self.J, omega.unsqueeze(-1)).squeeze(-1), dim=1) + tau.squeeze(-1)).unsqueeze(-1)).squeeze(-1)
        
        # Compute Euler angles derivatives (Equation 1c)
        Phi = self.euler_derivative_matrix(eta[:,0], eta[:,1])
        eta_dot = torch.bmm(Phi, omega.unsqueeze(-1)).squeeze(-1)
        
        return torch.cat([v_dot, omega_dot, eta_dot], dim=1)

# PINN implementation
class QuadrotorPINN(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, output_dim=6, device='cpu'):
        super().__init__()
        self.device = device
        self.dynamics = FullQuadrotorDynamics(device=device)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)
        
    def forward(self, state, u):
        """Predicts state derivatives [v_dot, omega_dot]"""
        return self.net(state)
    
    def physics_loss(self, state, u):
        """Computes physics-informed loss using full dynamics"""
        # Network prediction
        pred_derivatives = self.forward(state, u)
        
        # True derivatives from physics
        true_derivatives = self.dynamics.compute_derivatives(state, u)
        
        # Calculate MSE loss
        return torch.mean((pred_derivatives - true_derivatives[:, :6])**2)

# Training and collocation points generation
def generate_collocation_points(batch_size=1024, device='cpu'):
    """Generates random collocation points within plausible ranges"""
    state = torch.rand(batch_size, 9, device=device) * 2 - 1  # Normalized [-1, 1]
    
    # Physical constraints
    state[:, 2] = torch.abs(state[:, 2])  # vz positive (NED)
    state[:, 6:8] = state[:, 6:8] * np.pi/6  # phi, theta ±30°
    state[:, 8] = torch.rand(batch_size, device=device) * 2*np.pi  # psi [0, 2π]
    
    # Control inputs (normalized rotor speeds squared)
    u = torch.rand(batch_size, 4, device=device) * 0.5 + 0.7  # 0.7-1.2 normalized
    
    return state, u

def train_pinn(model, epochs=5000, batch_size=1024, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500)
    
    losses = []
    for epoch in range(epochs):
        state, u = generate_collocation_points(batch_size, model.device)
        optimizer.zero_grad()
        
        loss = model.physics_loss(state, u)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        losses.append(loss.item())
        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4e}")
            
    return losses

# Validation against numerical integration
def runge_kutta(f, y0, t, args=()):
    """Numerical integration using RK4 method"""
    y = torch.zeros((len(t), len(y0)), device=y0.device)
    y[0] = y0
    for i in range(len(t)-1):
        h = t[i+1] - t[i]
        k1 = f(y[i], *args)
        k2 = f(y[i] + k1*h/2, *args)
        k3 = f(y[i] + k2*h/2, *args)
        k4 = f(y[i] + k3*h, *args)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4)*h/6
    return y

def validate_model(model):
    """Compare PINN predictions with numerical integration"""
    # Initial conditions (hover)
    y0 = torch.tensor([0, 0, 0,  # vx, vy, vz
                      0, 0, 0,    # p, q, r
                      0, 0, 0], device=model.device)   # phi, theta, psi
    
    # Control inputs (hover)
    u_hover = torch.full((4,), (model.dynamics.m * model.dynamics.g) / 
                            (4 * model.dynamics.c_T), device=model.device)
    
    # Time span
    t = torch.linspace(0, 5, 50, device=model.device)
    
    # Numerical integration
    def physics_derivatives(y, u):
        return model.dynamics.compute_derivatives(y.unsqueeze(0), u.unsqueeze(0)).squeeze(0)
    
    y_rk = runge_kutta(physics_derivatives, y0, t, args=(u_hover,))
    
    # PINN prediction
    with torch.no_grad():
        y_pinn = torch.zeros_like(y_rk)
        y_pinn[0] = y0
        for i in range(len(t)-1):
            state = y_pinn[i].unsqueeze(0)
            control = u_hover.unsqueeze(0)
            dy = model(state, control)
            y_pinn[i+1] = y_pinn[i] + dy.squeeze(0) * (t[i+1] - t[i])
    
    # Plot results
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    labels = ['vx', 'vy', 'vz', 'p', 'q', 'r', 'phi', 'theta', 'psi']
    for i in range(9):
        ax = axs[i//3, i%3]
        ax.plot(t.cpu().numpy(), y_rk[:, i].cpu().numpy(), label='RK4')
        ax.plot(t.cpu().numpy(), y_pinn[:, i].cpu().numpy(), '--', label='PINN')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(labels[i])
        ax.legend()
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize and train PINN
    pinn = QuadrotorPINN(input_dim=9, hidden_dim=64, output_dim=6, device=device)
    losses = train_pinn(pinn, epochs=5000, lr=1e-3)
    
    # Plot training loss
    plt.figure()
    plt.semilogy(losses)
    plt.title("Training Loss Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Physics Loss")
    plt.grid(True)
    plt.show()
    
    # Validate against numerical solution
    validate_model(pinn)
