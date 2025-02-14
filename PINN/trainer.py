import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from dynamic_simplified import SimplifiedQuadrotorDynamics
from dynamic_simplified import SimplifiedQuadrotorPINN
from validation import comprehensive_validation
def generate_training_data(n_samples=1000):
    """
    Generates training data for the simplified quadrotor model.
    We'll create states within reasonable ranges and compute their derivatives
    using nominal control inputs.
    
    Returns:
        states: tensor of shape (n_samples, 8) containing [vx, vy, vz, p, q, r, φ, θ]
        derivatives: tensor of shape (n_samples, 6) containing [v̇x, v̇y, v̇z, ṗ, q̇, ṙ]
    """
    # Generate random states within physically reasonable ranges
    states = torch.zeros(n_samples, 8)
    
    # Velocities (m/s) - small values around hover
    states[:, 0:3] = torch.randn(n_samples, 3) * 0.5  # vx, vy, vz
    
    # Angular rates (rad/s) - small values
    states[:, 3:6] = torch.randn(n_samples, 3) * 0.1  # p, q, r
    
    # Angles (rad) - small values around hover
    states[:, 6:8] = torch.randn(n_samples, 2) * 0.1  # φ, θ
    
    # Create nominal control inputs for hover condition
    m = 1.5  # mass in kg
    g = 9.81  # gravity
    T = torch.ones(n_samples, 1) * (m * g)  # thrust to maintain hover
    tau = torch.zeros(n_samples, 3)  # no rotational motion in hover
    
    # Compute corresponding derivatives using simplified dynamics
    dynamics = SimplifiedQuadrotorDynamics()
    derivatives = dynamics.get_state_derivative(states, T, tau)
    
    return states, derivatives

def train_pinn(model, n_epochs=1000, batch_size=64, learning_rate=0.001):
    """
    Trains the Physics-Informed Neural Network using generated data.
    
    Args:
        model: SimplifiedQuadrotorPINN instance
        n_epochs: number of training epochs
        batch_size: size of training batches
        learning_rate: learning rate for optimizer
    """
    # Generate training data
    states, derivatives = generate_training_data(n_samples=10000)
    n_samples = states.shape[0]
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Lists to store losses for plotting
    losses = []
    
    # Training loop
    for epoch in range(n_epochs):
        epoch_losses = []
        
        # Process data in batches
        for i in range(0, n_samples, batch_size):
            # Get batch
            state_batch = states[i:i+batch_size]
            derivative_batch = derivatives[i:i+batch_size]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute physics loss
            loss = model.physics_loss(state_batch, derivative_batch)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Average loss for this epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        
        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.6f}')
    
    return losses

def plot_training_results(losses):
    """
    Plots the training losses over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

def validate_model(model):
    """
    Validates the trained model by comparing its predictions
    with the simplified dynamics for a few test cases.
    """
    # Generate some test states
    test_states, test_derivatives = generate_training_data(n_samples=10)
    
    # Get model predictions
    with torch.no_grad():
        T, tau = model(test_states)
        predicted_derivatives = model.dynamics.get_state_derivative(test_states, T, tau)
    
    # Print comparison for first test case
    print("\nValidation Results (First Test Case):")
    print("True Derivatives:")
    print(test_derivatives[0].numpy())
    print("\nPredicted Derivatives:")
    print(predicted_derivatives[0].numpy())
    print("\nControl Inputs:")
    print(f"Thrust: {T[0].item():.3f} N")
    print(f"Torques: {tau[0].numpy()} Nm")


# Example usage:
if __name__ == "__main__":
    states, derivatives = generate_training_data(n_samples=10)
    print("\nStates shape:", states.shape)
    print("Derivatives shape:", derivatives.shape)
    print("\nFirst state vector:")
    print(states[0])
    print("\nFirst derivative vector:")
    print(derivatives[0])
    # Create model
    model = SimplifiedQuadrotorPINN()
    
    # Train model
    losses = train_pinn(model)
    
    # Plot training results
    plot_training_results(losses)
    
    # Validate model
    validate_model(model)
    comprehensive_validation(model)