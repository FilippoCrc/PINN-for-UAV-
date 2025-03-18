import torch
import torch.nn as nn
import torch.nn.functional as F
from pinn import QuadrotorPINN
from pinn import PhysicsInformedLoss

def train_pinn(model, train_loader, val_loader, num_epochs=1000, learning_rate=1e-3):
    """
    Train the Physics-Informed Neural Network.
    
    Args:
        model: QuadrotorPINN instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Initialize loss function
    criterion = PhysicsInformedLoss()
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_mse': [], 'train_physics': [],
        'val_mse': [], 'val_physics': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_total_loss = 0.0
        train_total_mse = 0.0
        train_total_physics = 0.0
        
        for batch_idx, (states, targets) in enumerate(train_loader):
            states, targets = states.to(device), targets.to(device)
            
            # Forward pass
            predictions = model(states)
            
            # Calculate loss
            loss, mse, physics = criterion(predictions, targets, states, epoch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record losses
            train_total_loss += loss.item()
            train_total_mse += mse
            train_total_physics += physics
        
        # Validation phase
        model.eval()
        val_total_loss = 0.0
        val_total_mse = 0.0
        val_total_physics = 0.0
        
        with torch.no_grad():
            for states, targets in val_loader:
                states, targets = states.to(device), targets.to(device)
                predictions = model(states)
                loss, mse, physics = criterion(predictions, targets, states, epoch)
                
                val_total_loss += loss.item()
                val_total_mse += mse
                val_total_physics += physics
        
        # Record epoch statistics
        history['train_loss'].append(train_total_loss / len(train_loader))
        history['train_mse'].append(train_total_mse / len(train_loader))
        history['train_physics'].append(train_total_physics / len(train_loader))
        history['val_loss'].append(val_total_loss / len(val_loader))
        history['val_mse'].append(val_total_mse / len(val_loader))
        history['val_physics'].append(val_total_physics / len(val_loader))
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {history["train_loss"][-1]:.4f} '
                  f'(MSE: {history["train_mse"][-1]:.4f}, '
                  f'Physics: {history["train_physics"][-1]:.4f})')
            print(f'Val Loss: {history["val_loss"][-1]:.4f} '
                  f'(MSE: {history["val_mse"][-1]:.4f}, '
                  f'Physics: {history["val_physics"][-1]:.4f})')
    
    return history