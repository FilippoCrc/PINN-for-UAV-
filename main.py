from dataset import create_dataloaders
from pinn import QuadrotorPINN
from trainer import train_pinn
import matplotlib.pyplot as plt
import torch
def plot_training_history(history):
    """Plot training and validation losses."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot total loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot component losses
    ax2.plot(history['train_mse'], label='Train MSE')
    ax2.plot(history['train_physics'], label='Train Physics')
    ax2.plot(history['val_mse'], label='Val MSE')
    ax2.plot(history['val_physics'], label='Val Physics')
    ax2.set_title('Component Losses')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create data loaders
    file_path = "pinn_data.npz"
    train_loader, val_loader = create_dataloaders(file_path)
    
    # Initialize model
    model = QuadrotorPINN()
    
    # Train the model
    history = train_pinn(model, train_loader, val_loader)
    
    # Plot training history
    plot_training_history(history)
    
    # Save the trained model
    torch.save(model.state_dict(), 'quadrotor_pinn.pth')