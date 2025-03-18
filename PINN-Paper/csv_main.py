import torch
import matplotlib.pyplot as plt
from csv_dataset import QuadrotorDataset, create_dataloaders
from pinn import QuadrotorPINN, PhysicsInformedLoss
from trainer import train_pinn
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def visualize_training_history(history):
    """Visualizes the training progress including both MSE and physics-informed losses."""
    plt.figure(figsize=(15, 5))
    
    # Plot total loss
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MSE component
    plt.subplot(1, 3, 2)
    plt.plot(history['train_mse'], label='Train')
    plt.plot(history['val_mse'], label='Validation')
    plt.title('MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    
    # Plot physics component
    plt.subplot(1, 3, 3)
    plt.plot(history['train_physics'], label='Train')
    plt.plot(history['val_physics'], label='Validation')
    plt.title('Physics Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Physics Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_covariance_confidence_ellipse(predictions, states, title):
    """Updated CCE visualization for 3-axis systems."""
    # Extract angular accelerations from PREDICTIONS (not ground truth)
    angular_accels = predictions[:, 3:6].cpu().numpy()  # Use model outputs
    predictions = predictions.cpu().numpy()
    
    # Create subplots only for existing axes
    valid_axes = min(angular_accels.shape[1], 3)  # Handle <3 dimensions
    fig, axes = plt.subplots(1, valid_axes, figsize=(5*valid_axes, 5))
    
    for i in range(valid_axes):
        data = np.column_stack([angular_accels[:, i], predictions[:, 0]])  # Compare with first prediction dimension
        mean = np.mean(data, axis=0)
        cov = np.cov(data.T)
        
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        
        axes[i].scatter(data[:, 0], data[:, 1], alpha=0.5)
        axes[i].add_patch(plt.matplotlib.patches.Ellipse(
            mean, 3*np.sqrt(eigenvals[0]), 3*np.sqrt(eigenvals[1]),
            angle=angle, fill=False, color='red'
        ))
        axes[i].set_title(f'Motion Axis {i+1}')
        axes[i].set_xlabel('Angular Acceleration')
        axes[i].set_ylabel('State Prediction')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_loader, device):
    """Evaluates the model on test data and visualizes results."""
    model.eval()
    test_loss = 0
    predictions_list = []
    states_list = []
    targets_list = []
    
    with torch.no_grad():
        for states, targets in test_loader:
            states, targets = states.to(device), targets.to(device)
            predictions = model(states)
            
            # Store for later visualization
            predictions_list.append(predictions)
            states_list.append(states)
            targets_list.append(targets)
            
            # Calculate MSE
            test_loss += torch.nn.functional.mse_loss(predictions, targets).item()
    
    # Combine all batches
    all_predictions = torch.cat(predictions_list)
    all_states = torch.cat(states_list)
    all_targets = torch.cat(targets_list)
    
    # Print test metrics
    print(f"\nTest MSE: {test_loss/len(test_loader):.6f}")
    
     # Visualize using predictions only
    plot_covariance_confidence_ellipse(
        all_predictions,  # Pass predictions as both arguments
        all_predictions,  # Use predictions for angular accels
        "Physical Consistency Visualization (Test Data)"
    )
    
    return test_loss/len(test_loader)

def main():
    # Set device and random seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    print(f"Using device: {device}")
    
    # Create dataset and dataloaders
    print("\nLoading dataset...")
    dataset = QuadrotorDataset(
        state_folder="UAV_dataset/state_dataset",  # Update with your state folder path
        input_folder="UAV_dataset/input_dataset"   # Update with your input folder path
    )
    
    train_loader, val_loader, test_loader = create_dataloaders(dataset, batch_size=64)

    print(f"Dataset size: {len(dataset)}")
    
    # Initialize model
    print("\nInitializing PINN...")
    model = QuadrotorPINN(input_dim=5, output_dim=13).to(device)
    
    # Train model
    print("\nStarting training...")
    history = train_pinn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1000,
        learning_rate=1e-3
    )
    
    # Visualize training progress
    print("\nVisualizing training history...")
    visualize_training_history(history)
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    test_loss = evaluate_model(model, test_loader, device)
    
    # Save model
    print("\nSaving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'test_loss': test_loss
    }, 'trained_quadrotor_pinn.pth')
    print("Model saved to 'trained_quadrotor_pinn.pth'")

if __name__ == "__main__":
    main()