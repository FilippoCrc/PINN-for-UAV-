import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class QuadrotorDataset(Dataset):
    """Custom Dataset for Quadrotor data from preprocessed NPZ file."""
    def __init__(self, data_path):
        """
        Initialize the dataset using the preprocessed NPZ file.
        
        Args:
            data_path: Path to the NPZ file containing preprocessed data
                      The file should contain:
                      - X: normalized features [v_dot, omega_dot, v, omega, phi, theta, sin(psi), cos(psi)]
                      - Y: normalized targets (PWM signals or proxy)
        """
        # Load the preprocessed data
        data = np.load(data_path)
        
        # Convert to PyTorch tensors
        self.X = torch.FloatTensor(data['X'])
        self.Y = torch.FloatTensor(data['Y'])
        
        # Store normalization parameters for later use if needed
        self.X_mean = data['X_mean']
        self.X_std = data['X_std']
        self.Y_mean = data['Y_mean']
        self.Y_std = data['Y_std']
        
    def __len__(self):
        """Return the size of the dataset."""
        return len(self.X)
    
    def __getitem__(self, idx):
        """Return a single data point."""
        return self.X[idx], self.Y[idx]

def create_dataloaders(data_path, batch_size=64, train_split=0.8):
    """
    Create train and validation DataLoaders from preprocessed NPZ file.
    
    Args:
        data_path: Path to NPZ file
        batch_size: Size of batches for training
        train_split: Proportion of data to use for training
    """
    # Create the full dataset
    full_dataset = QuadrotorDataset(data_path)
    
    # Calculate split sizes
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Split into train and validation sets
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True  # Speeds up data transfer to GPU
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test the data loading pipeline
    data_path = "pinn_data.npz"  # The preprocessed data file
    train_loader, val_loader = create_dataloaders(data_path)
    
    # Print dataset information
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # Test a single batch
    states, targets = next(iter(train_loader))
    print(f"\nSample batch shapes:")
    print(f"States shape: {states.shape}")    # Should be [batch_size, 15]
    print(f"Targets shape: {targets.shape}")  # Should be [batch_size, 4]
    
    # Print feature statistics to verify normalization
    print("\nFeature statistics:")
    print(f"States mean: {states.mean():.3f}")
    print(f"States std: {states.std():.3f}")
    print(f"Targets mean: {targets.mean():.3f}")
    print(f"Targets std: {targets.std():.3f}")