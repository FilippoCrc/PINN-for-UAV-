import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os

class QuadrotorDataset(Dataset):
    def __init__(self, state_folder, input_folder):
        """
        Dataset class for loading quadrotor state and input data.
        
        Args:
            state_folder: Path to the folder containing state CSV files.
            input_folder: Path to the folder containing input CSV files.
        """
        self.state_folder = state_folder
        self.input_folder = input_folder
        
        # Load all state and input files
        self.state_files = sorted([os.path.join(state_folder, f) for f in os.listdir(state_folder) if f.endswith('.csv')])
        self.input_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')])
        
        # Load data into memory
        self.states = []
        self.inputs = []
        
        for state_file, input_file in zip(self.state_files, self.input_files):
            state_data = pd.read_csv(state_file, header=None).values
            input_data = pd.read_csv(input_file, header=None).values
            self.states.append(state_data)
            self.inputs.append(input_data)
        
        # Concatenate all data
        self.states = np.concatenate(self.states, axis=0)
        self.inputs = np.concatenate(self.inputs, axis=0)
        
        # Convert to PyTorch tensors
        self.states = torch.tensor(self.states, dtype=torch.float32)
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.inputs[idx]

def create_dataloaders(dataset, batch_size=64, train_ratio=0.7, val_ratio=0.15):
    """
    Create train, validation, and test dataloaders from the dataset.
    
    Args:
        dataset: The dataset to split.
        batch_size: Batch size for the dataloaders.
        train_ratio: Ratio of data to use for training.
        val_ratio: Ratio of data to use for validation.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader