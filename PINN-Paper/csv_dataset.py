import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os

""" class QuadrotorDataset(Dataset):
    def __init__(self, state_folder, input_folder):
        
        # Dataset class for loading quadrotor input and state data.
        
        # Args:
        #     state_folder: Path to the folder containing state CSV files.
        #     input_folder: Path to the folder containing input CSV files.
        
        self.state_folder = state_folder
        self.input_folder = input_folder
        
        # Load all state and input files
        self.state_files = sorted([os.path.join(state_folder, f) for f in os.listdir(state_folder) if f.endswith('.csv')])
        self.input_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')])
        
        # Load data into memory
        self.inputs = []
        self.states = []
        
        for input_file, state_file in zip(self.input_files, self.state_files):
            input_data = pd.read_csv(input_file, header=None).values
            state_data = pd.read_csv(state_file, header=None).values
            self.inputs.append(input_data)
            self.states.append(state_data)
        
        # Concatenate all data
        self.inputs = np.concatenate(self.inputs, axis=0)
        self.states = np.concatenate(self.states, axis=0)
        
        # Convert to PyTorch tensors
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.states = torch.tensor(self.states, dtype=torch.float32)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.states[idx] """

""" class QuadrotorDataset(Dataset):
    def __init__(self, state_folder, input_folder):
        self.state_folder = state_folder
        self.input_folder = input_folder
        
        # Load all files
        self.state_files = sorted([os.path.join(state_folder, f) for f in os.listdir(state_folder) if f.endswith('.csv')])
        self.input_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')])
        
        # Load and concatenate data
        self.states = []
        self.inputs = []
        
        for state_file, input_file in zip(self.state_files, self.input_files):
            state_data = pd.read_csv(state_file, header=None).values
            input_data = pd.read_csv(input_file, header=None).values
            self.states.append(state_data)
            self.inputs.append(input_data)
        
        self.states = np.concatenate(self.states, axis=0)
        self.inputs = np.concatenate(self.inputs, axis=0)

        # 1. Add Normalization Here
        from sklearn.preprocessing import StandardScaler
        self.state_scaler = StandardScaler()
        self.input_scaler = StandardScaler()
        
        # Fit on full dataset (temporary solution)
        self.states = self.state_scaler.fit_transform(self.states)  # Normalize states
        self.inputs = self.input_scaler.fit_transform(self.inputs)  # Normalize inputs

        # Convert to tensors
        self.states = torch.tensor(self.states, dtype=torch.float32)
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.states[idx]  # (normalized_input, normalized_state) """

class QuadrotorDataset(Dataset):
    def __init__(self, state_folder, input_folder):
        self.state_folder = state_folder
        self.input_folder = input_folder
        
        # Load all files
        self.state_files = sorted([os.path.join(state_folder, f) for f in os.listdir(state_folder) if f.endswith('.csv')])
        self.input_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')])
        
        # Load and concatenate data (excluding first column)
        self.states = []
        self.inputs = []
        
        for state_file, input_file in zip(self.state_files, self.input_files):
            # Read data and exclude first column using [:, 1:]
            state_data = pd.read_csv(state_file, header=None).values[:, 1:]
            input_data = pd.read_csv(input_file, header=None).values[:, 1:]
            self.states.append(state_data)
            self.inputs.append(input_data)
        
        self.states = np.concatenate(self.states, axis=0)
        self.inputs = np.concatenate(self.inputs, axis=0)

        """
        train_size = int(0.7 * len(self.states))
        self.state_scaler.fit(self.states[:train_size])  # Fit only on training data
        self.states = self.state_scaler.transform(self.states)  # Normalize states

         # Normalization
        from sklearn.preprocessing import StandardScaler
        self.state_scaler = StandardScaler()
        self.input_scaler = StandardScaler() 
        
        self.states = self.state_scaler.fit_transform(self.states)
        self.inputs = self.input_scaler.fit_transform(self.inputs) """

        # Convert to tensors
        self.states = torch.tensor(self.states, dtype=torch.float32)
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.states[idx]

def create_dataloaders(dataset, batch_size=128, train_ratio=0.7, val_ratio=0.15):
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
    
    # Extract training data for scaling
    train_inputs = torch.stack([dataset.inputs[i] for i in train_dataset.indices])
    train_states = torch.stack([dataset.states[i] for i in train_dataset.indices])
    
    # Initialize scalers using training data only
    from sklearn.preprocessing import StandardScaler
    state_scaler = StandardScaler()
    input_scaler = StandardScaler()
    state_scaler.fit(train_states.numpy())
    input_scaler.fit(train_inputs.numpy())
    
    # Apply normalization to ENTIRE dataset (safe because scalers use training stats)
    dataset.inputs = torch.tensor(input_scaler.transform(dataset.inputs.numpy()), dtype=torch.float32)
    dataset.states = torch.tensor(state_scaler.transform(dataset.states.numpy()), dtype=torch.float32)
    
    # Create dataloaders (now normalized)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader