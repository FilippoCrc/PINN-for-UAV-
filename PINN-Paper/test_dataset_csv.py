import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

class QuadrotorDataset(Dataset):
    def __init__(self, state_folder, input_folder, transform=None):
        """
        Dataset class for quadrotor data from CSV files.
        
        Args:
            state_folder: Path to folder containing state CSV files
            input_folder: Path to folder containing input CSV files
            transform: Optional transform to apply to samples
        """
        self.state_data = []
        self.input_data = []
        self.transform = transform
        
        # Get sorted file lists to ensure matching between state and input
        state_files = sorted([f for f in os.listdir(state_folder) if f.endswith('.csv')])
        input_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.csv')])
        
        # Check that we have matching files
        assert len(state_files) == len(input_files), "Number of state and input files must match"
        
        # Load and process each file pair
        for state_file, input_file in zip(state_files, input_files):
            # Load state and input data
            state_df = pd.read_csv(os.path.join(state_folder, state_file))
            input_df = pd.read_csv(os.path.join(input_folder, input_file))
            
            # Print column names for debugging
            print(f"State columns: {state_df.columns.tolist()}")
            print(f"Input columns: {input_df.columns.tolist()}")
            
            # Get the first row to see the structure
            print(f"First row of state data:\n{state_df.iloc[0]}")
            print(f"First row of input data:\n{input_df.iloc[0]}")
            
            # For now, let's just break after the first file to see the structure
            break