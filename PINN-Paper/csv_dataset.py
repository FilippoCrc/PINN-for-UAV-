# csv_dataset.py
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# --- QuadrotorDataset class remains unchanged ---
class QuadrotorDataset(Dataset):
    """
    Dataset for State -> Control Input prediction WITH Physics Info.
    Prepares data for a model predicting CONTROL INPUTS from STATES,
    and provides unscaled time/omega for physics loss calculation.
    """
    def __init__(self, state_csv_path, input_csv_path):
        self.state_csv_path = state_csv_path
        self.input_csv_path = input_csv_path

        print(f"Loading data for State->Input prediction:\n State (Model Input): {self.state_csv_path}\n Input (Model Target): {self.input_csv_path}")

        try:
            state_data = pd.read_csv(self.state_csv_path, header=None).values # t, x..wz
            input_data = pd.read_csv(self.input_csv_path, header=None).values # t, thrust, taux..tauz
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error finding CSV files: {e}. State='{self.state_csv_path}', Input='{self.input_csv_path}'") from e
        except Exception as e:
            raise RuntimeError(f"Error reading CSV files: {e}") from e

        if state_data.shape[0] != input_data.shape[0]:
            raise ValueError("Row mismatch between state and input files.")
        if state_data.shape[1] != 13: # Expect t + 12 states
            print(f"Warning: Expected 13 columns in state file (t + 12 states), found {state_data.shape[1]}.")
            # Adjust if your state dim is different, but time MUST be col 0
        if input_data.shape[1] != 5: # Expect t + 4 controls
             print(f"Warning: Expected 5 columns in input file (t + 4 controls), found {input_data.shape[1]}.")
             # Adjust if your control dim is different, but time MUST be col 0

        print(f"Data loaded. State shape: {state_data.shape}, Input shape: {input_data.shape}")

        # --- DATA PREPARATION ---
        # Model Input source (States, excluding time) -> needs scaling
        self.model_inputs_unscaled = torch.tensor(state_data[:, 1:], dtype=torch.float32) # Shape (N, state_dim)
        # Model Target source (Control Inputs, excluding time) -> needs scaling
        self.model_targets_unscaled = torch.tensor(input_data[:, 1:], dtype=torch.float32) # Shape (N, control_dim)

        # --- Data for Physics Loss (UNSCALED) ---
        self.times_unscaled = torch.tensor(state_data[:, 0], dtype=torch.float32)       # Shape (N,) Time t
        # Assuming omega (wx, wy, wz) are the last 3 columns of the state data
        self.omega_unscaled = torch.tensor(state_data[:, -3:], dtype=torch.float32)    # Shape (N, 3) Angular velocities

        # Placeholders
        self.state_scaler = None # Scales model inputs (states)
        self.input_scaler = None # Scales model targets (control inputs)
        self.scaled_model_inputs = None
        self.scaled_model_targets = None

    def __len__(self):
        # Length is the full number of samples
        return len(self.model_inputs_unscaled)

    def __getitem__(self, idx):
        """
        Returns data for sample `idx`. The physics loss will operate on pairs (idx, idx+1) within a batch.
        Args:
            idx (int): Index.
        Returns:
            tuple: (model_input, model_target, physics_info)
                - model_input (Tensor): Scaled state (excluding time) at time t. Shape (state_dim,).
                - model_target (Tensor): Scaled control input (excluding time) at time t. Shape (control_dim,).
                - physics_info (Tensor): Unscaled [time_t, omega_x(t), omega_y(t), omega_z(t)] for physics loss. Shape (4,).
        """
        if self.scaled_model_inputs is None or self.scaled_model_targets is None:
             raise RuntimeError("Dataset not scaled. Call create_dataloaders first.")

        # 1. Model Input (scaled state at t) + optional noise
        scaled_state = self.scaled_model_inputs[idx]
        # noise = torch.randn_like(scaled_state) * 0.01 # Optional noise
        # model_input = scaled_state + noise
        model_input = scaled_state


        # 2. Model Target (scaled control input at t)
        model_target = self.scaled_model_targets[idx]

        # 3. Physics Info (unscaled time and omega at t)
        #    Concatenate time scalar with omega vector
        physics_info = torch.cat(
            (self.times_unscaled[idx].unsqueeze(0), self.omega_unscaled[idx])
        ) # Shape (1+3 = 4,)

        return model_input, model_target, physics_info

# --- create_dataloaders (MODIFIED) ---
# Added shuffle_train_val and drop_last_train_val flags
def create_dataloaders(dataset, batch_size=128, train_ratio=0.7, val_ratio=0.15,
                       shuffle_train_val=False, # Default False for physics loss
                       drop_last_train_val=True, # Default True for physics loss
                       seed=42):
    """
    Creates dataloaders with SEQUENTIAL split and SCALES data.
    Args:
        ...
        shuffle_train_val (bool): If True, shuffle train/val loaders. MUST be False for physics loss.
        drop_last_train_val (bool): If True, drop last incomplete batch in train/val. MUST be True for physics loss.
        ...
    """
    if shuffle_train_val:
        print("WARNING: shuffle_train_val=True. This will likely break the physics loss calculation which relies on sequential data.")
    if not drop_last_train_val:
         print("WARNING: drop_last_train_val=False. This can lead to errors in physics loss if the last batch size is < 2.")


    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_size = len(dataset)
    if dataset_size < 2:
        raise ValueError("Dataset too small (< 2 samples) for physics loss calculation.")

    # --- Split calculation (same as before) ---
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    # Ensure non-negative sizes and adjust validation if test becomes negative
    train_size = max(0, train_size)
    val_size = max(0, val_size)
    if test_size < 0:
        val_size += test_size # Reduce validation size
        val_size = max(0, val_size)
        test_size = 0
    # Final check if rounding caused issues
    if train_size + val_size + test_size > dataset_size:
         test_size = dataset_size - train_size - val_size
         test_size = max(0, test_size)


    print(f"Dataset size: {dataset_size}")
    print(f"Sequential split: Train={train_size}, Val={val_size}, Test={test_size}")

    if train_size < 2:
        raise ValueError(f"Training split size ({train_size}) is too small (< 2) for physics loss calculation.")

    # --- SEQUENTIAL INDICES ---
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, dataset_size))

    # --- FIT SCALERS ON TRAINING DATA ---
    print("Fitting scalers on sequential training split...")
    train_states_unscaled = dataset.model_inputs_unscaled[train_indices]
    train_inputs_unscaled = dataset.model_targets_unscaled[train_indices]
    state_scaler = StandardScaler().fit(train_states_unscaled.numpy())
    input_scaler = StandardScaler().fit(train_inputs_unscaled.numpy())
    print("Scalers fitted.")

    # --- APPLY SCALING TO ENTIRE DATASET ---
    print("Applying scaling to the entire dataset...")
    dataset.scaled_model_inputs = torch.tensor(state_scaler.transform(dataset.model_inputs_unscaled.numpy()), dtype=torch.float32)
    dataset.scaled_model_targets = torch.tensor(input_scaler.transform(dataset.model_targets_unscaled.numpy()), dtype=torch.float32)
    dataset.state_scaler = state_scaler
    dataset.input_scaler = input_scaler
    print("Scaling applied.")

    # --- CREATE SUBSETS AND DATALOADERS ---
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices) if val_size > 0 else None
    test_subset = Subset(dataset, test_indices) if test_size > 0 else None

    # --- IMPORTANT: Apply shuffle/drop_last flags ---
    print(f"Creating DataLoaders (shuffle={shuffle_train_val}, drop_last={drop_last_train_val} for train/val)...")
    num_workers = 0
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset, batch_size=batch_size,
        shuffle=shuffle_train_val, # Use flag
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=drop_last_train_val # Use flag
    )
    val_loader = None
    if val_subset:
        # Check if val set is large enough even *with* drop_last
        if drop_last_train_val and len(val_indices) < batch_size:
             print(f"Warning: Validation set size ({len(val_indices)}) < batch size ({batch_size}) and drop_last=True. Validation loader will be empty.")
             # val_loader remains None
        elif len(val_indices) < 2 and drop_last_train_val:
            print(f"Warning: Validation set size ({len(val_indices)}) < 2 and drop_last=True. Cannot compute physics loss. Validation loader will be empty.")
            # val_loader remains None
        else:
            val_loader = DataLoader(
                val_subset, batch_size=batch_size,
                shuffle=shuffle_train_val, # Use flag
                num_workers=num_workers, pin_memory=pin_memory,
                drop_last=drop_last_train_val # Use flag
            )

    test_loader = None
    if test_subset:
        test_loader = DataLoader( # Test loader doesn't need drop_last=True usually
            test_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False
        )

    print(f"Data scaled. Scaled model inputs shape: {dataset.scaled_model_inputs.shape}, Scaled model targets shape: {dataset.scaled_model_targets.shape}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader) if val_loader else 0}, Test batches: {len(test_loader) if test_loader else 0}")

    return train_loader, val_loader, test_loader, state_scaler, input_scaler