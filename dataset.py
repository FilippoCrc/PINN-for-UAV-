import os
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
from typing import Tuple, List, Dict

BASE_PATH = "sensor_records"   # BASE_PATH: Path to dataset root directory

def setup_dataset_structure():
    # Create directory structure
    base_dir = "sensor_records"
    subdirs = ["rgb", "depth", "imu", "gps", "ground_truth"]
    for dir in subdirs:
        os.makedirs(os.path.join(base_dir, dir), exist_ok=True)

def load_flight_data(trajectory_id):
    # Load sensor data from HDF5
    with h5py.File(f'trajectory_{trajectory_id}.h5', 'r') as f:
        imu_data = f['imu'][:]
        gps_data = f['gps'][:]
        

    return imu_data, gps_data

def synchronize_sensors(imu_data: np.ndarray, gps_data: np.ndarray, timestamp_base: np.ndarray) -> np.ndarray:
    """
    Synchronize IMU and GPS data to a common time base (25Hz).
    
    Args:
        imu_data: Raw IMU measurements at 100Hz
        gps_data: Raw GPS measurements at 1Hz
        timestamp_base: Common timebase for synchronization (25Hz)
    
    Returns:
        Synchronized sensor data aligned with timestamp_base
    """
    # Interpolate IMU data (100Hz → 25Hz)
    imu_timestamps = np.linspace(0, len(imu_data)/100, len(imu_data))
    synchronized_imu = np.zeros((len(timestamp_base), imu_data.shape[1]))
    for i in range(imu_data.shape[1]):
        synchronized_imu[:, i] = np.interp(timestamp_base, imu_timestamps, imu_data[:, i])
    
    # Interpolate GPS data (1Hz → 25Hz)
    gps_timestamps = np.linspace(0, len(gps_data), len(gps_data))
    synchronized_gps = np.zeros((len(timestamp_base), gps_data.shape[1]))
    for i in range(gps_data.shape[1]):
        synchronized_gps[:, i] = np.interp(timestamp_base, gps_timestamps, gps_data[:, i])
    
    # Combine synchronized data
    return np.hstack([synchronized_imu, synchronized_gps])

def load_trajectory_list(BASE_PATH: str, weather_condition: str) -> List[str]:
    """
    Load list of available trajectories for given weather condition.
    
    Args:
        base_path: Path to dataset root directory
        weather_condition: Selected weather condition
        
    Returns:
        List of trajectory IDs
    """
    trajectory_path = os.path.join(BASE_PATH, 'trajectories', weather_condition)
    trajectory_files = [f for f in os.listdir(trajectory_path) if f.endswith('.h5')]
    return [os.path.splitext(f)[0] for f in trajectory_files]

def preprocess_data(imu_data, gps_data):
    # Synchronize data to common timebase (25 Hz)
    synced_data = synchronize_sensors(imu_data, gps_data)
    
    # Create input features for PINN
    # Extract components from synchronized data
    # These indices should match your HDF5 file structure
    v_dot = synced_data[:, 0:3]  # Linear accelerations
    omega_dot = synced_data[:, 3:6]  # Angular accelerations
    v = synced_data[:, 6:9]  # Linear velocities
    omega = synced_data[:, 9:12]  # Angular velocities
    euler = synced_data[:, 12:15]  # Euler angles
    
    # Calculate sin(psi) and cos(psi) for yaw angle
    sin_psi = np.sin(euler[:, 2:3])
    cos_psi = np.cos(euler[:, 2:3])
    
    # Combine into state vector
    states = np.hstack([v_dot, omega_dot, v, omega, euler[:, 0:2], sin_psi, cos_psi])
    controls = synced_data[:, -4:]
    
    return states, controls


class MidAirDataset(torch.utils.data.Dataset):
    """
    Dataset class for Mid-Air quadrotor data.
    """

    def __init__(self, BASE_PATH, weather_condition='clear'):
        self.base_path = BASE_PATH
        self.weather_condition = weather_condition
        self.trajectories = load_trajectory_list(BASE_PATH, weather_condition)
        self.data = self.prepare_data()
    
    def prepare_data(self):
        processed_data = []
        for traj in self.trajectories:
            imu, gps, images = load_flight_data(traj)
            states, controls = preprocess_data(imu, gps, images)

            # Convert to torch tensors
            states = torch.FloatTensor(states)
            controls = torch.FloatTensor(controls)

            processed_data.append((states, controls))

        return processed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    
def create_dataLoaders(dataset, batch_size=64, split=[0.7, 0.15, 0.15]):
    """
    Create train/val/test dataloaders.
    
    Args:
        base_path: Path to dataset root
        batch_size: Batch size for training
        
    Returns:
        Training, validation, and test dataloaders
    """

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

# Usage example
base_path = "path/to/midair/dataset"
train_loader, val_loader, test_loader = create_dataLoaders(base_path)