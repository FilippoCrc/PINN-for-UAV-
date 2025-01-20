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


""" class MidAirDataset(torch.utils.data.Dataset):
    
    Dataset class for Mid-Air quadrotor data.
    

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
    
    def _get_trajectory_path(self, trajectory_id: str) -> str:
        Get full path to trajectory HDF5 file.
        return os.path.join(self.base_path, 'trajectories', 
                          self.weather_condition, f'{trajectory_id}.h5')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx] """

class MidAirDataset(Dataset):
    def __init__(self, hdf5_path):
        """
        Initialize dataset from HDF5 file containing multiple trajectories.
        
        Args:
            hdf5_path: Path to the HDF5 file containing flight data
        """
        self.hdf5_path = hdf5_path
        self.data = self.prepare_data()
    
    def prepare_data(self):
        """
        Load and preprocess data from HDF5 file.
        Returns a list of (state, control) pairs for all trajectories.
        """
        processed_data = []
        
        with h5py.File(self.hdf5_path, 'r') as f:
            # Get all trajectory groups
            trajectory_groups = [key for key in f.keys() if key.startswith('trajectory_')]
            
            for traj_name in trajectory_groups:
                traj_group = f[traj_name]
                
                # Load IMU data (100Hz)
                imu_data = traj_group['imu']
                gyro = imu_data['gyroscope'][:]  # Angular velocities
                accel = imu_data['accelerometer'][:]  # Linear accelerations
                
                # Load groundtruth data (100Hz)
                gt = traj_group['groundtruth']
                attitude = gt['attitude'][:]  # Note: shape is (N, 4) - might be quaternions
                position = gt['position'][:]
                velocity = gt['velocity'][:]
                angular_velocity = gt['angular_velocity'][:]
                
                # Process data to 25Hz (downsample by taking every 4th sample)
                sample_rate = 4  # 100Hz to 25Hz
                
                for i in range(0, len(accel), sample_rate):
                    # Get current sample indices
                    idx = i
                    next_idx = min(i + sample_rate, len(accel))
                    
                    # Calculate derivatives (using finite differences)
                    v_dot = np.mean(accel[idx:next_idx], axis=0)
                    omega_dot = (gyro[next_idx-1] - gyro[idx]) / (sample_rate/100.0) if next_idx > idx else np.zeros(3)
                    
                    # Get current state values
                    v = velocity[idx]
                    omega = angular_velocity[idx]
                    
                    # Convert quaternion to euler angles if needed
                    if attitude.shape[1] == 4:
                        # Implement quaternion to euler conversion here if needed
                        # For now, assuming the first three components are euler angles
                        phi, theta, psi = attitude[idx][:3]
                    else:
                        phi, theta, psi = attitude[idx][:3]
                    
                    # Create state vector
                    state = np.concatenate([
                        v_dot,              # Linear accelerations (3)
                        omega_dot,          # Angular accelerations (3)
                        v,                  # Linear velocities (3)
                        omega,              # Angular velocities (3)
                        [phi, theta],       # Roll and pitch (2)
                        [np.sin(psi), np.cos(psi)]  # Yaw encoding (2)
                    ])
                    
                    # Create synthetic PWM values based on accelerations and velocities
                    hover_thrust = 0.5  # Base thrust for hovering
                    vertical_adjust = 0.1 * v_dot[2]  # Adjustment based on vertical acceleration
                    roll_pitch_adjust = 0.05 * (abs(phi) + abs(theta))  # Adjustment based on attitude
                    yaw_adjust = 0.03 * omega[2]  # Adjustment based on yaw rate
                    
                    # Individual motor commands
                    pwm1 = hover_thrust + vertical_adjust + roll_pitch_adjust + yaw_adjust
                    pwm2 = hover_thrust + vertical_adjust - roll_pitch_adjust + yaw_adjust
                    pwm3 = hover_thrust + vertical_adjust + roll_pitch_adjust - yaw_adjust
                    pwm4 = hover_thrust + vertical_adjust - roll_pitch_adjust - yaw_adjust
                    
                    # Combine and clip PWM values
                    pwm_values = np.clip([pwm1, pwm2, pwm3, pwm4], 0, 1)
                    
                    # Convert to torch tensors
                    state_tensor = torch.FloatTensor(state)
                    pwm_tensor = torch.FloatTensor(pwm_values)
                    
                    processed_data.append((state_tensor, pwm_tensor))
        
        print(f"Loaded {len(processed_data)} total samples from {len(trajectory_groups)} trajectories")
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