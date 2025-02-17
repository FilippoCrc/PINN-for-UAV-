import h5py
import numpy as np
from scipy.spatial.transform import Rotation

def quaternion_to_euler(quaternion):
    """Convert quaternion to euler angles (roll, pitch, yaw)"""
    r = Rotation.from_quat(quaternion)
    return r.as_euler('xyz')

def process_trajectory_data(trajectory_group):
    """
    Process a single trajectory's data into PINN format.
    Returns input features X and output labels Y.
    """
    # Extract groundtruth data
    gt_accel = np.array(trajectory_group['groundtruth/acceleration'])
    gt_angular_vel = np.array(trajectory_group['groundtruth/angular_velocity'])
    gt_attitude = np.array(trajectory_group['groundtruth/attitude'])
    gt_velocity = np.array(trajectory_group['groundtruth/velocity'])
    
    # Extract IMU data
    imu_accel = np.array(trajectory_group['imu/accelerometer'])
    imu_gyro = np.array(trajectory_group['imu/gyroscope'])
    
    # Convert quaternions to euler angles
    euler_angles = np.array([quaternion_to_euler(q) for q in gt_attitude])
    
    # Calculate angular acceleration using finite differences
    dt = 0.05  # 20Hz sampling rate (assumed from paper)
    angular_accel = np.gradient(gt_angular_vel, dt, axis=0)
    
    # Prepare input features X
    X = np.concatenate([
        gt_accel,          # Linear acceleration (v̇)
        angular_accel,     # Angular acceleration (ω̇)
        gt_velocity,       # Linear velocity (v)
        gt_angular_vel,    # Angular velocity (ω)
        euler_angles[:, 0:1],  # Roll (ϕ)
        euler_angles[:, 1:2],  # Pitch (θ)
        np.sin(euler_angles[:, 2:3]),  # sin(ψ)
        np.cos(euler_angles[:, 2:3])   # cos(ψ)
    ], axis=1)
    
    # For now, we'll use IMU measurements as proxy for rotor commands
    # In a real implementation, you would need actual PWM signals
    Y = imu_accel  # This is just a placeholder
    
    return X, Y

def load_and_process_data(file_path):
    """
    Load and process all trajectories from the HDF5 file.
    """
    with h5py.File(file_path, 'r') as f:
        all_X = []
        all_Y = []
        
        # Process each trajectory
        for trajectory_name in [k for k in f.keys() if k.startswith('trajectory_')]:
            print(f"Processing {trajectory_name}...")
            X, Y = process_trajectory_data(f[trajectory_name])
            all_X.append(X)
            all_Y.append(Y)
        
        # Concatenate all trajectories
        X = np.concatenate(all_X, axis=0)
        Y = np.concatenate(all_Y, axis=0)
        
        # Normalize data
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        Y_mean = Y.mean(axis=0)
        Y_std = Y.std(axis=0)
        
        X_normalized = (X - X_mean) / X_std
        Y_normalized = (Y - Y_mean) / Y_std
        
        return {
            'X': X_normalized,
            'Y': Y_normalized,
            'X_mean': X_mean,
            'X_std': X_std,
            'Y_mean': Y_mean,
            'Y_std': Y_std
        }

def visualize_data(data):
    """
    Create visualizations to verify the processed data.
    """
    import matplotlib.pyplot as plt
    
    # Plot a sample of the features
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle('Sample of Processed Features')
    
    feature_names = ['Linear Accel', 'Angular Accel', 'Linear Vel', 'Angular Vel', 
                    'Roll', 'Pitch', 'Sin(Yaw)', 'Cos(Yaw)']
    
    for i, (ax, name) in enumerate(zip(axes.flat, feature_names)):
        ax.plot(data['X'][:1000, i])
        ax.set_title(name)
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = "sensor_records.hdf5"
    
    # Load and process data
    print("Processing data...")
    processed_data = load_and_process_data(file_path)
    
    # Save processed data
    print("Saving processed data...")
    np.savez('pinn_data.npz', **processed_data)
    
    # Visualize the processed data
    print("Creating visualizations...")
    visualize_data(processed_data)
    
    print(f"Final data shapes:")
    print(f"X shape: {processed_data['X'].shape}")
    print(f"Y shape: {processed_data['Y'].shape}")