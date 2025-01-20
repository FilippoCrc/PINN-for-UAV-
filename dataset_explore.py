import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def get_file_path():
    """
    Helps locate the HDF5 file by checking the current directory and printing useful debugging information.
    Returns the correct file path if found.
    """
    # Print current working directory
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # List all files in current directory
    print("\nFiles in current directory:")
    for file in os.listdir(current_dir):
        print(f"- {file}")
    
    # Look for HDF5 files specifically
    hdf5_files = [f for f in os.listdir(current_dir) if f.endswith('.hdf5')]
    print("\nHDF5 files found:")
    for file in hdf5_files:
        print(f"- {file}")
    
    # If we find exactly one HDF5 file, use that
    if len(hdf5_files) == 1:
        return os.path.join(current_dir, hdf5_files[0])
    # If we find multiple HDF5 files, look for sensor_record.hdf5
    elif len(hdf5_files) > 1:
        if 'sensor_record.hdf5' in hdf5_files:
            return os.path.join(current_dir, 'sensor_record.hdf5')
        else:
            print("\nWarning: Multiple HDF5 files found but none named 'sensor_record.hdf5'")
            return None
    else:
        print("\nWarning: No HDF5 files found in current directory")
        return None

def explore_hdf5_file(file_path):
    """
    Explores and prints the complete structure of an HDF5 file.
    Now with better error handling.
    """
    if file_path is None:
        print("No valid file path provided")
        return
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    print(f"\nAttempting to open: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            def print_structure(name, obj):
                indent = "  " * name.count("/")
                if isinstance(obj, h5py.Dataset):
                    print(f"{indent}Dataset: {name}")
                    print(f"{indent}  Shape: {obj.shape}")
                    print(f"{indent}  Type: {obj.dtype}")
                    try:
                        sample = obj[0] if obj.shape[0] > 0 else None
                        print(f"{indent}  First value: {sample}")
                    except Exception as e:
                        print(f"{indent}  Cannot read first value: {e}")
                elif isinstance(obj, h5py.Group):
                    print(f"{indent}Group: {name}")
            
            print("\n=== HDF5 File Structure ===")
            f.visititems(print_structure)
            
    except Exception as e:
        print(f"Error reading HDF5 file: {str(e)}")

# Main execution
if __name__ == "__main__":
    print("Starting HDF5 file exploration...")
    
    # First locate the file
    file_path = get_file_path()
    
    if file_path:
        print(f"\nFound HDF5 file at: {file_path}")
        explore_hdf5_file(file_path)
    else:
        print("\nPlease make sure the sensor_record.hdf5 file is in the current directory")
        print("You can also provide the full path to the file by modifying the code")