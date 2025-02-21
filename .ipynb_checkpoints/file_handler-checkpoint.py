import scipy.io
import numpy as np

# Load the .mat file
file_path = "L23_neuron_20210228_Y54_Z320_test.mat"  # Update with the correct path
mat_data = scipy.io.loadmat(file_path)

# Convert MATLAB arrays to NumPy arrays
data_dict = {
    "Eigenface_0_trials_evoked": np.array(mat_data["Eigenface_0_trials_evoked"]),
    "Eigenface_0_trials_isi": np.array(mat_data["Eigenface_0_trials_isi"]),
    "dFF0_trials_evoked": np.array(mat_data["dFF0_trials_evoked"]),
    "dFF0_trials_isi": np.array(mat_data["dFF0_trials_isi"]),
}

# Save each array as an .npy file
for key, data in data_dict.items():
    np.save(f"{key}.npy", data)
    print(f"Saved {key}.npy with shape {data.shape}")

# Now, you can reload an array using:
# loaded_array = np.load("Eigenface_0_trials_evoked.npy")

import scipy.io
import numpy as np

def load_mat_to_numpy(file_path):
    """Loads a MATLAB .mat file and converts its arrays to NumPy format."""
    # Load the .mat file
    mat_data = scipy.io.loadmat(file_path)

    # Extract variables and convert to NumPy arrays
    data_dict = {
        key: np.array(value) for key, value in mat_data.items() 
        if not key.startswith("__")  # Ignore MATLAB metadata fields
    }

    # Print shapes of the extracted arrays
    for key, data in data_dict.items():
        print(f"{key}: shape {data.shape}")

    return data_dict

# Example usage:
file_path = "L23_neuron_20210228_Y54_Z320_test.mat"  # Update with the correct path
data_dict = load_mat_to_numpy(file_path)

