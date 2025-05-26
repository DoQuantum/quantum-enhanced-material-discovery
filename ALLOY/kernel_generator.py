import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib # For saving the scaler

from quantum_utils import compute_pennylane_quantum_kernel_matrix, pennylane_scalar_feature_map_ops # Assuming quantum_utils.py is in the same directory
# Note: pennylane_scalar_feature_map_ops is not directly called here, but good to be aware of its role.

# --- Configuration ---
PROCESSED_DATA_DIR = "processed_qm9_data"
QUANTUM_KERNELS_DIR = "quantum_kernels_qm9"
SCALER_FILENAME = "scalar_features_scaler.joblib"

# These are the 7 scalar descriptors we decided to use
SCALAR_FEATURE_COLUMNS = ['mol_weight', 'num_rotatable_bonds', 'num_h_donors', 'num_h_acceptors', 'logp', 'tpsa', 'num_rings']
NUM_QUANTUM_FEATURES = len(SCALAR_FEATURE_COLUMNS)

# PennyLane device for kernel computation
DEVICE_NAME = "default.qubit" # Use "lightning.qubit" for potentially faster simulation if installed

# Add TARGET_PROPERTY to configuration, same as in train_evaluate_qkr.py
TARGET_PROPERTY = 'gap' 

def main():
    if not os.path.exists(QUANTUM_KERNELS_DIR):
        os.makedirs(QUANTUM_KERNELS_DIR)

    # --- Load Data ---
    df_train_full = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "train.parquet"))
    df_valid_full = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "valid.parquet"))
    df_test_full = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "test.parquet"))

    # --- Filter out NaN target values BEFORE processing features for those samples ---
    print(f"Filtering samples with NaN for target property: {TARGET_PROPERTY}")
    df_train = df_train_full.dropna(subset=[TARGET_PROPERTY]).reset_index(drop=True)
    df_valid = df_valid_full.dropna(subset=[TARGET_PROPERTY]).reset_index(drop=True)
    df_test = df_test_full.dropna(subset=[TARGET_PROPERTY]).reset_index(drop=True)
    
    print(f"Original train size: {len(df_train_full)}, After NaN filter: {len(df_train)}")
    print(f"Original valid size: {len(df_valid_full)}, After NaN filter: {len(df_valid)}")
    print(f"Original test size:  {len(df_test_full)}, After NaN filter: {len(df_test)}")


    # Extract features from filtered dataframes
    x_train_raw = df_train[SCALAR_FEATURE_COLUMNS].values.astype(np.float64)
    x_valid_raw = df_valid[SCALAR_FEATURE_COLUMNS].values.astype(np.float64)
    x_test_raw = df_test[SCALAR_FEATURE_COLUMNS].values.astype(np.float64)
    
    if len(x_train_raw) == 0:
        print(f"Error: No training samples left after filtering NaNs for target '{TARGET_PROPERTY}'. Check your data or target property name.")
        return

    # --- Scale Features ---
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    print("Fitting scaler on training data and transforming...")
    x_train_scaled = scaler.fit_transform(x_train_raw)
    
    scaler_path = os.path.join(QUANTUM_KERNELS_DIR, SCALER_FILENAME)
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    print("Transforming validation and test data...")
    if len(x_valid_raw) > 0:
        x_valid_scaled = scaler.transform(x_valid_raw)
    else:
        x_valid_scaled = np.array([]) # Handle empty validation set
        print("Validation set is empty after NaN filtering.")
        
    if len(x_test_raw) > 0:
        x_test_scaled = scaler.transform(x_test_raw)
    else:
        x_test_scaled = np.array([]) # Handle empty test set
        print("Test set is empty after NaN filtering.")


    print(f"\nScaled training features shape: {x_train_scaled.shape}")
    if x_train_scaled.shape[0] > 0:
        print(f"Example of first scaled training vector: {x_train_scaled[0]}")
    else:
        print("No training data to show example from.")
        return # Cannot proceed if no training data

    # --- Compute and Save Quantum Kernels ---
    # Adjust subset sizes based on potentially smaller filtered data
    subset_size_train = min(100, len(x_train_scaled))
    subset_size_test_valid = min(20, len(x_test_scaled)) # Use x_test_scaled for test subset size
    subset_size_valid = min(20, len(x_valid_scaled)) # Use x_valid_scaled for valid subset size


    print(f"\nUsing subset_size_train: {subset_size_train}")
    print(f"Using subset_size_test_valid (for test): {subset_size_test_valid}")
    print(f"Using subset_size_valid (for validation): {subset_size_valid}")


    if subset_size_train == 0:
        print("No training samples for kernel computation after subsetting. Exiting.")
        return

    x_train_scaled_subset = x_train_scaled[:subset_size_train]
    
    print(f"\nCalculating training kernel matrix (K_train_train) for {len(x_train_scaled_subset)} samples...")
    kernel_train_train = compute_pennylane_quantum_kernel_matrix(
        x_data=x_train_scaled_subset, 
        num_features=NUM_QUANTUM_FEATURES,
        device_name=DEVICE_NAME
    )
    train_kernel_path = os.path.join(QUANTUM_KERNELS_DIR, f"train_kernel_subset{subset_size_train}_{TARGET_PROPERTY}.npy")
    np.save(train_kernel_path, kernel_train_train)
    print(f"Training kernel (subset) saved to {train_kernel_path}, shape: {kernel_train_train.shape}")

    # Validation Kernel
    if subset_size_valid > 0 and len(x_valid_scaled) > 0:
        x_valid_scaled_subset = x_valid_scaled[:subset_size_valid]
        print(f"\nCalculating validation kernel matrix (K_valid_train) for {len(x_valid_scaled_subset)} vs {len(x_train_scaled_subset)} samples...")
        kernel_valid_train = compute_pennylane_quantum_kernel_matrix(
            x_data=x_valid_scaled_subset, 
            y_data=x_train_scaled_subset,
            num_features=NUM_QUANTUM_FEATURES,
            device_name=DEVICE_NAME
        )
        valid_kernel_path = os.path.join(QUANTUM_KERNELS_DIR, f"valid_kernel_subset{subset_size_valid}_vs_train{subset_size_train}_{TARGET_PROPERTY}.npy")
        np.save(valid_kernel_path, kernel_valid_train)
        print(f"Validation kernel (subset) saved to {valid_kernel_path}, shape: {kernel_valid_train.shape}")
    else:
        print("\nSkipping validation kernel calculation (no validation data after filtering/subsetting).")

    # Test Kernel
    if subset_size_test_valid > 0 and len(x_test_scaled) > 0:
        x_test_scaled_subset = x_test_scaled[:subset_size_test_valid]
        print(f"\nCalculating test kernel matrix (K_test_train) for {len(x_test_scaled_subset)} vs {len(x_train_scaled_subset)} samples...")
        kernel_test_train = compute_pennylane_quantum_kernel_matrix(
            x_data=x_test_scaled_subset, 
            y_data=x_train_scaled_subset,
            num_features=NUM_QUANTUM_FEATURES,
            device_name=DEVICE_NAME
        )
        test_kernel_path = os.path.join(QUANTUM_KERNELS_DIR, f"test_kernel_subset{subset_size_test_valid}_vs_train{subset_size_train}_{TARGET_PROPERTY}.npy")
        np.save(test_kernel_path, kernel_test_train)
        print(f"Test kernel (subset) saved to {test_kernel_path}, shape: {kernel_test_train.shape}")
    else:
        print("\nSkipping test kernel calculation (no test data after filtering/subsetting).")

    print("\nQuantum kernel generation script finished.")
    print(f"IMPORTANT: Kernels were generated for SUBSETS of data for speed.")
    print(f"For a full run, remove or adjust subset_size_train and subset_size_test_valid in the script.")

if __name__ == "__main__":
    main()