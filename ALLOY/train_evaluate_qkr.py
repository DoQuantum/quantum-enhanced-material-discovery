import os
import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- Configuration ---
PROCESSED_DATA_DIR = "processed_qm9_data"
QUANTUM_KERNELS_DIR = "quantum_kernels_qm9"
TARGET_PROPERTY = 'gap' # Example target property from QM9 (e.g., 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv')

# Match subset sizes used in kernel_generator.py
# IMPORTANT: For a full run, these must match the (non-subset) sizes or be removed if full kernels were generated.
SUBSET_SIZE_TRAIN = 100 
SUBSET_SIZE_TEST_VALID = 20

def load_target_property(dataset_name, target_column_name, subset_size=None):
    """Loads the target property for a given dataset, filters NaNs, and optional subset size."""
    df_full = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, f"{dataset_name}.parquet"))
    
    # Filter NaNs for the specific target property
    df_filtered = df_full.dropna(subset=[target_column_name]).reset_index(drop=True)
    
    if subset_size:
        # Ensure subset_size does not exceed available data after filtering
        actual_subset_size = min(subset_size, len(df_filtered))
        if actual_subset_size < subset_size:
            print(f"Warning: Requested subset size {subset_size} for {dataset_name} reduced to {actual_subset_size} due to NaN filtering.")
        return df_filtered[target_column_name].values[:actual_subset_size].astype(np.float64)
        
    return df_filtered[target_column_name].values.astype(np.float64)

def main():
    print(f"--- Quantum Kernel Regression for Target: {TARGET_PROPERTY} ---")

    # --- Load Target Properties (NaNs filtered by load_target_property) ---
    print("Loading target properties (NaNs will be filtered)...")
    y_train = load_target_property("train", TARGET_PROPERTY, subset_size=SUBSET_SIZE_TRAIN)
    y_test = load_target_property("test", TARGET_PROPERTY, subset_size=SUBSET_SIZE_TEST_VALID)

    # Adjust subset sizes based on actual data loaded after NaN filtering for kernel name consistency
    # This is important because kernel_generator also filters NaNs now.
    # The number of samples for kernels should match the length of y_train, y_test.
    actual_train_subset_size = len(y_train)
    actual_test_subset_size = len(y_test)

    if actual_train_subset_size == 0:
        print(f"Error: No training target data left after filtering NaNs for '{TARGET_PROPERTY}'. Cannot proceed.")
        return
    if actual_test_subset_size == 0 :
         print(f"Warning: No test target data left after filtering NaNs for '{TARGET_PROPERTY}'. Test evaluation will be skipped.")
        # Or exit if test evaluation is critical

    print(f"Loaded {actual_train_subset_size} training targets (after NaN filter & subset).")
    print(f"Loaded {actual_test_subset_size} test targets (after NaN filter & subset).")

    # --- Load Precomputed Quantum Kernels ---
    # Kernel filenames should now reflect the target property if you changed them in kernel_generator
    print("\nLoading precomputed quantum kernels...")
    try:
        # Update kernel filenames to match those saved by the revised kernel_generator.py
        # (which now include TARGET_PROPERTY in the name and use potentially reduced subset sizes)
        kernel_train_train_path = os.path.join(QUANTUM_KERNELS_DIR, f"train_kernel_subset{actual_train_subset_size}_{TARGET_PROPERTY}.npy")
        kernel_test_train_path = os.path.join(QUANTUM_KERNELS_DIR, f"test_kernel_subset{actual_test_subset_size}_vs_train{actual_train_subset_size}_{TARGET_PROPERTY}.npy")

        K_train_train = np.load(kernel_train_train_path)
        if actual_test_subset_size > 0:
            K_test_train = np.load(kernel_test_train_path)
        else:
            K_test_train = np.array([]) # Empty if no test data
            
    except FileNotFoundError as e:
        print(f"Error loading kernel file: {e}")
        print("Please ensure kernel_generator.py was run successfully with NaN filtering and subset sizes match.")
        print("Also check that TARGET_PROPERTY is consistent in filenames.")
        return

    print(f"Loaded K_train_train with shape: {K_train_train.shape}")
    if actual_test_subset_size > 0:
        print(f"Loaded K_test_train with shape: {K_test_train.shape}")
    else:
        print("No test kernel loaded as no test data available after filtering.")


    # Consistency checks
    if K_train_train.shape[0] != actual_train_subset_size or K_train_train.shape[1] != actual_train_subset_size:
        print(f"Error: Training kernel shape {K_train_train.shape} inconsistent with y_train length {actual_train_subset_size}")
        return
    
    if actual_test_subset_size > 0:
        if K_test_train.shape[0] != actual_test_subset_size or K_test_train.shape[1] != actual_train_subset_size:
            print(f"Error: Test kernel shape {K_test_train.shape} inconsistent with y_test ({actual_test_subset_size}) or y_train ({actual_train_subset_size}) lengths.")
            return
    
    # --- Train SVR Model ---
    print("\nTraining SVR model with precomputed quantum kernel...")
    svr_model = SVR(kernel='precomputed', C=1.0, epsilon=0.1)
    svr_model.fit(K_train_train, y_train)
    print("SVR model training complete.")

    # --- Evaluate Model ---
    y_train_pred = svr_model.predict(K_train_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    print(f"\nTraining Set Performance (on {actual_train_subset_size} samples):")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  R2:   {train_r2:.4f}")

    if actual_test_subset_size > 0:
        y_test_pred = svr_model.predict(K_test_train)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        print(f"\nTest Set Performance (on {actual_test_subset_size} samples):")
        print(f"  MAE:  {test_mae:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  R2:   {test_r2:.4f}")
    else:
        print("\nSkipping test set evaluation as no test data was available after filtering.")
    
    print("\nSVR model training and evaluation script finished.")
    print(f"IMPORTANT: Model was trained and evaluated on SUBSETS of data.")
    print(f"For a full run, ensure kernel_generator.py produced full kernels and update subset sizes here.")

if __name__ == "__main__":
    main()