import os
import pandas as pd
import numpy as np
from ase.io import read as ase_read
from ase.atoms import Atoms
from io import StringIO
from rdkit import Chem # For atomic numbers from symbols

# --- Configuration ---
PROCESSED_DATA_DIR = "processed_qm9_data"
FEATURES_OUTPUT_DIR = "features_qm9" # Directory to save new features
MAX_ATOMS = 29 # QM9 molecules have up to 9 heavy atoms (C,N,O,F). Max H count can make it up to ~29.
               # We should determine this dynamically from the dataset or set a safe upper bound.

# --- Helper Functions ---
def xyz_string_to_ase_atoms(xyz_string):
    """Converts an XYZ coordinate string to an ASE Atoms object."""
    try:
        # The XYZ string from RDKit (and likely stored) starts with num_atoms and a title line.
        xyz_file_like = StringIO(xyz_string)
        atoms = ase_read(xyz_file_like, format="xyz")
        return atoms
    except Exception as e:
        print(f"Error converting XYZ string to ASE Atoms: {e}\nXYZ String was:\n{xyz_string[:200]}...") # Print first 200 chars
        return None

def calculate_coulomb_matrix(ase_atoms_obj, max_atoms=MAX_ATOMS):
    """
    Calculates the Coulomb matrix for a molecule.

    Args:
        ase_atoms_obj (ase.Atoms): ASE Atoms object for the molecule.
        max_atoms (int): The size to pad the matrix to (N x N).

    Returns:
        np.ndarray: The padded Coulomb matrix (max_atoms x max_atoms).
                    Returns None if calculation fails.
    """
    if ase_atoms_obj is None:
        return None
        
    num_atoms = len(ase_atoms_obj)
    if num_atoms == 0:
        return np.zeros((max_atoms, max_atoms))
    if num_atoms > max_atoms:
        print(f"Warning: Molecule with {num_atoms} atoms exceeds max_atoms ({max_atoms}). Truncating.")
        # Or handle differently, e.g., by returning None or raising error
        actual_num_atoms_to_use = max_atoms
    else:
        actual_num_atoms_to_use = num_atoms

    # Get atomic numbers (Z)
    # ASE stores symbols, RDKit's GetAtomicNum can convert symbol to Z
    atomic_numbers = np.array([Chem.GetPeriodicTable().GetAtomicNumber(sym) for sym in ase_atoms_obj.get_chemical_symbols()[:actual_num_atoms_to_use]])
    
    # Get positions
    positions = ase_atoms_obj.get_positions()[:actual_num_atoms_to_use]

    # Initialize Coulomb matrix for the actual number of atoms
    coulomb_matrix_unpadded = np.zeros((actual_num_atoms_to_use, actual_num_atoms_to_use))

    # Calculate off-diagonal elements: Z_i * Z_j / R_ij
    # And diagonal elements: 0.5 * Z_i^2.4
    for i in range(actual_num_atoms_to_use):
        coulomb_matrix_unpadded[i, i] = 0.5 * (atomic_numbers[i] ** 2.4)
        for j in range(i + 1, actual_num_atoms_to_use):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist > 0: # Avoid division by zero if atoms are at the same position (should not happen)
                value = (atomic_numbers[i] * atomic_numbers[j]) / dist
                coulomb_matrix_unpadded[i, j] = value
                coulomb_matrix_unpadded[j, i] = value # Matrix is symmetric
            else: # Should be rare, indicates an issue with geometry
                coulomb_matrix_unpadded[i, j] = 0 
                coulomb_matrix_unpadded[j, i] = 0


    # Pad the matrix to max_atoms x max_atoms
    padded_coulomb_matrix = np.zeros((max_atoms, max_atoms))
    padded_coulomb_matrix[:actual_num_atoms_to_use, :actual_num_atoms_to_use] = coulomb_matrix_unpadded
    
    return padded_coulomb_matrix

def main():
    if not os.path.exists(FEATURES_OUTPUT_DIR):
        os.makedirs(FEATURES_OUTPUT_DIR)

    global MAX_ATOMS # Allow modification based on dataset scan

    # Determine the overall MAX_ATOMS by checking all splits first
    # This ensures consistent padding size across train, valid, and test sets.
    overall_max_atoms_found = 0
    for dataset_name in ["train", "valid", "test"]:
        parquet_path = os.path.join(PROCESSED_DATA_DIR, f"{dataset_name}.parquet")
        if os.path.exists(parquet_path):
            df_temp = pd.read_parquet(parquet_path)
            current_max = df_temp['num_atoms'].max()
            if current_max > overall_max_atoms_found:
                overall_max_atoms_found = current_max
        else:
            print(f"Warning: {parquet_path} not found. Skipping for MAX_ATOMS check.")
    
    if overall_max_atoms_found > MAX_ATOMS:
        print(f"Updating MAX_ATOMS from {MAX_ATOMS} to {overall_max_atoms_found} based on scan of all datasets.")
        MAX_ATOMS = overall_max_atoms_found
    elif overall_max_atoms_found > 0 and overall_max_atoms_found < MAX_ATOMS:
         print(f"Note: Max atoms found across all sets ({overall_max_atoms_found}) is less than configured MAX_ATOMS ({MAX_ATOMS}). Padding to {MAX_ATOMS}.")
         # Or, you could set MAX_ATOMS = overall_max_atoms_found
    elif overall_max_atoms_found == 0:
        print(f"Warning: Could not determine max atoms from datasets. Using default MAX_ATOMS: {MAX_ATOMS}")


    for dataset_name in ["train", "valid", "test"]:
        input_parquet_path = os.path.join(PROCESSED_DATA_DIR, f"{dataset_name}.parquet")
        
        if not os.path.exists(input_parquet_path):
            print(f"INFO: Processed data file not found at {input_parquet_path}. Skipping this set.")
            continue

        print(f"\n--- Processing {dataset_name} set ---")
        print(f"Loading data from {input_parquet_path}...")
        df = pd.read_parquet(input_parquet_path)

        print(f"Calculating Coulomb matrices for {len(df)} molecules (padding to {MAX_ATOMS}x{MAX_ATOMS})...")
        
        coulomb_matrices = []
        molecule_ids_processed = []
        failed_cm_calc = 0

        for index, row in df.iterrows():
            if index % 10000 == 0 and index > 0 and dataset_name == "train": # Only print progress for larger train set
                print(f"Processing molecule {index} in {dataset_name} set...")
            
            ase_atoms = xyz_string_to_ase_atoms(row['xyz_coordinates'])
            if ase_atoms:
                # Ensure ase_atoms does not exceed MAX_ATOMS before passing to calculation
                if len(ase_atoms) > MAX_ATOMS:
                     # This case should ideally be handled by how ase_atoms is sliced in calculate_coulomb_matrix
                     # or by filtering molecules that are too large if that's the desired strategy.
                     # For now, calculate_coulomb_matrix itself handles truncation based on its max_atoms arg.
                     pass

                cm = calculate_coulomb_matrix(ase_atoms, max_atoms=MAX_ATOMS)
                if cm is not None:
                    coulomb_matrices.append(cm)
                    molecule_ids_processed.append(row['molecule_id'])
                else:
                    failed_cm_calc +=1
                    print(f"Failed to calculate Coulomb matrix for molecule_id: {row['molecule_id']} in {dataset_name} set.")
            else:
                failed_cm_calc +=1
                print(f"Failed to parse XYZ for molecule_id: {row['molecule_id']} in {dataset_name} set.")

        print(f"\nFinished calculating Coulomb matrices for {dataset_name} set.")
        print(f"Successfully generated {len(coulomb_matrices)} matrices.")
        print(f"Failed calculations: {failed_cm_calc}")

        if coulomb_matrices:
            coulomb_matrices_array = np.array(coulomb_matrices)
            
            output_filename_cm = f"{dataset_name}_coulomb_matrices.npy"
            output_path_cm = os.path.join(FEATURES_OUTPUT_DIR, output_filename_cm)
            
            print(f"Saving Coulomb matrices to {output_path_cm} (shape: {coulomb_matrices_array.shape})")
            np.save(output_path_cm, coulomb_matrices_array)
            print(f"Coulomb matrices for {dataset_name} set saved.")

            ids_output_filename = f"{dataset_name}_molecule_ids.npy"
            ids_output_path = os.path.join(FEATURES_OUTPUT_DIR, ids_output_filename)
            np.save(ids_output_path, np.array(molecule_ids_processed))
            print(f"Molecule IDs for {dataset_name} set saved to {ids_output_path}")
        else:
            print(f"No Coulomb matrices were generated for {dataset_name} set.")
    
    print("\nAll processing complete for feature_builder.py.")

if __name__ == "__main__":
    main()