import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, GetMolFrags
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import json
import sys # For flushing output

# --- Configuration ---
QM9_CSV_PATH = "qm9.csv"
XYZ_DIR_PATH = "gdb9_xyz"
PROCESSED_DATA_DIR = "processed_qm9_data"
MANIFEST_FILE = os.path.join(PROCESSED_DATA_DIR, "manifest.json")
MAX_WORKERS = os.cpu_count()

QM9_TARGET_PROPERTIES_FROM_CSV = [
    "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u298", "h298", "g298", "cv"
]

CUSTOM_DESCRIPTORS_TO_COMPUTE = {
    "mol_weight": Descriptors.MolWt,
    "num_rotatable_bonds": Descriptors.NumRotatableBonds,
    "num_h_donors": Descriptors.NumHDonors,
    "num_h_acceptors": Descriptors.NumHAcceptors,
    "logp": Descriptors.MolLogP,
    "tpsa": Descriptors.TPSA,
    "num_rings": Descriptors.RingCount,
}

# --- Utility Functions ---
def get_xyz_string_from_conformer(mol):
    """Extracts XYZ coordinate string from the first conformer of an RDKit molecule."""
    if mol is None or mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer(0) # Get the first conformer
    xyz_lines = [str(mol.GetNumAtoms()), mol.GetProp("_Name") if mol.HasProp("_Name") else "Mol"]
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        atom_symbol = mol.GetAtomWithIdx(i).GetSymbol()
        xyz_lines.append(f"{atom_symbol} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    return "\n".join(xyz_lines)

def compute_custom_descriptors(mol):
    """Computes a predefined set of descriptors for an RDKit molecule."""
    desc = {}
    for name, func in CUSTOM_DESCRIPTORS_TO_COMPUTE.items():
        try:
            desc[name] = func(mol)
        except Exception: # Catch any error during descriptor calculation
            desc[name] = np.nan 
    return desc

def check_interatomic_distances(mol, min_dist=0.7):
    """Checks for unusually short interatomic distances in the first conformer."""
    if mol is None or mol.GetNumAtoms() <= 1 or mol.GetNumConformers() == 0:
        return True 
    conf = mol.GetConformer(0)
    for i in range(mol.GetNumAtoms()):
        for j in range(i + 1, mol.GetNumAtoms()):
            dist = conf.GetAtomPosition(i).Distance(conf.GetAtomPosition(j))
            if dist < min_dist :
                return False # Atoms too close
    return True

def add_conformer_from_xyz_file(rdkit_mol_with_hs_from_smiles, xyz_filepath, mol_idx_for_debug):
    """
    Reads an XYZ file, creates a new molecule, and attempts to add its conformer 
    to the provided rdkit_mol (typically created from SMILES).
    Returns True if successful, False otherwise.
    """
    mol_name_for_debug = rdkit_mol_with_hs_from_smiles.GetProp("_Name") if rdkit_mol_with_hs_from_smiles.HasProp("_Name") else "unknown_mol"

    if not os.path.exists(xyz_filepath):
        if mol_idx_for_debug < 5: # Limit printing
            print(f"Debug XYZ ({mol_name_for_debug}, idx {mol_idx_for_debug}): XYZ file not found: {xyz_filepath}")
            sys.stdout.flush()
        return False
        
    mol_from_xyz = Chem.MolFromXYZFile(xyz_filepath)
    if mol_from_xyz is None:
        if mol_idx_for_debug < 5:
            print(f"Debug XYZ ({mol_name_for_debug}, idx {mol_idx_for_debug}): Chem.MolFromXYZFile failed for {xyz_filepath}")
            sys.stdout.flush()
        return False
    
    if mol_from_xyz.GetNumConformers() == 0:
        if mol_idx_for_debug < 5:
            print(f"Debug XYZ ({mol_name_for_debug}, idx {mol_idx_for_debug}): No conformers found in molecule from {xyz_filepath}")
            sys.stdout.flush()
        return False

    # Compare atom counts: rdkit_mol_with_hs_from_smiles (SMILES -> AddHs()) vs. mol_from_xyz (XYZ file)
    # Both should now have explicit hydrogens.
    num_atoms_smiles_hs = rdkit_mol_with_hs_from_smiles.GetNumAtoms()
    num_atoms_xyz = mol_from_xyz.GetNumAtoms()

    if num_atoms_smiles_hs != num_atoms_xyz:
        if mol_idx_for_debug < 5:
            print(f"Debug XYZ ({mol_name_for_debug}, idx {mol_idx_for_debug}): Atom count mismatch. SMILES+Hs: {num_atoms_smiles_hs}, XYZ: {num_atoms_xyz}. File: {xyz_filepath}")
            sys.stdout.flush()
        return False
    
    # If atom counts match, assign the conformer from XYZ to the SMILES-derived molecule (which already has Hs)
    conf = mol_from_xyz.GetConformer(0)
    rdkit_mol_with_hs_from_smiles.RemoveAllConformers() # Remove any existing (e.g. 2D) conformer
    rdkit_mol_with_hs_from_smiles.AddConformer(conf, assignId=True)
    
    # if mol_idx_for_debug < 5: # Less critical print, can be enabled if needed
    #     print(f"Debug XYZ ({mol_name_for_debug}, idx {mol_idx_for_debug}): Successfully added conformer from {xyz_filepath}")
    #     sys.stdout.flush()
    return True

# --- Molecule Processing Function ---
def process_molecule_data(args):
    idx, data_row = args # idx is the 0-based row index
    
    # This will be 'gdb_1', 'gdb_2', etc., based on your CSV's 'mol_id' column or fallback
    mol_id_for_output_df = data_row.get('mol_id', f"qm9_row_{idx+1}") # Use idx+1 for 1-based row num if 'mol_id' is missing
    
    smiles_from_csv = data_row.get('smiles', None)

    if not smiles_from_csv:
        return None

    rdkit_mol_from_smiles = Chem.MolFromSmiles(smiles_from_csv)
    if rdkit_mol_from_smiles is None:
        if idx < 5:
            print(f"Debug Main ({mol_id_for_output_df}, idx {idx}): Failed to parse SMILES: {smiles_from_csv}")
            sys.stdout.flush()
        return None
    # Set the RDKit molecule name property using the ID from CSV for consistency if needed later
    rdkit_mol_from_smiles.SetProp("_Name", mol_id_for_output_df) 
    
    mol_with_hs = Chem.AddHs(rdkit_mol_from_smiles, addCoords=False)

    # --- Construct XYZ filename based on index ---
    # Assumes XYZ files are named dsgdb9nsd_000001.xyz, dsgdb9nsd_000002.xyz, etc.
    # (idx is 0-based, so idx+1 for 1-based filenames)
    xyz_file_basename = f"dsgdb9nsd_{idx+1:06d}.xyz" 
    xyz_filepath = os.path.join(XYZ_DIR_PATH, xyz_file_basename)
    # --- End of XYZ filename construction change ---
    
    conformer_added_successfully = False

    if os.path.exists(xyz_filepath):
        try:
            if add_conformer_from_xyz_file(mol_with_hs, xyz_filepath, idx): # Pass idx for debug prints inside
                conformer_added_successfully = True
            else:
                if idx < 5: # Limit printing
                    # This will now print if add_conformer_from_xyz_file returns false for dsgdb9nsd_xxxxxx.xyz
                    print(f"Debug Main ({mol_id_for_output_df}, idx {idx}): add_conformer_from_xyz_file returned False for {xyz_filepath}")
                    sys.stdout.flush()
        except Exception as e:
            if idx < 5:
                print(f"Debug Main ({mol_id_for_output_df}, idx {idx}): Exception during XYZ processing for {xyz_filepath}: {e}")
                sys.stdout.flush()
            conformer_added_successfully = False
    else:
        if idx < 5: # This will now check for dsgdb9nsd_xxxxxx.xyz
            print(f"Debug Main ({mol_id_for_output_df}, idx {idx}): XYZ file not found at {xyz_filepath}")
            sys.stdout.flush()
    
    processed_mol = mol_with_hs 

    try:
        Chem.SanitizeMol(processed_mol)
        canonical_smiles = Chem.MolToSmiles(processed_mol, isomericSmiles=True, canonical=True)
    except Exception as e_sanitize:
        if idx < 5:
            print(f"Debug Main ({mol_id_for_output_df}, idx {idx}): Sanitization failed: {e_sanitize}")
            sys.stdout.flush()
        return None
    
    if processed_mol.GetNumAtoms() == 0: return None
    if AllChem.GetFormalCharge(processed_mol) != 0: return None
    if len(GetMolFrags(processed_mol)) > 1: return None
    if processed_mol.GetNumConformers() > 0 and not check_interatomic_distances(processed_mol): return None

    custom_descriptors = compute_custom_descriptors(processed_mol)

    final_xyz_str_for_storage = None
    if processed_mol.GetNumConformers() > 0:
        final_xyz_str_for_storage = get_xyz_string_from_conformer(processed_mol)
    
    if idx < 5:
        print(f"Debug Main ({mol_id_for_output_df}, idx {idx}): Attempted XYZ: {xyz_filepath}. XYZ stored: {'NOT None' if final_xyz_str_for_storage else 'None'}. Conformer added: {conformer_added_successfully}. Num conformers: {processed_mol.GetNumConformers()}")
        sys.stdout.flush()

    target_properties = {prop: data_row.get(prop, np.nan) for prop in QM9_TARGET_PROPERTIES_FROM_CSV}

    return {
        "molecule_id": mol_id_for_output_df, # Keep using the ID from CSV for the final DataFrame
        "smiles_csv": smiles_from_csv,
        "canonical_smiles": canonical_smiles,
        "num_atoms": processed_mol.GetNumAtoms(),
        "xyz_coordinates": final_xyz_str_for_storage,
        **custom_descriptors,
        **target_properties
    }

# --- Main Data Processing Logic ---
def main():
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    print(f"Starting data processing using {QM9_CSV_PATH} and XYZ files from {XYZ_DIR_PATH}")
    print(f"Using up to {MAX_WORKERS} worker processes.")

    if not os.path.exists(QM9_CSV_PATH):
        print(f"Error: QM9 CSV file not found at {QM9_CSV_PATH}")
        return
    if not os.path.exists(XYZ_DIR_PATH):
        print(f"Error: XYZ directory not found at {XYZ_DIR_PATH}")
        return

    df_qm9_csv = pd.read_csv(QM9_CSV_PATH)
    print(f"Loaded {len(df_qm9_csv)} records from {QM9_CSV_PATH}.")

    # --- For focused debugging, process only a few molecules ---
    # df_to_process = df_qm9_csv.head(10) 
    # print(f"DEBUG MODE: Processing only first {len(df_to_process)} molecules.")
    # mol_args_list = list(df_to_process.iterrows())
    # --- Remove or comment out above for full run ---
    mol_args_list = list(df_qm9_csv.iterrows()) # Full run

    processed_molecules_data = []
    failed_count = 0

    # --- To debug prints without multiprocessing interference, run sequentially: ---
    # print("DEBUG MODE: Running sequentially (no ProcessPoolExecutor).")
    # for args in tqdm(mol_args_list, desc="Processing molecules sequentially"):
    #     result = process_molecule_data(args)
    #     if result is not None:
    #         processed_molecules_data.append(result)
    #     else:
    #         failed_count += 1
    # --- Remove or comment out above for full multiprocessing run ---
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor: # Multiprocessing run
        results = list(tqdm(executor.map(process_molecule_data, mol_args_list), total=len(mol_args_list), desc="Processing molecules"))

    for result in results:
        if result is not None:
            processed_molecules_data.append(result)
        else:
            failed_count += 1

    print(f"\nFinished processing.")
    print(f"Successfully processed and validated molecules: {len(processed_molecules_data)}")
    print(f"Failed or skipped molecules during processing: {failed_count}")

    if not processed_molecules_data:
        print("No molecules were successfully processed. Exiting.")
        return

    df_final = pd.DataFrame(processed_molecules_data)
    
    print("\nFirst 5 processed molecules (with descriptors and properties):")
    print(df_final.head().to_string()) # Print full head for better inspection
    # print(df_final.head(10)[['molecule_id', 'smiles_csv', 'canonical_smiles', 'num_atoms', 'xyz_coordinates']].to_string()) # More focused print
    
    print(f"\nDataFrame columns: {df_final.columns.tolist()}")

    print("\nSplitting data...")
    df_final_shuffled = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df, temp_df = np.split(df_final_shuffled, [int(0.8 * len(df_final_shuffled))])
    valid_df, test_df = np.split(temp_df.sample(frac=1, random_state=42).reset_index(drop=True), [int(0.5 * len(temp_df))])

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(valid_df)}")
    print(f"Test set size: {len(test_df)}")

    print(f"\nSaving data to {PROCESSED_DATA_DIR}...")
    train_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "train.parquet"), index=False)
    valid_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "valid.parquet"), index=False)
    test_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "test.parquet"), index=False)
    print("Data saved successfully.")
    
    manifest = {
        "source_csv_file": QM9_CSV_PATH,
        "source_xyz_directory": XYZ_DIR_PATH,
        "total_records_in_csv": len(df_qm9_csv),
        "successfully_processed_molecules": len(processed_molecules_data),
        "failed_or_skipped_molecules": failed_count,
        "data_columns": df_final.columns.tolist(),
        "train_set_size": len(train_df),
        "validation_set_size": len(valid_df),
        "test_set_size": len(test_df),
        "target_properties_from_csv": QM9_TARGET_PROPERTIES_FROM_CSV,
        "custom_descriptor_columns": list(CUSTOM_DESCRIPTORS_TO_COMPUTE.keys())
    }
    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=4)
    print(f"Manifest file saved to {MANIFEST_FILE}")

    print("\nData processing script finished.")

if __name__ == "__main__":
    main()