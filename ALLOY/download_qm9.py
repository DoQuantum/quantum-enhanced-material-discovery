import requests
import os
import sys
import tarfile
import pandas as pd

def download_file_with_progress(url, local_filename, desc="file"):
    """Downloads a file from a URL to a local path with progress."""
    print(f"Attempting to download {desc}: {local_filename} from {url}")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            block_size = 8192  # 8KB
            
            with open(local_filename, 'wb') as f:
                downloaded_size = 0
                for chunk in r.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    done = int(50 * downloaded_size / total_size) if total_size > 0 else 0
                    sys.stdout.write(f"\rDownloading {desc}: [{'=' * done}{' ' * (50-done)}] {downloaded_size/1024/1024:.2f} MB / {total_size/1024/1024:.2f} MB")
                    sys.stdout.flush()
            sys.stdout.write('\n')
            print(f"{desc} downloaded successfully to {local_filename}.")
            return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {desc} ({local_filename}): {e}")
        if os.path.exists(local_filename):
            os.remove(local_filename)
            print(f"Removed partially downloaded file: {local_filename}")
        return False
    except IOError as e:
        print(f"Error writing file {local_filename}: {e}")
        return False

def extract_tar_gz(tar_gz_path, extract_to_path):
    """Extracts a .tar.gz file."""
    if not os.path.exists(tar_gz_path):
        print(f"Error: Cannot extract. File not found: {tar_gz_path}")
        return False
    print(f"Extracting {tar_gz_path} to {extract_to_path}...")
    try:
        with tarfile.open(tar_gz_path, "r:gz") as tar:
            tar.extractall(path=extract_to_path)
        print(f"Successfully extracted to {extract_to_path}.")
        return True
    except tarfile.TarError as e:
        print(f"Error extracting {tar_gz_path}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during extraction: {e}")
        return False

if __name__ == "__main__":
    # URLs provided by you
    GDB9_TAR_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
    QM9_CSV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"

    # Local filenames and paths
    local_gdb9_tar_filename = "gdb9.tar.gz"  # Will be saved in the current directory (ALLOY)
    local_qm9_csv_filename = "qm9.csv"      # Will be saved in the current directory (ALLOY)
    xyz_extract_path = "gdb9_xyz"           # Subdirectory to extract XYZ files

    download_gdb9_tar = True
    download_qm9_csv = True

    # Check for gdb9.tar.gz
    if os.path.exists(local_gdb9_tar_filename):
        user_input = input(f"File '{local_gdb9_tar_filename}' already exists. Redownload? (y/n): ").strip().lower()
        if user_input != 'y':
            download_gdb9_tar = False
            print("Skipping download of gdb9.tar.gz.")
    
    if download_gdb9_tar:
        if not download_file_with_progress(GDB9_TAR_URL, local_gdb9_tar_filename, desc="gdb9.tar.gz"):
            print("Failed to download gdb9.tar.gz. Please check the URL or your connection.")
            sys.exit(1) # Exit if essential file download fails

    # Check for qm9.csv
    if os.path.exists(local_qm9_csv_filename):
        user_input = input(f"File '{local_qm9_csv_filename}' already exists. Redownload? (y/n): ").strip().lower()
        if user_input != 'y':
            download_qm9_csv = False
            print("Skipping download of qm9.csv.")

    if download_qm9_csv:
        if not download_file_with_progress(QM9_CSV_URL, local_qm9_csv_filename, desc="qm9.csv"):
            print("Failed to download qm9.csv. Please check the URL or your connection.")
            sys.exit(1) # Exit if essential file download fails
            
    # Extract gdb9.tar.gz
    if os.path.exists(local_gdb9_tar_filename):
        if os.path.exists(xyz_extract_path):
            user_input_extract = input(f"Extraction path '{xyz_extract_path}' already exists. Re-extract? (y/n): ").strip().lower()
            if user_input_extract == 'y':
                print(f"Removing existing extraction directory: {xyz_extract_path}")
                import shutil
                shutil.rmtree(xyz_extract_path) # Be careful with rmtree
                os.makedirs(xyz_extract_path, exist_ok=True)
                extract_tar_gz(local_gdb9_tar_filename, xyz_extract_path)
            else:
                print(f"Skipping extraction of {local_gdb9_tar_filename}.")
        else:
            os.makedirs(xyz_extract_path, exist_ok=True)
            extract_tar_gz(local_gdb9_tar_filename, xyz_extract_path)
    else:
        print(f"{local_gdb9_tar_filename} not found. Cannot extract.")


    print("\n--- Download and Extraction Summary ---")
    if os.path.exists(local_qm9_csv_filename):
        print(f"CSV file for properties: ./{local_qm9_csv_filename}")
        try:
            df_check = pd.read_csv(local_qm9_csv_filename)
            print(f"  Successfully read CSV. Shape: {df_check.shape}. Columns: {df_check.columns.tolist()[:15]}...") # Print first 15 cols
        except Exception as e:
            print(f"  Could not verify CSV content: {e}")
    else:
        print(f"CSV file ./{local_qm9_csv_filename} was not downloaded or is missing.")

    if os.path.exists(xyz_extract_path) and os.listdir(xyz_extract_path):
        print(f"XYZ files extracted to: ./{xyz_extract_path}/ (contains {len(os.listdir(xyz_extract_path))} items)")
    elif os.path.exists(local_gdb9_tar_filename):
        print(f"XYZ files are in ./{local_gdb9_tar_filename} but not extracted to ./{xyz_extract_path}/")
    else:
        print(f"XYZ file archive ./{local_gdb9_tar_filename} was not downloaded or is missing.")
        
    print("\nNext step: Modify 'process_qm9.py' to use these new data sources.")
    print(f"  - Read properties and SMILES from '{local_qm9_csv_filename}'.")
    print(f"  - Optionally, load 3D coordinates from XYZ files in '{xyz_extract_path}'.")