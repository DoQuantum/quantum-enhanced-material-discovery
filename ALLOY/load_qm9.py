import deepchem as dc
import os

def load_qm9_dataset(data_dir="qm9_data"):
    """
    Loads the QM9 dataset using DeepChem's MoleculeNet.
    Saves the data to a specified directory.

    Args:
        data_dir (str): Directory to save/load QM9 data.

    Returns:
        tuple: tasks, (train_dataset, valid_dataset, test_dataset), transformers
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # QM9 tasks (properties)
    # See https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html#qm9-datasets
    # We are primarily interested in the molecules themselves (SMILES and 3D coords)
    # and potentially one or two properties for initial testing.
    # For now, let's load all tasks and then decide which ones to use.
    tasks, datasets, transformers = dc.molnet.load_qm9(
        data_dir=data_dir,
        splitter='random', # We will re-split later as per your plan
        reload=True,       # Reload from disk if already downloaded
        # We don't need a featurizer at this stage,
        # as we'll handle SMILES and XYZ separately.
        # featurizer='Raw' # or leave as default and extract SMILES/coords
    )
    return tasks, datasets, transformers

if __name__ == "__main__":
    print("Loading QM9 dataset...")
    tasks, (train_dataset, valid_dataset, test_dataset), transformers = load_qm9_dataset()

    print(f"\nQM9 Tasks (Properties): {tasks}")
    print(f"Number of tasks: {len(tasks)}")

    # Let's inspect the first few molecules from the training set
    print(f"\nTotal training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(valid_dataset)}")
    print(f"Total test samples: {len(test_dataset)}")

    print("\nInspecting the first 3 molecules from the training set:")
    for i in range(min(3, len(train_dataset))):
        molecule_entry = train_dataset.ids[i]  # SMILES string
        properties = train_dataset.y[i]      # Property values
        # 3D coordinates are typically stored in train_dataset.X if a featurizer
        # that preserves them is used, or need to be accessed from the raw files.
        # DeepChem's default QM9 loader might not directly expose XYZ in .X
        # if a graph featurizer is implicitly used.
        # We will need to ensure we get the XYZ files.

        print(f"\nSample {i+1}:")
        print(f"  SMILES (ID): {molecule_entry}")
        print(f"  Properties: {properties}")
        # Note: Accessing XYZ coordinates directly from the loaded `Dataset` object
        # might require a specific featurizer or looking into how DeepChem stores them
        # for QM9. The `data_dir` ("qm9_data") will contain the raw SDF files,
        # which include both SMILES and 3D coordinates. We'll parse these directly
        # in the next steps as per your plan (SMILES with RDKit, XYZ with ASE).

    print(f"\nDataset files are stored in the 'qm9_data' directory.")
    print("This directory contains gdb9.sdf (main data) and other supporting files.")
    print("We will parse gdb9.sdf in the subsequent steps for SMILES and XYZ coordinates.")

    # To confirm XYZ availability, let's check the contents of the data_dir
    qm9_data_path = "qm9_data"
    if os.path.exists(os.path.join(qm9_data_path, "gdb9.sdf")):
        print("\nFound gdb9.sdf. This file contains SMILES and 3D coordinates.")
    else:
        print(f"\nWarning: gdb9.sdf not found in {qm9_data_path}. Check the download.")

    print("\nScript finished.")