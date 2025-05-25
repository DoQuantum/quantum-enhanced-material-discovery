# Asphalt Quantum Chemistry Simulations

This project performs quantum chemistry calculations, specifically Restricted Hartree-Fock (RHF) and Variational Quantum Eigensolver (VQE) simulations, for molecular systems. It leverages PySCF for classical quantum chemistry, OpenFermion for Hamiltonian transformations, and PennyLane for quantum circuit simulations.

The primary focus is on simulating Dibenzothiophene (DBT) using different basis sets and active space selections.

## Features

*   **Molecular Geometry:** Fetches 3D molecular coordinates from PubChem using `src/get_dbt_geometry.py`.
*   **RHF Calculations:** Performs RHF calculations using PySCF to obtain molecular orbitals and energies.
*   **Active Space Selection:** Allows for the selection of an active space of orbitals and electrons for further calculations.
*   **Hamiltonian Generation:**
    *   Constructs one- and two-body integrals.
    *   Generates Fermionic Hamiltonians using OpenFermion.
    *   Transforms Fermionic Hamiltonians to Qubit Hamiltonians using the Jordan-Wigner transformation.
*   **VQE Simulation:**
    *   Implements VQE using PennyLane with the UCCSD ansatz.
    *   Uses the Adam optimizer for parameter optimization.
    *   Calculates the ground state energy of the molecule within the active space.
*   **Logging and Plotting:**
    *   Logs detailed output of the calculations to timestamped files in the `logs/` directory.
    *   Saves VQE energy convergence data (`.dat`) and plots (`.png`) to the `logs/` directory.

## Project Structure

```
asphalt-quantum/
├── .gitignore
├── requirements.txt
├── src/
│   ├── dibenzothiophene.xyz  # Example geometry file
│   ├── get_dbt_geometry.py # Script to fetch molecular geometry
│   ├── plot_vqe_convergence.py # Utility to plot VQE convergence (also integrated into main script)
│   └── run_pyscf_rhf.py    # Main script for RHF and VQE calculations
└── logs/                     # Directory for output logs, data, and plots (created automatically)
```

## Prerequisites

*   Python 3.x
*   Git

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd asphalt-quantum
    ```

2.  **Create and activate a Python virtual environment (recommended):**
    ```bash
    python3 -m venv aq_env
    source aq_env/bin/activate
    # On Windows: aq_env\Scripts\activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main script for running simulations is [`src/run_pyscf_rhf.py`](/Users/omtailor/asphalt-quantum/src/run_pyscf_rhf.py).

1.  **Prepare Molecular Geometry:**
    *   The script [`src/get_dbt_geometry.py`](/Users/omtailor/asphalt-quantum/src/get_dbt_geometry.py) can be used to fetch the XYZ coordinates for a molecule (e.g., Dibenzothiophene) and save it.
        ```bash
        python src/get_dbt_geometry.py
        ```
        This will create `dibenzothiophene.xyz` in the root directory. The main script expects this file to be in the `src/` directory, so you might need to move it or adjust the path in `run_pyscf_rhf.py`.
    *   An example `dibenzothiophene.xyz` is already provided in the `src/` directory.

2.  **Configure and Run the Main Simulation:**
    *   Open [`src/run_pyscf_rhf.py`](/Users/omtailor/asphalt-quantum/src/run_pyscf_rhf.py).
    *   Modify the configuration parameters at the bottom of the script in the `if __name__ == "__main__":` block:
        *   `current_basis_set`: e.g., `'sto-3g'`, `'6-31g*'`.
        *   `xyz_file`: Path to the molecule's XYZ file (relative to the script's location, e.g., `"dibenzothiophene.xyz"` if it's in `src/`).
        *   `num_active_orbitals_config`: Number of active spatial orbitals.
        *   `num_active_electrons_config`: Number of active electrons.
        *   `vqe_iterations`: Number of optimization steps for VQE.
    *   Run the script from the project's root directory:
        ```bash
        python src/run_pyscf_rhf.py
        ```

3.  **Outputs:**
    *   **Log files:** Detailed logs of the calculation process are saved to the `logs/` directory with names like `dbt_<basis>_<active_space>_<timestamp>_run.log`.
    *   **Energy data:** VQE energy convergence data is saved to `.dat` files in `logs/`.
    *   **Convergence plots:** Plots of VQE energy convergence are saved as `.png` files in `logs/`.

## Key Libraries Used

*   [PySCF](https://pyscf.org/): Python-based Simulations of Chemistry Framework.
*   [OpenFermion](https://quantumai.google/openfermion): Library for compiling and analyzing quantum algorithms for chemistry.
*   [PennyLane](https://pennylane.ai/): Cross-platform Python library for differentiable programming of quantum computers.
*   [NumPy](https://numpy.org/): Fundamental package for scientific computing with Python.
*   [Matplotlib](https://matplotlib.org/): Comprehensive library for creating static, animated, and interactive visualizations in Python.
*   [PubChemPy](https://pubchempy.readthedocs.io/): A Python wrapper for the PubChem PUG REST API.

## To Do / Potential Improvements

*   Parameterize more VQE settings (optimizer, ansatz details).
*   Implement other quantum algorithms.
*   Add support for different types of molecular systems or calculations.
*   More robust error handling and input validation.