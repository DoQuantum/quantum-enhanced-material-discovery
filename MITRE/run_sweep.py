import os
import subprocess
import sys

import pandas as pd

# --- Configuration Matrix ---
CONFIGURATIONS = []
ANSATZE = ["ADAPT-VQE", "k-UpCCGSD", "HEA"]
OPTIMIZERS = {
    "SPSA": [0.2, 0.1, 0.05],
    "Adam": [0.2, 0.1, 0.05],
    "COBYLA": [0.2, 0.1, 0.05],  # Represents step-size 'rhobeg'
    "L-BFGS-B": [0.0],  # No learning rate, use a placeholder
}

for ansatz in ANSATZE:
    for opt_name, lrs in OPTIMIZERS.items():
        for lr in lrs:
            CONFIGURATIONS.append({"ansatz": ansatz, "optimizer": opt_name, "lr": lr})

# --- Run Sweep ---
RESULTS_FILE = "results/optim_grid_tmp.csv"
LOG_FILE = "logs/optim_failures.txt"
DRIVER_SCRIPT = "src/experiment_driver.py"

# IMPORTANT: Replace this with the reference energy from your adapt_trace.csv
BASELINE_ENERGY = -1870.926454  # Example value, please verify

# Create directories if they don't exist
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Prepare results file header
header = [
    "Ansatz",
    "Optimiser",
    "Hyper-params",
    "Final energy (Ha)",
    "ΔE vs baseline (mHa)",
    "CNOT depth",
    "Iterations",
    "Wall-time (s)",
]
if not os.path.exists(RESULTS_FILE):
    pd.DataFrame(columns=header).to_csv(RESULTS_FILE, index=False)

# Loop over configurations
for i, config in enumerate(CONFIGURATIONS):
    job_tag = f"{config['ansatz']}-{config['optimizer']}-lr{config['lr']}"
    print(f"--- Running Job {i+1}/{len(CONFIGURATIONS)}: {job_tag} ---")

    # Construct the command to run the experiment driver
    command = [
        sys.executable,  # Use the same python interpreter that is running this script
        DRIVER_SCRIPT,
        "--ansatz",
        config['ansatz'],
        "--optimizer",
        config['optimizer'],
        "--lr",
        str(config['lr']),
    ]

    try:
        # Execute the driver script and capture its output
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,  # Raises an exception for non-zero exit codes
            encoding='utf-8',
        )

        # The driver prints a single CSV row. Parse it.
        # Format: Ansatz,Optimiser,Hyper-params,Final energy (Ha),CNOT depth,Iterations,Wall-time (s)
        output_data = result.stdout.strip().split(',')

        # Extract metrics from the driver's output
        ansatz, optimiser, hyper_params = output_data[0], output_data[1], output_data[2]
        final_energy = float(output_data[3])
        cnot_depth = int(output_data[4])
        iterations = int(output_data[5])
        wall_time = float(output_data[6])

        # Calculate the energy difference in milliHartrees
        delta_e_mha = (final_energy - BASELINE_ENERGY) * 1000

        # Assemble the full row of metrics in the correct order
        full_result_row = {
            "Ansatz": ansatz,
            "Optimiser": optimiser,
            "Hyper-params": hyper_params,
            "Final energy (Ha)": final_energy,
            "ΔE vs baseline (mHa)": delta_e_mha,
            "CNOT depth": cnot_depth,
            "Iterations": iterations,
            "Wall-time (s)": wall_time,
        }

        # Append the results to the temporary CSV file
        pd.DataFrame([full_result_row]).to_csv(
            RESULTS_FILE, mode='a', header=False, index=False
        )
        print(
            f"SUCCESS: {job_tag}. Final Energy: {final_energy:.6f} Ha. Results appended.\n"
        )

    except subprocess.CalledProcessError as e:
        # Log any failed configurations for follow-up
        error_message = "--- FAILED CONFIGURATION ---\n"
        error_message += f"Command: {' '.join(e.cmd)}\n"
        error_message += f"Exit Code: {e.returncode}\n"
        error_message += f"STDOUT:\n{e.stdout}\n"
        error_message += f"STDERR:\n{e.stderr}\n"

        print(f"ERROR: {job_tag} failed. See {LOG_FILE} for details.\n")
        with open(LOG_FILE, 'a') as f:
            f.write(error_message)
    except FileNotFoundError:
        print(f"ERROR: Driver script not found at '{DRIVER_SCRIPT}'. Aborting.")
        break

print("Sweep complete.")
