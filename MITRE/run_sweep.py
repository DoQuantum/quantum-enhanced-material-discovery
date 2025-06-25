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
    # The "-u" flag ensures unbuffered output from the child process
    command = [
        sys.executable,
        "-u",
        DRIVER_SCRIPT,
        "--ansatz",
        config['ansatz'],
        "--optimizer",
        config['optimizer'],
        "--lr",
        str(config['lr']),
    ]

    last_line = ""
    try:
        # Use Popen to stream output in real-time
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout to see errors immediately
            text=True,
            encoding='utf-8',
            bufsize=1,  # Line-buffered
        ) as process:
            # Read and print output line by line as it is generated
            for line in process.stdout:
                print(line, end='')  # Print the line to the console
                # The final CSV output is the last non-empty line
                if line.strip():
                    last_line = line.strip()

        # Check if the process exited with an error after it has finished
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, command, output=last_line
            )

        # The driver prints a single CSV row as its last output. Parse it.
        output_data = last_line.split(',')

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
        print(f"\nSUCCESS: {job_tag}. Results appended.\n")

    except subprocess.CalledProcessError as e:
        # Log any failed configurations for follow-up
        error_message = "--- FAILED CONFIGURATION ---\n"
        error_message += f"Command: {' '.join(e.cmd)}\n"
        error_message += f"Exit Code: {e.returncode}\n"
        error_message += f"LAST OUTPUT:\n{e.output}\n"

        print(f"\nERROR: {job_tag} failed. See {LOG_FILE} for details.\n")
        with open(LOG_FILE, 'a') as f:
            f.write(error_message)
    except (IndexError, ValueError) as e:
        # This will catch errors if the last line was not the expected CSV
        error_message = "--- FAILED CONFIGURATION (Output Parsing Error) ---\n"
        error_message += f"Command: {' '.join(command)}\n"
        error_message += f"Error: {e}\n"
        error_message += f"Could not parse the last line of output: '{last_line}'\n"

        print(
            f"\nERROR: {job_tag} failed during output parsing. See {LOG_FILE} for details.\n"
        )
        with open(LOG_FILE, 'a') as f:
            f.write(error_message)
    except FileNotFoundError:
        print(f"ERROR: Driver script not found at '{DRIVER_SCRIPT}'. Aborting.")
        break

print("Sweep complete.")
