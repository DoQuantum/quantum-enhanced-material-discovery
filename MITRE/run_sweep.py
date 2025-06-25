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
BASELINE_ENERGY = -1870.926454

# Create directories if they don't exist
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Prepare results file header or load existing
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
    # Fresh file: write header
    pd.DataFrame(columns=header).to_csv(RESULTS_FILE, index=False)
    done_tags = set()
else:
    # Resume mode: read existing and build set of tags to skip
    done_df = pd.read_csv(RESULTS_FILE)
    # build tags like "ADAPT-VQE-SPSA-lr0.2"
    done_tags = {
        f"{row['Ansatz']}-{row['Optimiser']}-lr{row['Hyper-params']}"
        for _, row in done_df.iterrows()
    }

# Loop over configurations
for i, config in enumerate(CONFIGURATIONS):
    job_tag = f"{config['ansatz']}-{config['optimizer']}-lr{config['lr']}"
    if job_tag in done_tags:
        print(f"--- Skipping done Job {i+1}/{len(CONFIGURATIONS)}: {job_tag} ---")
        continue

    print(f"--- Running Job {i+1}/{len(CONFIGURATIONS)}: {job_tag} ---")
    command = [
        sys.executable,
        "-u",  # unbuffered
        DRIVER_SCRIPT,
        "--ansatz",
        config["ansatz"],
        "--optimizer",
        config["optimizer"],
        "--lr",
        str(config["lr"]),
    ]

    last_line = ""
    try:
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
        ) as process:
            for line in process.stdout:
                print(line, end="")
                if line.strip():
                    last_line = line.strip()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, command, output=last_line
            )

        # Parse the final CSV row from driver output
        output_data = last_line.split(",")
        ansatz, optimiser, hyper_params = (
            output_data[0],
            output_data[1],
            output_data[2],
        )
        final_energy = float(output_data[3])
        cnot_depth = int(output_data[4])
        iterations = int(output_data[5])
        wall_time = float(output_data[6])

        delta_e_mha = (final_energy - BASELINE_ENERGY) * 1000

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

        pd.DataFrame([full_result_row]).to_csv(
            RESULTS_FILE, mode="a", header=False, index=False
        )
        print(f"\nSUCCESS: {job_tag}. Results appended.\n")

    except subprocess.CalledProcessError as e:
        error_message = "--- FAILED CONFIGURATION ---\n"
        error_message += f"Command: {' '.join(e.cmd)}\n"
        error_message += f"Exit Code: {e.returncode}\n"
        error_message += f"LAST OUTPUT:\n{e.output}\n"

        print(f"\nERROR: {job_tag} failed. See {LOG_FILE} for details.\n")
        with open(LOG_FILE, "a") as f:
            f.write(error_message)

    except (IndexError, ValueError) as e:
        error_message = "--- FAILED CONFIGURATION (Parsing Error) ---\n"
        error_message += f"Command: {' '.join(command)}\n"
        error_message += f"Error: {e}\n"
        error_message += f"Could not parse output: '{last_line}'\n"

        print(
            f"\nERROR: {job_tag} failed during output parsing. See {LOG_FILE} for details.\n"
        )
        with open(LOG_FILE, "a") as f:
            f.write(error_message)

    except FileNotFoundError:
        print(f"ERROR: Driver script not found at '{DRIVER_SCRIPT}'. Aborting.")
        break

print("Sweep complete.")
