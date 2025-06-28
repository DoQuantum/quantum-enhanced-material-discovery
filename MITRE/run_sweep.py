#!/usr/bin/env python3
# ------------------------------------------------------------------
#  Grid-sweep driver   —   now includes the novel “1PH-SPSA” optimiser
# ------------------------------------------------------------------
import os
import subprocess
import sys

import pandas as pd

# ── Configuration matrix ───────────────────────────────────────────
CONFIGURATIONS = []
ANSATZE = ["ADAPT-VQE", "HEA", "k-UpCCGSD"]

OPTIMIZERS = {
    #   optimizer :  list of learning-rate / step-size values to test
    "hqng-la": [0.05],  # HQNG-LA: learning rate
    "Adam": [0.2, 0.1, 0.05],
    "SPSA": [0.2, 0.1, 0.05],
    "COBYLA": [0.2, 0.1, 0.05],  # 'rhobeg' step size
    "1PH-SPSA": [0.2, 0.1, 0.05],  # ★ new, one-eval Hadamard SPSA
}

for ansatz in ANSATZE:
    for opt, lrs in OPTIMIZERS.items():
        for lr in lrs:
            CONFIGURATIONS.append({"ansatz": ansatz, "optimizer": opt, "lr": lr})

# ── Paths & constants ──────────────────────────────────────────────
RESULTS_FILE = "results/optim_grid_tmp1.csv"
LOG_FILE = "logs/optim_failures.txt"
DRIVER_SCRIPT = "src/experiment_driver.py"
BASELINE_ENERGY = -1870.926454  # reference from adapt_trace.csv

os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

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

# resume support
if not os.path.exists(RESULTS_FILE):
    pd.DataFrame(columns=header).to_csv(RESULTS_FILE, index=False)
    done_tags = set()
else:
    done = pd.read_csv(RESULTS_FILE)
    done_tags = {
        f"{r['Ansatz']}-{r['Optimiser']}-lr{r['Hyper-params']}"
        for _, r in done.iterrows()
    }

# ── Sweep loop ─────────────────────────────────────────────────────
for i, cfg in enumerate(CONFIGURATIONS, 1):
    tag = f"{cfg['ansatz']}-{cfg['optimizer']}-lr{cfg['lr']}"
    if tag in done_tags:
        print(f"--- Skipping done Job {i}/{len(CONFIGURATIONS)}: {tag} ---")
        continue

    print(f"--- Running Job {i}/{len(CONFIGURATIONS)}: {tag} ---")
    cmd = [
        sys.executable,
        "-u",
        DRIVER_SCRIPT,
        "--ansatz",
        cfg["ansatz"],
        "--optimizer",
        cfg["optimizer"],
        "--lr",
        str(cfg["lr"]),
    ]

    last_line = ""
    try:
        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        ) as proc:
            for line in proc.stdout:
                print(line, end="")
                if line.strip():
                    last_line = line.strip()

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd, output=last_line)

        # parse the one-line CSV emitted by experiment_driver.py
        out = last_line.split(",")
        ansatz, optimiser, hyper = out[0], out[1], out[2]
        final_E = float(out[3])
        cnot_d = int(out[4])
        iters = int(out[5])
        wall_sec = float(out[6])
        delta_mHa = (final_E - BASELINE_ENERGY) * 1000

        row = {
            "Ansatz": ansatz,
            "Optimiser": optimiser,
            "Hyper-params": hyper,
            "Final energy (Ha)": final_E,
            "ΔE vs baseline (mHa)": delta_mHa,
            "CNOT depth": cnot_d,
            "Iterations": iters,
            "Wall-time (s)": wall_sec,
        }
        pd.DataFrame([row]).to_csv(RESULTS_FILE, mode="a", header=False, index=False)
        print(f"\nSUCCESS: {tag}. Results appended.\n")

    except subprocess.CalledProcessError as e:
        msg = (
            f"--- FAILED CONFIGURATION ---\nCommand: {' '.join(e.cmd)}\n"
            f"Exit Code: {e.returncode}\nLAST OUTPUT:\n{e.output}\n"
        )
        print(f"\nERROR: {tag} failed. See {LOG_FILE} for details.\n")
        with open(LOG_FILE, "a") as f:
            f.write(msg)

    except (IndexError, ValueError) as e:
        msg = (
            f"--- FAILED CONFIGURATION (Parsing Error) ---\n"
            f"Command: {' '.join(cmd)}\nError: {e}\n"
            f"Could not parse output: '{last_line}'\n"
        )
        print(
            f"\nERROR: {tag} failed during output parsing. See {LOG_FILE} for details.\n"
        )
        with open(LOG_FILE, "a") as f:
            f.write(msg)

    except FileNotFoundError:
        print(f"ERROR: Driver script not found at '{DRIVER_SCRIPT}'. Aborting.")
        break

print("Sweep complete.")
