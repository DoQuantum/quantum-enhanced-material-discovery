import os
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_experiments.library.characterization import LocalReadoutError
from qiskit.qasm2 import dumps

# === SETUP ===
# Use your IBM Quantum account credentials if not already done.
# QiskitRuntimeService() will auto-load from stored IBMQ credentials.
backend_name = "ibm_brisbane"   # Update if different
service = QiskitRuntimeService()

backend = service.backend(backend_name)

qubits = list(range(16))
shots_per = 1024  # Standard is 1024, can be 4096 if you want higher stats

# === GENERATE CALIBRATION CIRCUITS ===
exp = LocalReadoutError(physical_qubits=qubits, backend=backend)
exp.set_run_options(shots=shots_per)
calib_circs = exp.circuits()   # List of QuantumCircuit objects

# === SAVE CIRCUITS ===
os.makedirs("circuits/calib_set_1", exist_ok=True)
for i, circ in enumerate(calib_circs):
    qasm_str = dumps(circ)  # QASM string for the circuit
    with open(f"circuits/calib_set_1/calib_{i}.qasm", "w") as f:
        f.write(qasm_str)

# === CALCULATE STATS ===
n_calib = len(calib_circs)
total_shots = n_calib * shots_per

# --- SET YOUR COST PER SHOT! (Update this as needed) ---
cost_per_shot = 1  # <-- Replace with the real value for IBM Heron (credits/shot)
total_credits = total_shots * cost_per_shot

print(f"Calibration circuits: {n_calib}")
print(f"Shots per circuit: {shots_per}")
print(f"Total calibration shots: {total_shots}")
print(f"Estimated credit cost: {total_credits}")

# === LOG TO FILE ===
with open("logs/preflight_stats.txt", "a") as f:
    f.write("\n# Calibration circuits\n")
    f.write(f"Calibration circuits: {n_calib}\n")
    f.write(f"Shots per circuit: {shots_per}\n")
    f.write(f"Total calibration shots: {total_shots}\n")
    f.write(f"Estimated credit cost: {total_credits}\n")

print("Step 5 complete! Calibration circuits generated and stats logged.")
