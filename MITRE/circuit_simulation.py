import os

import numpy as np
import pennylane as qml
import yaml
from qbraid.runtime import BraketProvider  # qBraid auth & billing

# --- 1. Load Frozen Config ---
with open("logs/best_config_25Jun.yaml") as f:
    config = yaml.safe_load(f)
theta = np.array(config["parameter_vector"])
active = np.array(config["active_gates"], dtype=bool)
ELEC = 8
N_QUBITS = 16


# --- 2. Construct the Braket SV1 Device ---
def make_device(n_qubits: int):
    provider = BraketProvider()
    sv1 = provider.get_device("arn:aws:braket:::device/quantum-simulator/amazon/sv1")
    bucket = os.getenv("QBRAID_S3_BUCKET", "qbraid-adapt-results")
    s3_prefix = "adapt-vqe-dbt"
    dev = qml.device(
        "braket.aws.qubit",
        device_arn=sv1.id,
        s3_destination_folder=(bucket, s3_prefix),
        wires=n_qubits,
        shots=0,  # analytic expectation values (not sampled)
    )
    return dev


dev = make_device(N_QUBITS)
print("Device created:", dev)

# --- 3. Rebuild Excitation Pool (ADAPT-VQE) ---
singles, doubles = qml.qchem.excitations(ELEC, N_QUBITS)
hf_state = qml.qchem.hf_state(ELEC, N_QUBITS)
act_indices = np.where(active)[0]
act_thetas = theta[act_indices]


# --- 4. Define the Final Circuit ---
@qml.qnode(dev)
def final_ansatz_circuit():
    qml.BasisState(hf_state, wires=range(N_QUBITS))
    for idx, angle in zip(act_indices, act_thetas):
        if idx < len(singles):
            qml.SingleExcitation(angle, singles[idx])
        else:
            qml.DoubleExcitation(angle, doubles[idx - len(singles)])
    # Example observable: PauliZ on first qubit (change as needed)
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# --- 5. Run the Circuit and Get Results ---
results = final_ansatz_circuit()
print("Expectation values:", results)
