import json
import os
import re

import numpy as np
import pennylane as qml
import yaml
from qbraid.runtime import BraketProvider

# 1. Load frozen params
with open("logs/best_config_25Jun.yaml") as f:
    config = yaml.safe_load(f)
theta = np.array(config["parameter_vector"])
active = np.array(config["active_gates"], dtype=bool)

ELEC = 8
N_QUBITS = 16

# 2. Load Hamiltonian
with open("data/dbt/qubit_hamiltonian.json") as f:
    h_raw = json.load(f)
pauli = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
coeffs, ops = [], []

for pstr, c in h_raw.items():
    matches = re.findall(r"([XYZ])(\d+)", pstr)
    term_ops = [pauli[p](int(i)) for p, i in matches]
    op = qml.prod(*term_ops) if term_ops else qml.Identity(0)
    coeffs.append(c)
    ops.append(op)
H = qml.Hamiltonian(coeffs, ops)

# 3. Prepare excitation pool
singles, doubles = qml.qchem.excitations(ELEC, N_QUBITS)
hf_state = qml.qchem.hf_state(ELEC, N_QUBITS)
act_indices = np.where(active)[0]
act_thetas = theta[act_indices]


# 4. Device and circuit
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
        shots=0,  # analytic
    )
    return dev


dev = make_device(N_QUBITS)
print("Device created:", dev)


@qml.qnode(dev)
def ansatz_circuit(params):
    qml.BasisState(hf_state, wires=range(N_QUBITS))
    for idx, angle in zip(act_indices, params):
        if idx < len(singles):
            qml.SingleExcitation(angle, singles[idx])
        else:
            qml.DoubleExcitation(angle, doubles[idx - len(singles)])
    return [qml.expval(op) for op in H.ops]


# 5. Compute energy
expvals = ansatz_circuit(act_thetas)
energy = sum(c * v for c, v in zip(H.coeffs, expvals))
print("VQE Energy:", energy)
