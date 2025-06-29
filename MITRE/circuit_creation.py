import pennylane as qml
import numpy as np
import yaml
from qiskit import QuantumCircuit
from qiskit import qasm2  # Requires qiskit[openqasm2-export] or qiskit-qasm2
import os

ELEC = 8
N_QUBITS = 16

# Load configuration
with open("logs/best_config_25Jun.yaml") as f:
    config = yaml.safe_load(f)

theta = np.array(config["parameter_vector"])
active = np.array(config["active_gates"], dtype=bool)

# Generate excitation lists
singles, doubles = qml.qchem.excitations(ELEC, N_QUBITS)

# Set up the device
dev = qml.device("default.qubit", wires=N_QUBITS)

# Prepare Hartree–Fock state
hf_state = qml.qchem.hf_state(ELEC, N_QUBITS)
act_indices = np.where(active)[0]
act_thetas = theta[act_indices]

@qml.qnode(dev)
def final_circuit():
    qml.BasisState(hf_state, wires=range(N_QUBITS))
    for idx, angle in zip(act_indices, act_thetas):
        if idx < len(singles):
            qml.SingleExcitation(angle, singles[idx])
        else:
            qml.DoubleExcitation(angle, doubles[idx - len(singles)])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# Build the circuit
final_circuit()  # Build the tape

tape = final_circuit.qtape
qasm_str = tape.to_openqasm()  # PennyLane → OpenQASM

# Load into Qiskit
qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_str)

# Export to OpenQASM 2
openqasm2_text = qasm2.dumps(qiskit_circuit)

os.makedirs("circuits", exist_ok=True)
with open("circuits/vqe_baseline_unmapped.qasm", "w") as f:
    f.write(openqasm2_text)

print("✅ Exported QASM 2.0 format successfully.")