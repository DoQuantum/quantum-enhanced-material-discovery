import json
import os
import re
import time

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdaptiveOptimizer
from tqdm import tqdm


def load_hamiltonian_from_json(filepath):
    """
    Loads a qubit Hamiltonian from a JSON file and returns
    (Hamiltonian, num_qubits).
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    if not data:
        raise ValueError("Hamiltonian file is empty.")
    num_qubits = 16
    coeffs, ops = [], []
    pauli_map = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}

    for pauli_str, coeff in data.items():
        if not isinstance(coeff, (int, float)):
            raise ValueError(f"Invalid coefficient for {pauli_str}: {coeff}")
        coeffs.append(coeff)

        if pauli_str.strip() == "I":
            ops.append(qml.Identity(0))
            continue

        terms = re.findall(r"([XYZ])(\d+)", pauli_str)
        if not terms:
            raise ValueError(f"Could not parse Pauli string: {pauli_str}")

        term_ops = []
        for op_char, idx in terms:
            wire = int(idx)
            if wire >= num_qubits:
                raise ValueError(f"Index {wire} out of bounds for {num_qubits} qubits.")
            term_ops.append(pauli_map[op_char](wire))

        ops.append(term_ops[0] if len(term_ops) == 1 else qml.prod(*term_ops))

    H = qml.Hamiltonian(coeffs, ops)
    print(f"Loaded Hamiltonian with {len(coeffs)} terms on {num_qubits} qubits.")
    return H, num_qubits


def run_adapt_vqe_baseline(max_adapt=50, vqe_steps=300):
    """
    Runs ADAPT-VQE using PennyLane's AdaptiveOptimizer.

    Returns:
        trace (list): [[step, energy, n_params, cnot_count], ...]
    """
    H, num_qubits = load_hamiltonian_from_json("data/dbt/qubit_hamiltonian.json")

    # HF ref state and operator pool
    hf_state = qml.qchem.hf_state(8, num_qubits)
    singles, doubles = qml.qchem.excitations(8, num_qubits)
    pool = [qml.SingleExcitation(0.0, wires=w) for w in singles] + [
        qml.DoubleExcitation(0.0, wires=w) for w in doubles
    ]

    print(f"Initial operator pool size: {len(pool)}")

    # Device & seed
    dev = qml.device("lightning.qubit", wires=num_qubits)
    np.random.seed(42)

    # Print HF energy
    @qml.qnode(dev)
    def hf_energy():
        qml.BasisState(hf_state, wires=range(num_qubits))
        return qml.expval(H)

    e_hf = hf_energy()
    print(f"HF reference energy: {e_hf:.8f} Ha\n")

    # Adaptive optimizer
    opt = AdaptiveOptimizer(param_steps=vqe_steps, stepsize=0.1)

    # Growing ansatz QNode (no params in signature)
    @qml.qnode(dev)
    def adapt_circuit():
        qml.BasisState(hf_state, wires=range(num_qubits))
        return qml.expval(H)

    # Prepare trace & output dir
    trace = []
    os.makedirs("circuits", exist_ok=True)

    print("--- Starting ADAPT-VQE baseline simulation ---")
    start_total = time.time()

    for step in tqdm(range(1, max_adapt + 1), desc="ADAPT-VQE steps", unit="step"):
        iter_start = time.time()
        print(f"\n>>> Iteration {step}")

        # show a snapshot of first few pool ops
        sample = pool[:3]
        print(
            "  Pool sample:",
            [f"{op.name}{tuple(op.wires)}" for op in sample],
            "... (total pool size:",
            len(pool),
            ")",
        )

        # Grow & optimize one operator
        adapt_circuit, energy, gradient = opt.step_and_cost(
            adapt_circuit, pool, drain_pool=True
        )

        iter_time = time.time() - iter_start
        print(f"  Time for this iteration: {iter_time:.2f} s")

        # Inspect the new tape
        tape = adapt_circuit.qtape
        total_ops = len(tape.operations)
        cnot_count = sum(1 for op in tape.operations if op.name == "CNOT")
        print(f"  Total ansatz operations: {total_ops}")
        print(f"  CNOT count: {cnot_count}")

        # Record trace: after step i, we have i parameters
        trace.append([step, energy, step, cnot_count])
        print(f"  Energy = {energy:.8f} Ha | Selected gradient = {gradient:.5f}")

        # Save QASM every 5 steps
        if step % 5 == 0:
            fname = f"circuits/iter_{step:02d}.qasm"
            with open(fname, "w") as f:
                f.write(qml.qasm(adapt_circuit)())
            print(f"  -> Saved circuit to {fname}")

        # Convergence check
        if abs(gradient) < 1e-3:
            print(f"Converged after {step} ADAPT steps (|grad| < 1e-3).")
            break

    total_time = time.time() - start_total
    print(f"\nCompleted in {total_time:.2f} s. Final energy: {energy:.8f} Ha")

    return trace
