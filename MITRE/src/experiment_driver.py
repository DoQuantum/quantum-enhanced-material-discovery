import argparse
import json
import os
import re
import time
import warnings

import pandas as pd
import pennylane as qml
from pennylane import numpy as np

# --- Environment and Utility Functions (from adapt_vqe_driver.py) ---

# Set environment hints for performance
os.environ.setdefault("PL_OPENMP_PARALLEL", "true")
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 16))
warnings.filterwarnings("ignore", message="Explicitly requested dtype.*")


def load_sparse_hamiltonian(path: str, n_qubits: int = 16, coeff_cut: float = 1e-6):
    """Loads a Hamiltonian from a JSON file, as seen in adapt_vqe_driver.py."""
    with open(path) as f:
        raw = json.load(f)
    pauli = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
    coeffs, ops = [], []
    for pstr, c in raw.items():
        if abs(c) < coeff_cut:
            continue
        pstr = pstr.strip()
        coeffs.append(c)
        if pstr == "I":
            ops.append(qml.Identity(0))
            continue
        term_ops = [pauli[p](int(i)) for p, i in re.findall(r"([XYZ])(\d+)", pstr)]
        ops.append(
            qml.Identity(0)
            if not term_ops
            else term_ops[0] if len(term_ops) == 1 else qml.prod(*term_ops)
        )
    H = qml.Hamiltonian(coeffs, ops)
    return H, n_qubits


def make_device(n_qubits: int):
    """Creates a PennyLane device, defaulting to lightning.qubit as requested."""
    return qml.device("lightning.gpu", wires=n_qubits)


def get_cnot_depth(qnode, params):
    """Helper to calculate CNOT depth from a circuit."""
    try:
        specs = qml.specs(qnode)(params)
        # The depth calculation might need adjustment based on your precise gate set
        # and how PennyLane reports it for your device and ansatz.
        return specs.get("resources", {}).get("depth", 0)
    except Exception:
        return -1  # Return -1 if specs calculation fails


# --- VQE Logic ---


def run_static_vqe(config, H, n_qubits, dev):
    """Runs a standard VQE for a static ansatz (HEA, k-UpCCGSD)."""
    ansatz_name = config['ansatz']
    optimizer_name = config['optimizer']
    lr = config['lr']
    vqe_steps = 300
    ELEC = 8  # Number of electrons from your existing driver

    # 1. Define Ansatz
    if ansatz_name.lower() == "hea":
        n_layers = 4  # A reasonable default for HEA
        shape = qml.templates.StronglyEntanglingLayers.shape(
            n_layers=n_layers, n_wires=n_qubits
        )

        @qml.qnode(dev)
        def circuit(params):
            qml.templates.StronglyEntanglingLayers(params, wires=range(n_qubits))
            return qml.expval(H)

    elif ansatz_name.lower() == "kupccgsd":
        k = 2  # As specified
        hf_state = qml.qchem.hf_state(ELEC, n_qubits)
        singles, doubles = qml.qchem.excitations(ELEC, n_qubits)
        num_params = len(singles) + len(doubles)
        shape = (k, num_params)

        @qml.qnode(dev)
        def circuit(params):
            qml.BasisState(hf_state, wires=range(n_qubits))
            qml.templates.UpCCGSD(
                params, wires=range(n_qubits), s_wires=singles, d_wires=doubles, k=k
            )
            return qml.expval(H)

    else:
        raise ValueError(f"Unsupported static ansatz: {ansatz_name}")

    # 2. Define Optimizer
    if optimizer_name.lower() == "spsa":
        opt = qml.SPSAOptimizer(maxiter=vqe_steps, a=lr)
    elif optimizer_name.lower() == "adam":
        opt = qml.AdamOptimizer(stepsize=lr)
    elif optimizer_name.lower() == "cobyla":
        opt = qml.COBYLAOptimizer(maxiter=vqe_steps, rhobeg=lr)
    elif optimizer_name.lower() == "l-bfgs-b":
        opt = qml.LBFGSBOptimizer()
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # 3. Run VQE
    print(f"Running VQE for {ansatz_name} with {optimizer_name} (lr={lr})")
    params = np.random.uniform(0, 2 * np.pi, size=shape, requires_grad=True)
    cost_fn = circuit

    t0 = time.time()
    for step in range(vqe_steps):
        params, energy = opt.step_and_cost(cost_fn, params)
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1:03d}: Energy = {energy:.8f} Ha")

    wall_time = time.time() - t0
    print(f"Final energy: {energy:.8f} Ha")

    # 4. Calculate final metrics
    cnot_depth = get_cnot_depth(circuit, params)

    return {
        "final_energy": float(energy),
        "cnot_depth": cnot_depth,
        "iterations": vqe_steps,
        "wall_time": wall_time,
    }


def run_adapt_vqe(config, H, n_qubits, dev):
    """Runs a full ADAPT-VQE algorithm with a configurable inner-loop optimizer."""
    # --- Configuration ---
    optimizer_name = config['optimizer']
    lr = config['lr']
    max_adapt_cycles = 50
    vqe_steps_per_cycle = 100
    grad_tol = 1e-4  # Gradient tolerance for convergence
    ELEC = 8

    # --- Initialization ---
    hf_state = qml.qchem.hf_state(ELEC, n_qubits)
    singles, doubles = qml.qchem.excitations(ELEC, n_qubits)
    # Full operator pool
    pool = [qml.SingleExcitation(0.0, wires=w) for w in singles] + [
        qml.DoubleExcitation(0.0, wires=w) for w in doubles
    ]

    active_operators = []
    active_params = np.array([], requires_grad=True)
    energy = 0.0  # Will be updated after the first cycle

    t0 = time.time()

    # --- Main ADAPT-VQE Loop ---
    for cycle in range(max_adapt_cycles):
        print(f"\n--- ADAPT Cycle {cycle + 1}/{max_adapt_cycles} ---")

        # 1. Select operator with the largest gradient
        # First, prepare the current ansatz statevector
        @qml.qnode(dev)
        def previous_ansatz_state(params):
            qml.BasisState(hf_state, wires=range(n_qubits))
            for i, op in enumerate(active_operators):
                op(params[i])
            return qml.state()

        current_state = previous_ansatz_state(active_params)

        # Calculate gradients for all operators remaining in the pool
        gradients = []
        for op in pool:
            # Define a temporary QNode to compute the gradient for one operator
            @qml.qnode(dev, diff_method="parameter-shift")
            def grad_eval_circuit(param):
                qml.StatePrep(current_state, wires=range(n_qubits))
                op(param)
                return qml.expval(H)

            # The gradient at theta=0 is what we need
            grad_val = qml.grad(grad_eval_circuit, argnum=0)(0.0)
            gradients.append(np.abs(grad_val))

        max_grad_idx = np.argmax(gradients)
        max_grad = gradients[max_grad_idx]
        print(f"Max gradient in pool: {max_grad:.8f}")

        # 2. Check stopping condition
        if max_grad < grad_tol:
            print(
                f"ADAPT-VQE converged. Gradient {max_grad:.8f} is below tolerance {grad_tol}."
            )
            break

        # 3. Add the best operator to the ansatz
        best_op = pool.pop(max_grad_idx)
        active_operators.append(best_op)
        print(f"Adding operator to ansatz: {best_op.name} on wires {best_op.wires}")

        # Append a new parameter (initialized to 0.0) for the new operator
        active_params = np.append(active_params, 0.0)
        active_params.requires_grad = True

        # 4. Run VQE to optimize all active parameters
        @qml.qnode(dev, diff_method="adjoint")
        def vqe_circuit(params):
            qml.BasisState(hf_state, wires=range(n_qubits))
            for i, op in enumerate(active_operators):
                op(params[i])
            return qml.expval(H)

        # Instantiate the chosen optimizer for this cycle's VQE run
        if optimizer_name.lower() == "spsa":
            opt = qml.SPSAOptimizer(maxiter=vqe_steps_per_cycle, a=lr)
        elif optimizer_name.lower() == "adam":
            opt = qml.AdamOptimizer(stepsize=lr)
        elif optimizer_name.lower() == "cobyla":
            opt = qml.COBYLAOptimizer(maxiter=vqe_steps_per_cycle, rhobeg=lr)
        else:  # L-BFGS-B
            opt = qml.LBFGSBOptimizer()

        print(f"Optimizing {len(active_params)} parameters with {optimizer_name}...")
        for step in range(vqe_steps_per_cycle):
            active_params, energy = opt.step_and_cost(vqe_circuit, active_params)

        print(f"Cycle {cycle + 1} complete. Energy = {energy:.8f} Ha")

    # --- Finalization ---
    if cycle == max_adapt_cycles - 1:
        print("ADAPT-VQE finished by reaching max cycles.")

    wall_time = time.time() - t0

    @qml.qnode(dev)
    def final_circuit(params):
        qml.BasisState(hf_state, wires=range(n_qubits))
        for i, op in enumerate(active_operators):
            op(params[i])
        return qml.expval(H)

    cnot_depth = get_cnot_depth(final_circuit, active_params)
    # The number of "iterations" for ADAPT is the number of operators added
    iterations = len(active_operators)

    return {
        "final_energy": float(energy),
        "cnot_depth": cnot_depth,
        "iterations": iterations,
        "wall_time": wall_time,
    }


def main():
    """Main driver to run a single experiment from the command line."""
    parser = argparse.ArgumentParser(
        description="Run a VQE experiment with a specific configuration."
    )
    parser.add_argument(
        "--ansatz", type=str, required=True, choices=["ADAPT-VQE", "k-UpCCGSD", "HEA"]
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        required=True,
        choices=["SPSA", "Adam", "COBYLA", "L-BFGS-B"],
    )
    parser.add_argument(
        "--lr", type=float, required=True, help="Learning rate or step size."
    )
    parser.add_argument(
        "--hamiltonian_file", type=str, default="data/dbt/qubit_hamiltonian.json"
    )
    args = parser.parse_args()

    config = vars(args)

    # --- Setup ---
    n_qubits = 16
    H, _ = load_sparse_hamiltonian(args.hamiltonian_file, n_qubits=n_qubits)
    dev = make_device(n_qubits)

    # --- Run ---
    if config['ansatz'].lower() == "adapt-vqe":
        results = run_adapt_vqe(config, H, n_qubits, dev)
    else:
        results = run_static_vqe(config, H, n_qubits, dev)

    # --- Output ---
    hyper_param_tag = (
        f"lr={config['lr']}" if config['optimizer'].lower() != 'l-bfgs-b' else 'default'
    )

    output_data = {
        "Ansatz": config['ansatz'],
        "Optimiser": config['optimizer'],
        "Hyper-params": hyper_param_tag,
        "Final energy (Ha)": results['final_energy'],
        "CNOT depth": results['cnot_depth'],
        "Iterations": results['iterations'],
        "Wall-time (s)": round(results['wall_time'], 2),
    }

    # Print as a single CSV row (without header) for the batch script
    df = pd.DataFrame([output_data])
    print(df.to_csv(index=False, header=False).strip())


if __name__ == "__main__":
    main()
