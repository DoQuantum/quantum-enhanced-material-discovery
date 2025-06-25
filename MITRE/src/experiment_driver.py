import argparse
import json
import os
import re
import time
import warnings

import pandas as pd
import pennylane as qml
from pennylane import numpy as np

# Set environment variables for performance
os.environ.setdefault("PL_OPENMP_PARALLEL", "true")
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 16))
warnings.filterwarnings("ignore", message="Explicitly requested dtype.*")

# Energy tolerance for early stopping (in Hartree units)
energy_tol = 1e-4


def load_sparse_hamiltonian(path: str, n_qubits: int = 16, coeff_cut: float = 1e-6):
    with open(path) as f:
        raw = json.load(f)
    pauli = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
    coeffs, ops = [], []
    for pstr, c in raw.items():
        if abs(c) < coeff_cut:
            continue
        coeffs.append(c)
        pstr = pstr.strip()
        if pstr == "I":
            ops.append(qml.Identity(0))
            continue
        term_ops = [pauli[p](int(i)) for p, i in re.findall(r"([XYZ])(\d+)", pstr)]
        ops.append(
            qml.Identity(0)
            if not term_ops
            else term_ops[0] if len(term_ops) == 1 else qml.prod(*term_ops)
        )
    return qml.Hamiltonian(coeffs, ops), n_qubits


def make_device(n_qubits: int):
    """Creates a PennyLane device."""
    return qml.device("lightning.gpu", wires=n_qubits)


# --- Static VQE ---
def run_static_vqe(config, H, n_qubits, dev):
    ansatz = config['ansatz'].lower()
    opt_name = config['optimizer']
    lr = config['lr']
    vqe_steps = config.get('vqe_steps', 300)
    ELEC = 8

    # Define ansatz QNode
    if ansatz == "hea":
        n_layers = 4
        shape = qml.templates.StronglyEntanglingLayers.shape(
            n_layers=n_layers, n_wires=n_qubits
        )

        @qml.qnode(dev, diff_method='adjoint')
        def circuit(params):
            qml.templates.StronglyEntanglingLayers(params, wires=range(n_qubits))
            return qml.expval(H)

    elif ansatz == "k-upccgsd":
        k = 2
        hf_state = qml.qchem.hf_state(ELEC, n_qubits)
        singles, doubles = qml.qchem.excitations(ELEC, n_qubits)
        shape = (k, len(singles) + len(doubles))

        @qml.qnode(dev, diff_method='adjoint')
        def circuit(params):
            qml.BasisState(hf_state, wires=range(n_qubits))
            qml.templates.UpCCGSD(
                params, wires=range(n_qubits), s_wires=singles, d_wires=doubles, k=k
            )
            return qml.expval(H)

    else:
        raise ValueError(f"Unsupported static ansatz: {config['ansatz']}")

    # Choose optimizer
    if opt_name.lower() == 'spsa':
        opt = qml.SPSAOptimizer(maxiter=vqe_steps, a=lr)
    elif opt_name.lower() == 'adam':
        opt = qml.AdamOptimizer(stepsize=lr)
    elif opt_name.lower() == 'cobyla':
        opt = qml.COBYLAOptimizer(maxiter=vqe_steps, rhobeg=lr)
    else:
        opt = qml.LBFGSBOptimizer()

    print(f"Running static VQE ({config['ansatz']}) with {config['optimizer']}...")
    params = np.random.uniform(0, 2 * np.pi, size=shape, requires_grad=True)
    prev_energy = None
    t0 = time.time()
    for i in range(vqe_steps):
        params, energy = opt.step_and_cost(circuit, params)
        print(f" [Static VQE Step {i+1}/{vqe_steps}] E = {energy:.6f}", flush=True)
        if prev_energy is not None and abs(energy - prev_energy) < energy_tol:
            print(f"  ↳ Early stop at step {i+1}, |ΔE| < {energy_tol}")
            break
        prev_energy = energy
    wall = time.time() - t0
    depth = qml.specs(circuit)(params)['resources']['depth']
    print(f"Final E = {energy:.8f}, depth = {depth}, time = {wall:.1f}s")
    return {
        "final_energy": float(energy),
        "cnot_depth": depth,
        "iterations": i + 1,
        "wall_time": wall,
    }


# --- ADAPT-VQE with batch gradient ---
def run_adapt_vqe(config, H, n_qubits, dev):
    lr = config['lr']
    max_cycles = config.get('adapt_cycles', 50)
    vqe_steps = config.get('vqe_steps', 100)
    grad_tol = config.get('grad_tol', 1e-4)
    ELEC = 8

    hf_state = qml.qchem.hf_state(ELEC, n_qubits)
    singles, doubles = qml.qchem.excitations(ELEC, n_qubits)
    pool = [qml.SingleExcitation(0.0, wires=w) for w in singles] + [
        qml.DoubleExcitation(0.0, wires=w) for w in doubles
    ]
    active_ops = []
    active_params = np.array([], requires_grad=True)

    @qml.qnode(dev, diff_method='adjoint')
    def hf_energy():
        qml.BasisState(hf_state, wires=range(n_qubits))
        return qml.expval(H)

    print(f"Initial HF energy: {hf_energy():.6f} Ha")
    t0 = time.time()

    from pennylane.workflow import construct_batch

    for cycle in range(max_cycles):
        print(f"\n--- Cycle {cycle+1}/{max_cycles} ---")

        @qml.qnode(dev, diff_method='parameter-shift')
        def pool_qnode(params):
            qml.BasisState(hf_state, wires=range(n_qubits))
            for i, op in enumerate(active_ops):
                op.__class__(active_params[i], wires=op.wires)
            for i, op in enumerate(pool):
                op.__class__(params[i], wires=op.wires)
            return qml.expval(H)

        zero_params = np.zeros(len(pool))
        tapes, fn = construct_batch(pool_qnode, level='gradient')(zero_params)
        results = qml.execute(tapes, dev)
        raw_grads = fn(results)
        gradients = list(map(abs, raw_grads))
        print(f" Gradients: {gradients}")

        if max(gradients) < grad_tol:
            print("Converged by gradient tolerance.")
            break

        idx = int(np.argmax(gradients))
        active_ops.append(pool.pop(idx))
        active_params = np.append(active_params, np.random.randn())

        @qml.qnode(dev, diff_method='adjoint')
        def vqe_circuit(params):
            qml.BasisState(hf_state, wires=range(n_qubits))
            for i, op in enumerate(active_ops):
                op.__class__(params[i], wires=op.wires)
            return qml.expval(H)

        # Inner VQE with Adam and early stopping
        inner_opt = qml.AdamOptimizer(stepsize=lr)
        prev_e = None
        for i in range(vqe_steps):
            active_params, e = inner_opt.step_and_cost(vqe_circuit, active_params)
            print(f"    [VQE {cycle+1} Step {i+1}/{vqe_steps}] E = {e:.6f}", flush=True)
            if prev_e is not None and abs(e - prev_e) < energy_tol:
                print(f"    ↳ Early stop at step {i+1}, |ΔE| < {energy_tol}")
                break
            prev_e = e

    wall = time.time() - t0

    @qml.qnode(dev, diff_method='adjoint')
    def final_circuit(params):
        qml.BasisState(hf_state, wires=range(n_qubits))
        for i, op in enumerate(active_ops):
            op.__class__(params[i], wires=op.wires)
        return qml.expval(H)

    depth = qml.specs(final_circuit)(active_params)['resources']['depth']
    print(f"Final E={e:.6f}, depth={depth}, time={wall:.1f}s")
    return {
        "final_energy": float(e),
        "cnot_depth": depth,
        "iterations": len(active_ops),
        "wall_time": wall,
    }


# --- Main Driver ---
def main():
    parser = argparse.ArgumentParser(description="Run VQE or ADAPT-VQE")
    parser.add_argument(
        "--ansatz", choices=["ADAPT-VQE", "HEA", "k-UpCCGSD"], required=True
    )
    parser.add_argument(
        "--optimizer", choices=["SPSA", "Adam", "COBYLA", "L-BFGS-B"], required=True
    )
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--hamiltonian_file", default="data/dbt/qubit_hamiltonian.json")
    args = parser.parse_args()

    config = vars(args)
    print(f"--- Job: {config['ansatz']}-{config['optimizer']}-lr{config['lr']} ---")

    H, _ = load_sparse_hamiltonian(config['hamiltonian_file'], 16)
    dev = make_device(16)

    if config['ansatz'] == 'ADAPT-VQE':
        results = run_adapt_vqe(config, H, 16, dev)
    else:
        results = run_static_vqe(config, H, 16, dev)

    df = pd.DataFrame(
        [
            {
                "Ansatz": config['ansatz'],
                "Optimiser": config['optimizer'],
                "lr": config['lr'],
                "Final energy (Ha)": results['final_energy'],
                "CNOT depth": results['cnot_depth'],
                "Iterations": results['iterations'],
                "Wall-time (s)": round(results['wall_time'], 2),
            }
        ]
    )
    print(df.to_csv(index=False, header=False).strip())


if __name__ == "__main__":
    main()
