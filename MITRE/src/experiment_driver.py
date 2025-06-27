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
    """Load a sparse qubit Hamiltonian from JSON."""
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
        if not term_ops:
            ops.append(qml.Identity(0))
        elif len(term_ops) == 1:
            ops.append(term_ops[0])
        else:
            # Build multi-Pauli observable via operator overloading
            from functools import reduce

            ops.append(reduce(lambda a, b: a @ b, term_ops))
    return qml.Hamiltonian(coeffs, ops), n_qubits


def make_device(n_qubits: int):
    """Creates a PennyLane device (GPU fallback to CPU)."""
    try:
        return qml.device("lightning.gpu", wires=n_qubits)
    except qml.DeviceError:
        return qml.device("lightning.qubit", wires=n_qubits)


# --- Static VQE ---
def run_static_vqe(config, H, n_qubits, dev):
    ansatz = config['ansatz'].lower()
    opt_name = config['optimizer']
    lr = config['lr']
    vqe_steps = config.get('vqe_steps', 300)
    ELEC = 8

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


# --- ADAPT-VQE with individual gradients ---
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

    @qml.qnode(dev, diff_method='adjoint', interface='autograd')
    def energy(params):
        qml.BasisState(hf_state, wires=range(n_qubits))
        for p, gate in zip(params, pool):
            gate.__class__(p, wires=gate.wires)
        return qml.expval(H)

    grad_fn = qml.grad(energy)

    params = np.zeros(len(pool), requires_grad=True)
    e_hf = float(energy(params))
    print(f"Initial HF energy: {e_hf:.6f} Ha")
    cost_val = e_hf
    trace = []
    t0 = time.time()

    opt_name = config["optimizer"].lower()

    if opt_name == "spsa":
        opt = qml.SPSAOptimizer(maxiter=vqe_steps, a=lr)
    elif opt_name == "cobyla":
        opt = qml.SciPyOptimizer(
            method="COBYLA", maxiter=vqe_steps, options={"rhobeg": lr, "disp": False}
        )
    elif opt_name == "l-bfgs-b":
        opt = qml.LBFGSBOptimizer()
    else:  # default → Adam
        opt = qml.AdamOptimizer(stepsize=lr)
    recent = []

    for cycle in range(1, max_cycles + 1):
        start = time.time()
        print(f"\n--- Cycle {cycle}/{max_cycles} ---")

        g = grad_fn(params)
        g_abs = np.abs(g)
        top = np.argsort(g_abs)[::-1]
        print("  ↳ Top gradients:")
        for i in range(min(3, len(top))):
            print(f"    - idx {top[i]}, g={g[top[i]]:.4e}")

        idx = next((i for i in top if i not in recent), top[0])
        print(f"  ↳ Selected idx {idx}, g={g[idx]:.4e}")
        recent.append(idx)
        if len(recent) > 5:
            recent.pop(0)

        if g_abs[idx] < grad_tol:
            print("✔ Gradient tol met, stopping ADAPT.")
            break

        p = params.copy()
        p[idx] = 1e-4 * np.random.randn()
        best_e = float(energy(p))
        best_p = p.copy()
        print(f"  ↳ VQE start E={best_e:.6f}")

        lr0, lrF = lr, 1e-4
        decay = (lrF / lr0) ** (1 / vqe_steps)
        opt.stepsize = lr0
        prev_e = best_e
        for s in range(vqe_steps):
            p, e = opt.step_and_cost(energy, p)
            if s < 3 or s % 50 == 0:
                print(
                    f"    [VQE {s+1:03d}] E={e:.6f}, ΔE={e-prev_e:.2e}, lr={opt.stepsize:.2e}"
                )
            opt.stepsize *= decay
            if e < best_e:
                best_e, best_p = e, p.copy()
                print(f"    ↳ New best at step {s+1}: {best_e:.6f}")
            if s > 10 and abs(e - prev_e) < energy_tol:
                print(f"    ↳ Early stop at step {s+1}, |ΔE|<{energy_tol}")
                break
            prev_e = e

        print(f"  ↳ VQE done best E={best_e:.6f}")
        if best_e < cost_val:
            params, cost_val = best_p.copy(), best_e

        trace.append([cycle, cost_val, float(g_abs[idx]), time.time() - start])

    wall = time.time() - t0

    @qml.qnode(dev, diff_method='adjoint')
    def final_circuit(params):
        qml.BasisState(hf_state, wires=range(n_qubits))
        for p, gate in zip(params, pool):
            gate.__class__(p, wires=gate.wires)
        return qml.expval(H)

    depth = qml.specs(final_circuit)(params)['resources']['depth']
    print(f"Final E={cost_val:.6f}, depth={depth}, time={wall:.1f}s")

    return {
        "final_energy": float(cost_val),
        "cnot_depth": depth,
        "iterations": len(trace),
        "wall_time": wall,
    }


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
