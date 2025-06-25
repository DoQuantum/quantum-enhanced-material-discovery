"""
Blazing-fast ADAPT-VQE baseline with debug instrumentation (v0.2_debug_v3)
• 16-qubit DBT Hamiltonian
• PennyLane ≥ 0.35 | lightning-gpu (fallback: lightning.qubit)
• Debug prints, uses Autograd via Pennylane numpy for trainable params
"""

import json
import os
import pprint
import re
import time
import warnings

import pennylane as qml
import pennylane.numpy as np  # Autograd-compatible numpy
from tqdm.auto import tqdm

# Environment hints for maximum parallelism
os.environ.setdefault("PL_OPENMP_PARALLEL", "true")
os.environ.setdefault("OMP_NUM_THREADS", "16")
warnings.filterwarnings("ignore", message="Explicitly requested dtype.*")

pp = pprint.PrettyPrinter(indent=2, compact=True).pprint


def load_sparse_hamiltonian(path: str, n_qubits: int = 16, coeff_cut: float = 1e-6):
    print(f"  • path       : {path}")
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
    print(f"  • kept terms : {len(H.ops)} (cut <{coeff_cut})")
    print("  • representation: dense Hamiltonian\n")
    return H, n_qubits


def make_device(n_qubits: int):
    try:
        dev = qml.device("lightning.gpu", wires=n_qubits)
        print(">>> device: lightning.gpu\n")
    except qml.DeviceError:
        dev = qml.device("lightning.qubit", wires=n_qubits)
        print(">>> device: lightning.qubit (CPU)\n")
    return dev


def run_adapt_vqe_baseline(
    ham_json: str = "data/dbt/qubit_hamiltonian.json",
    max_adapt: int = 50,
    vqe_steps: int = 300,
    grad_tol: float = 5e-3,
    coeff_cut: float = 1e-6,
):
    print("\n=== ADAPT-VQE turbo launch ===")
    print(f"ham_json  = {ham_json}")
    print(f"max_adapt = {max_adapt}")
    print(f"vqe_steps = {vqe_steps}")
    print(f"grad_tol  = {grad_tol}")
    print(f"coeff_cut = {coeff_cut}\n")

    H, n_qubits = load_sparse_hamiltonian(ham_json, coeff_cut=coeff_cut)
    dev = make_device(n_qubits)

    ELEC = 8
    hf_state = qml.qchem.hf_state(ELEC, n_qubits)
    singles, doubles = qml.qchem.excitations(ELEC, n_qubits)
    pool = [qml.SingleExcitation(0.0, wires=w) for w in singles] + [
        qml.DoubleExcitation(0.0, wires=w) for w in doubles
    ]
    print(f"pool size  : {len(pool)}\n")

    @qml.qnode(dev, diff_method="adjoint", interface="autograd")
    def energy(params):
        qml.BasisState(hf_state, wires=range(n_qubits))
        for p, gate_class in zip(params, pool):
            gate_class.__class__(p, wires=gate_class.wires)
        return qml.expval(H)

    cost_fn = energy
    grad_fn = qml.grad(cost_fn)

    params = np.zeros(len(pool))
    e_hf = float(cost_fn(params))
    print(f"HF energy : {e_hf: .8f} Ha")
    print(f"  ↳ Initial params[0:8] = {params[:8]}\n")

    opt = qml.AdamOptimizer(stepsize=0.01)
    trace, t0 = [], time.time()
    cost_val = e_hf
    recent = []

    for k in tqdm(range(1, max_adapt + 1), desc="ADAPT", unit="step"):
        start = time.time()
        print(f"\n[{k:02d}] Calculating gradients...")
        g = grad_fn(params)
        g_abs = np.abs(g)
        top = np.argsort(g_abs)[::-1]
        print("  ↳ Top gradients:")
        for i in range(3):
            print(f"    - idx {top[i]}, g={g[top[i]]:.4e}")

        idx = next((i for i in top if i not in recent), top[0])  # diversity pick
        print(f"  ↳ Selected idx {idx}, g={g[idx]:.4e}")
        recent.append(idx)
        if len(recent) > 5:
            recent.pop(0)

        if g_abs[idx] < grad_tol:
            print("✔ Gradient tol met, stopping ADAPT.")
            break

        p = params.copy()
        p[idx] = 1e-4 * np.random.randn()
        best_e = float(cost_fn(p))
        best_p = p.copy()
        print(f"  ↳ VQE start E={best_e:.6f}")

        lr0, lrF = 0.01, 1e-4
        decay = (lrF / lr0) ** (1 / vqe_steps)
        opt.stepsize = lr0
        prev_e = best_e
        for s in range(vqe_steps):
            p, e = opt.step_and_cost(cost_fn, p)
            if s < 3 or s % 50 == 0:
                print(
                    f"    [VQE {s+1:03d}] E={e:.6f}, ΔE={e-prev_e:.2e}, lr={opt.stepsize:.2e}"
                )
            opt.stepsize *= decay
            if e < best_e:
                best_e, best_p = e, p.copy()
                print(f"    ↳ New best at step {s+1}: {best_e:.6f}")
            # early-stop based on consecutive-step change, not best change
            if s > 10 and abs(e - prev_e) < 1e-3:
                print(f"  ↳ VQE converged at step {s+1}, |ΔE|={abs(e-prev_e):.2e}")
                break
            prev_e = e

        print(f"  ↳ VQE done best E={best_e:.6f}")
        if best_e < cost_val:
            params, cost_val = best_p.copy(), best_e
        trace.append([k, cost_val, float(g_abs[idx]), time.time() - start])
        print(
            f"[{k:02d}]  E={cost_val: .6f}, |g|={g_abs[idx]:.2e}, dt={time.time()-start:.1f}s"
        )

    print(f"\n=== Done in {(time.time()-t0)/60:.2f}min; final E={cost_val:.6f}Ha ===")
    return trace
