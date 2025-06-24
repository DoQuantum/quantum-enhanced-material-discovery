"""
Blazing‑fast ADAPT‑VQE baseline (v0.2)
• 16‑qubit DBT Hamiltonian
• PennyLane ≥ 0.35  |  lightning‑gpu (fallback: lightning.qubit)
• JAX 0.4.28 (optional)  |  SPSA inner optimiser
"""

import json
# ───────────────────────────────────────────────────────────── imports ──
import os
import pprint
import re
import time
import warnings

import jax
import numpy as jnp  # Use NumPy for params/grads
import pennylane as qml
from tqdm.auto import tqdm

# JAX acceleration is disabled to allow for SparseHamiltonian with lightning.qubit
JAX_OK = False

pp = pprint.PrettyPrinter(indent=2, compact=True).pprint
# Environment hints for maximum parallelism on CPU/GPU
os.environ.setdefault("PL_OPENMP_PARALLEL", "true")
os.environ.setdefault("OMP_NUM_THREADS", "16")
warnings.filterwarnings("ignore", message="Explicitly requested dtype.*")

# ─────────────────────────────────── Hamiltonian helpers ──


def load_sparse_hamiltonian(
    path: str, n_qubits: int = 16, coeff_cut: float = 1e-6, use_sparse: bool = True
):
    """Return a (maybe‑sparse) Hamiltonian operator and #qubits.

    • Terms with |coeff| < *coeff_cut* are pruned.
    • Tries SparseHamiltonian first; if that fails, falls back to dense.
    """
    print
    print(f"  • path       : {path}")

    with open(path) as f:
        raw = json.load(f)

    pauli = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
    coeffs, ops = [], []

    for pstr, c in raw.items():
        if abs(c) < coeff_cut:
            continue  # ➊ prune tiny contributions

        pstr = pstr.strip()
        coeffs.append(c)

        if pstr == "I":
            ops.append(qml.Identity(0))
            continue

        term_ops = [pauli[p](int(i)) for p, i in re.findall(r"([XYZ])(\d+)", pstr)]
        if not term_ops:  # malformed string safeguard
            ops.append(qml.Identity(0))
        elif len(term_ops) == 1:
            ops.append(term_ops[0])
        else:
            ops.append(qml.prod(*term_ops))

    H_dense = qml.Hamiltonian(coeffs, ops)
    print(f"  • kept terms : {len(H_dense.ops)}  (cut <{coeff_cut})")

    # JAX interface doesn't support scipy-backed SparseHamiltonian
    if not use_sparse:
        print("  • representation: dense Hamiltonian (JAX compatibility)\n")
        return H_dense, n_qubits

    # Try to obtain SparseHamiltonian
    try:
        H_mat = H_dense.sparse_matrix()
        H = qml.SparseHamiltonian(H_mat, wires=range(n_qubits))
        kind = "SparseHamiltonian"
    except Exception as e:  # pragma: no cover
        print("  • Sparse conversion failed, using dense –", e)
        H = H_dense
        kind = "dense Hamiltonian"

    print(f"  • representation: {kind}\n")
    return H, n_qubits


# ────────────────────────────── device factory ──


def make_device(n_qubits: int):
    """Return lightning‑gpu if CUDA is present; else lightning.qubit."""
    try:
        dev = qml.device("lightning.gpu", wires=n_qubits)
        print(">>> device: lightning.gpu\n")
    except Exception:  # pragma: no cover – no GPU / missing plugin
        dev = qml.device("lightning.qubit", wires=n_qubits)
        print(">>> device: lightning.qubit (CPU)\n")
    return dev


# ──────────────────────────────── ADAPT‑VQE ──


def run_adapt_vqe_baseline(
    ham_json: str = "data/dbt/qubit_hamiltonian.json",
    max_adapt: int = 50,
    vqe_steps: int = 200,
    grad_tol: float = 1e-3,
    coeff_cut: float = 1e-6,
):
    print("\n=== ADAPT‑VQE turbo launch ===")
    print(f"ham_json  = {ham_json}")
    print(f"max_adapt = {max_adapt}")
    print(f"vqe_steps = {vqe_steps}")
    print(f"grad_tol  = {grad_tol}")
    print(f"coeff_cut = {coeff_cut}")

    H, n_qubits = load_sparse_hamiltonian(
        ham_json, coeff_cut=coeff_cut, use_sparse=not JAX_OK
    )
    dev = make_device(n_qubits)

    # Hartree–Fock reference state
    ELEC = 8
    hf_state = qml.qchem.hf_state(ELEC, n_qubits)

    # Excitation pool
    singles, doubles = qml.qchem.excitations(ELEC, n_qubits)
    pool = [qml.SingleExcitation(0.0, w) for w in singles] + [
        qml.DoubleExcitation(0.0, w) for w in doubles
    ]
    print(f"pool size  : {len(pool)}\n")

    # ── cost circuit (JAX if available) ──
    interface = "jax" if JAX_OK else "auto"  # PennyLane picks autograd/torch/etc.

    @qml.qnode(dev, interface=interface, diff_method="adjoint")
    def energy(params):
        qml.BasisState(hf_state, wires=range(n_qubits))
        # Apply each excitation with its parameter.
        # Gates with p=0 are identity, so we can apply all of them.
        for p, gate in zip(params, pool):
            gate.__class__(p, wires=gate.wires)  # re‑instantiate with param
        return qml.expval(H)

    if JAX_OK:
        cost_fn = jax.jit(energy)
        grad_fn = jax.jit(jax.grad(cost_fn))
    else:
        cost_fn = energy
        grad_fn = qml.grad(cost_fn, argnum=0)

    # initial params – all zeros, keep as jnp/np array
    params = jnp.zeros(len(pool))
    e_hf = float(cost_fn(params))
    print(f"HF energy : {e_hf: .8f} Ha\n")

    # Adam inner optimiser
    opt = qml.optimize.AdamOptimizer(stepsize=0.1)  # Initial stepsize for scheduler

    trace, t0 = [], time.time()
    cost_val = e_hf

    for k in tqdm(range(1, max_adapt + 1), desc="ADAPT", unit="step"):
        iter_start = time.time()

        # 1) operator‑selection – pick largest gradient component
        print(
            f"\n[{k:02d}] Calculating gradients... (JIT compilation on first step may be slow)"
        )
        g = grad_fn(params)
        idx = int(jnp.argmax(jnp.abs(g)))
        gnorm = float(jnp.max(jnp.abs(g)))
        if gnorm < grad_tol:
            print("✔ Gradient below threshold – convergence achieved.")
            break

        # unlock chosen parameter with small random init
        params[idx] = 1e-4 * jnp.random.randn()

        # 2) VQE refinement with Adam and LR scheduler
        # Keep track of best energy to prevent optimizer instability
        best_params_in_vqe = params
        cost_val = cost_fn(params)

        # Simple exponential decay scheduler for the VQE steps
        initial_lr = 0.02
        final_lr = 0.001
        decay_rate = (final_lr / initial_lr) ** (1 / vqe_steps)
        opt.stepsize = initial_lr

        for _ in range(vqe_steps):
            params, vqe_cost_step = opt.step_and_cost(cost_fn, params, grad_fn=grad_fn)
            opt.stepsize *= decay_rate  # decay learning rate
            if vqe_cost_step < cost_val:
                cost_val = vqe_cost_step
                best_params_in_vqe = params

        params = best_params_in_vqe

        iter_time = time.time() - iter_start
        trace.append([k, float(cost_val), gnorm, iter_time])
        print(
            f"[{k:02d}]  E={cost_val: .8f}  |g|max={gnorm:.2e}  time={iter_time:4.1f}s"
        )

    total = time.time() - t0
    final_e = float(cost_val)
    print(f"\n=== Finished in {total/60: .1f} min; final E = {final_e: .8f} Ha ===")
    return trace
