#!/usr/bin/env python3
# ---------------------------------------------------------------------------
#  Fast ADAPT-VQE / static VQE runner  –  multi-optimiser  (no JAX)
# ---------------------------------------------------------------------------
import argparse
import json
import os
import re
import time
import warnings

import pandas as pd
import pennylane as qml
from pennylane import numpy as np

# ── runtime hints ───────────────────────────────────────────────────────────
os.environ.setdefault("PL_OPENMP_PARALLEL", "true")
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 16))
warnings.filterwarnings("ignore", message="Explicitly requested dtype.*")
energy_tol = 1e-4  # early-stop tolerance (Ha)


# ── helpers: Hamiltonian + device ──────────────────────────────────────────
def load_sparse_hamiltonian(path, n_qubits=16, coeff_cut=1e-6):
    with open(path) as f:
        raw = json.load(f)
    pauli = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
    coeffs, ops = [], []
    for pstr, c in raw.items():
        if abs(c) < coeff_cut:
            continue
        term_ops = [
            pauli[p](int(i)) for p, i in re.findall(r"([XYZ])(\d+)", pstr.strip())
        ]
        ops.append(qml.prod(*term_ops) if term_ops else qml.Identity(0))
        coeffs.append(c)
    return qml.Hamiltonian(coeffs, ops), n_qubits


def make_device(n):
    for name in ("lightning.kokkos", "lightning.gpu", "lightning.qubit"):
        try:
            return qml.device(name, wires=n)
        except qml.DeviceError:
            continue
    raise RuntimeError("No PennyLane Lightning backend available")


# ── checkpoint helpers ─────────────────────────────────────────────────────
def save_ckpt(path, theta, active, energy):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, theta=theta, active=active, energy=energy)


def load_ckpt(path):
    ckpt = np.load(path)
    return ckpt["theta"], ckpt["active"], float(ckpt["energy"])


# ---------------------------------------------------------------
#  H-QNG-Lookahead Optimiser (Hadamard-sketched Natural Gradient)
# ---------------------------------------------------------------
class HQNGLookahead:
    def __init__(
        self,
        lr=0.05,
        beta1=0.9,
        beta2=0.999,
        k_sync=6,
        alpha=0.5,
        rho=0.1,
        sigma0=0.02,
        gamma=0.101,
        eps=1e-8,
    ):
        self.lr0, self.b1, self.b2 = lr, beta1, beta2
        self.k_sync, self.alpha = k_sync, alpha
        self.rho, self.sigma0, self.gamma = rho, sigma0, gamma
        self.eps = eps
        self.t = 0
        self.m = self.v = None
        self.slow = self.fast = None
        self.H = None
        self.h_row = 0

    # ---------- utilities -------------------------------------------------
    def _hadamard(self, n):
        H = np.array([[1]])
        while H.shape[0] < n:
            H = np.block([[H, H], [H, -H]])
        return H[:n, :n]

    # ---------- main step -------------------------------------------------
    def step_and_cost(self, cost_fn, theta):
        if self.m is None:  # first call
            d = len(theta)
            self.m = np.zeros(d)
            self.v = np.zeros(d)
            self.slow = self.fast = theta.copy()
            self.H = self._hadamard(d)
            self.c0 = cost_fn(theta)

        # analytic gradient  +  diagonal “metric”
        grad = qml.grad(cost_fn)(theta)
        rms = np.sqrt(np.mean(grad**2))
        g_nat = grad / np.maximum(rms, 1e-2)

        # Hadamard SPSA
        sigma_t = self.sigma0 / max(1, self.t) ** self.gamma
        h_t = self.H[self.h_row]
        self.h_row = (self.h_row + 1) % len(theta)
        e_shift = cost_fn(theta + sigma_t * h_t) - self.c0
        g_had = e_shift * h_t / sigma_t
        g_hat = (1 - self.rho) * g_nat + self.rho * g_had

        # Adam on fast weights
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g_hat
        self.v = self.b2 * self.v + (1 - self.b2) * (g_hat * g_hat)
        m_hat = self.m / (1 - self.b1**self.t)
        v_hat = self.v / (1 - self.b2**self.t)
        self.fast = theta - self.lr0 * m_hat / (np.sqrt(v_hat) + self.eps)

        # Look-ahead sync
        if self.t % self.k_sync == 0:
            self.slow += self.alpha * (self.fast - self.slow)
            self.fast = self.slow.copy()

        self.c0 = cost_fn(self.fast)
        print(f"[H-QNG-LA] step={self.t:04d}  E={self.c0:.6f}")
        return self.fast, float(self.c0)


# ── helper: QNode factory ──────────────────────────────────────────────────
def build_energy(dev, H, pool, hf_state, n_qubits):
    @qml.qnode(dev, diff_method="adjoint")
    def _energy(theta_full):
        qml.BasisState(hf_state, wires=range(n_qubits))
        for angle, gate in zip(theta_full, pool):
            gate.__class__(angle, wires=gate.wires)
        return qml.expval(H)

    return _energy


# ── ADAPT-VQE routine ──────────────────────────────────────────────────────
def run_adapt_vqe(cfg, H, n_qubits, dev):
    ELEC = 8
    hf_state = qml.qchem.hf_state(ELEC, n_qubits)
    singles, doubles = qml.qchem.excitations(ELEC, n_qubits)
    pool = [qml.SingleExcitation(0.0, w) for w in singles] + [
        qml.DoubleExcitation(0.0, w) for w in doubles
    ]

    theta = np.zeros(len(pool), requires_grad=True)
    active = np.zeros(len(pool), dtype=bool)
    energy = build_energy(dev, H, pool, hf_state, n_qubits)
    best_E = float(energy(theta))
    print(f"Initial HF energy: {best_E:.6f} Ha")

    max_cycles = cfg.get("adapt_cycles", 2)
    vqe_steps = cfg.get("vqe_steps", 100)
    grad_tol = cfg.get("grad_tol", 5e-4)
    opt_name = cfg["optimizer"].lower()
    lr_init = cfg["lr"]

    # per-optimiser checkpoint bookkeeping
    ckpt_path = f"results/best_{cfg['ansatz']}_{cfg['optimizer']}.npz"
    best_energy_this_opt = np.inf

    def make_inner():
        if opt_name == "hqng-la":
            return HQNGLookahead(lr=lr_init, k_sync=6, alpha=0.5)
        if opt_name == "spsa":
            return qml.SPSAOptimizer(maxiter=vqe_steps, a=lr_init)
        if opt_name == "adam":
            return qml.AdamOptimizer(stepsize=lr_init)

    start = time.time()
    for cycle in range(1, max_cycles + 1):
        g_vec = qml.grad(lambda t: energy(t))(theta)
        if np.max(np.abs(g_vec)) < grad_tol:
            print("Gradient tol reached – stopping ADAPT.")
            break

        idx = np.argmax(np.abs(g_vec) * (~active))
        active[idx] = True
        theta[idx] = 1e-3 * np.random.randn()

        opt_inner = make_inner()
        prev = 1e9
        for _ in range(vqe_steps):
            theta, cost = (
                opt_inner.step_and_cost(energy, theta)
                if hasattr(opt_inner, "step_and_cost")
                else opt_inner(theta)
            )
            if abs(cost - prev) < energy_tol:
                break
            prev = cost

        best_E = cost
        print(f"Cycle {cycle:02d}: E={best_E:.6f}")

        # save best for THIS optimiser
        if best_E < best_energy_this_opt:
            best_energy_this_opt = best_E
            save_ckpt(ckpt_path, theta, active, best_E)
            print(f"↑  saved BEST for {cfg['optimizer']} → {ckpt_path}")

    wall = time.time() - start
    depth = qml.specs(build_energy(dev, H, pool, hf_state, n_qubits))(theta)[
        "resources"
    ].depth
    return dict(
        final_energy=best_E, cnot_depth=depth, iterations=len(theta), wall_time=wall
    )


# ── CLI & driver ───────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ansatz", choices=["ADAPT-VQE"], required=True)
    p.add_argument(
        "--optimizer",
        choices=["hqng-la", "SPSA", "Adam"],
        default="hqng-la",
    )
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--hamiltonian_file", default="data/dbt/qubit_hamiltonian.json")
    args = vars(p.parse_args())

    H, _ = load_sparse_hamiltonian(args["hamiltonian_file"], 16)
    dev = make_device(16)

    res = run_adapt_vqe(args, H, 16, dev)

    df = pd.DataFrame(
        [
            {
                "Ansatz": args["ansatz"],
                "Optimiser": args["optimizer"],
                "lr": args["lr"],
                "Final energy (Ha)": res["final_energy"],
                "CNOT depth": res["cnot_depth"],
                "Iterations": res["iterations"],
                "Wall-time (s)": round(res["wall_time"], 2),
            }
        ]
    )
    print(df.to_csv(index=False, header=False).strip())


if __name__ == "__main__":
    main()
