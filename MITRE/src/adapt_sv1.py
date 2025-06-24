"""
Verbose ADAPT-VQE baseline.
All important objects and parameters are printed as soon as they are set,
so you can verify values while the code runs.
"""

# ── imports ─────────────────────────────────────────────────────────────
import json
import os
import pathlib
import pprint
import re
import time

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdaptiveOptimizer
from qbraid.runtime import BraketProvider  # qBraid auth & billing
from tqdm import tqdm

pp = pprint.PrettyPrinter(indent=2, compact=True).pprint


# ── helpers ─────────────────────────────────────────────────────────────
def load_hamiltonian(path: str, n_qubits: int = 16):
    """Return (Hamiltonian, n_qubits) from a {Pauli-string: coeff} JSON dict."""
    print("\n>>> load_hamiltonian")
    print(f"  • path       : {path}")
    print(f"  • n_qubits   : {n_qubits}")
    with open(path) as f:
        data = json.load(f)
    print(f"  • terms found: {len(data)}")

    if not data:
        raise ValueError("Hamiltonian file is empty.")

    pauli = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
    coeffs, ops = [], []
    for pstr, c in data.items():
        coeffs.append(c)
        if pstr.strip() == "I":
            ops.append(qml.Identity(0))
            continue
        term_ops = [pauli[p](int(i)) for p, i in re.findall(r"([XYZ])(\d+)", pstr)]
        ops.append(term_ops[0] if len(term_ops) == 1 else qml.prod(*term_ops))

    H = qml.Hamiltonian(coeffs, ops)
    print(f"  • Hamiltonian object created with {len(H.ops)} ops\n")
    return H, n_qubits


def make_device(n_qubits: int):
    """
    Construct a PennyLane Braket device (SV1) billed to qBraid credits.
    """
    print("\n>>> make_device")
    provider = BraketProvider()  # auto-auth
    print("  • provider       :", provider)

    sv1 = provider.get_device("arn:aws:braket:::device/quantum-simulator/amazon/sv1")
    print("  • sv1 type       :", type(sv1))
    print("  • sv1 metadata   :")
    pp(sv1.metadata())  # → shows device_id etc.

    bucket = os.getenv("QBRAID_S3_BUCKET", "qbraid-adapt-results")
    s3_prefix = "adapt-vqe-dbt"
    print(f"  • result bucket  : {bucket}")
    print(f"  • result prefix  : {s3_prefix}")
    print(f"  • n_qubits wires : {n_qubits}")

    dev = qml.device(
        "braket.aws.qubit",  # PennyLane-Braket backend:contentReference[oaicite:1]{index=1}
        device_arn=sv1.id,  # ARN exposed as .id:contentReference[oaicite:2]{index=2}
        s3_destination_folder=(bucket, s3_prefix),
        wires=n_qubits,
        shots=0,  # analytic expectations
    )
    print("  • PennyLane device:", dev, "\n")
    return dev


# ── main routine ────────────────────────────────────────────────────────
def run_adapt_vqe_baseline(
    ham_json: str = "data/dbt/qubit_hamiltonian.json",
    max_adapt: int = 50,
    vqe_steps: int = 300,
    grad_tol: float = 1e-3,
):
    print("\n=== ADAPT-VQE baseline launch ===")
    print(f"ham_json  = {ham_json}")
    print(f"max_adapt = {max_adapt}")
    print(f"vqe_steps = {vqe_steps}")
    print(f"grad_tol  = {grad_tol}\n")

    H, n_qubits = load_hamiltonian(ham_json)

    hf_state = qml.qchem.hf_state(8, n_qubits)
    singles, doubles = qml.qchem.excitations(8, n_qubits)
    pool = [qml.SingleExcitation(0.0, w) for w in singles] + [
        qml.DoubleExcitation(0.0, w) for w in doubles
    ]
    print(">>> pool statistics")
    print(f"  • singles len   : {len(singles)}")
    print(f"  • doubles len   : {len(doubles)}")
    print(f"  • total pool ops: {len(pool)}\n")

    dev = make_device(n_qubits)

    np.random.seed(42)
    print(">>> NumPy global seed set to 42\n")

    # Hartree–Fock energy
    @qml.qnode(dev)
    def hf_energy():
        qml.BasisState(hf_state, wires=range(n_qubits))
        return qml.expval(H)

    e_hf = hf_energy()
    print(f"HF reference energy: {e_hf: .8f} Ha\n")

    # ADAPT objects
    opt = AdaptiveOptimizer(param_steps=vqe_steps, stepsize=0.1)
    print(">>> AdaptiveOptimizer")
    print(f"  • param_steps: {vqe_steps}")
    print(f"  • stepsize   : {opt.stepsize}\n")

    @qml.qnode(dev)
    def ansatz():
        qml.BasisState(hf_state, wires=range(n_qubits))
        return qml.expval(H)

    trace = []
    outdir = pathlib.Path("circuits")
    outdir.mkdir(exist_ok=True)
    t0 = time.time()

    for k in tqdm(range(1, max_adapt + 1), desc="ADAPT steps", unit="step"):
        iter_start = time.time()
        ansatz, energy, grad = opt.step_and_cost(ansatz, pool, drain_pool=True)
        iter_time = time.time() - iter_start

        cnots = sum(op.name == "CNOT" for op in ansatz.qtape.operations)
        trace.append([k, energy, k, cnots])
        print(
            f"[{k:02d}] iter_time={iter_time:5.2f}s  |grad|={abs(grad):.2e}  "
            f"E={energy: .8f}  CNOTs={cnots}"
        )

        if abs(grad) < grad_tol:
            print("• Convergence reached (gradient tolerance).\n")
            break

        if k % 5 == 0:
            qasm_path = outdir / f"iter_{k:02d}.qasm"
            with open(qasm_path, "w") as f:
                f.write(qml.qasm(ansatz)())
            print("  ↳ circuit saved to", qasm_path)

    total_time = time.time() - t0
    print(f"\n=== Finished in {total_time: .1f}s; final energy {energy: .8f} Ha ===")
    return trace
