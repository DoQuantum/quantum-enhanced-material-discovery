#!/usr/bin/env python3
import time
import pennylane as qml
from pennylane import numpy as np, qchem
from bcqnso_optimizer import BCQNSO

def get_molecule_hamiltonian(symbols, coords, basis="sto-3g", charge=0):
    coords_arr = np.array(coords, dtype=float).flatten()
    print("[get_hamiltonian] coords_arr:", coords_arr.shape)
    H, qubits = qchem.molecular_hamiltonian(
        symbols, coords_arr, charge=charge, basis=basis
    )
    print("[get_hamiltonian] wires:", qubits)
    return H, qubits

def make_ansatz(k, wires, init_state, shape):
    def ansatz_flat(params_flat):
        params = params_flat.reshape(shape)
        qml.BasisState(init_state, wires=wires)
        qml.kUpCCGSD(
            params,
            wires=wires,
            k=k,
            delta_sz=0,
            init_state=init_state,
        )
    return ansatz_flat

def run_vqe(
    symbols,
    coords,
    k,
    optimizer_cls,
    lr=0.1,
    steps=5,            # short run
    init_shots=256,
    charge=0,
    basis="sto-3g",
):
    print("\n=== run_vqe start ===")
    # 1) Hamiltonian & device
    H, n_wires = get_molecule_hamiltonian(symbols, coords, basis, charge)
    dev = qml.device("default.qubit", wires=n_wires + 1, shots=init_shots)
    print("[run_vqe] device wires:", dev.wires, "shots:", init_shots)

    # 2) Electron count
    geom = np.array(coords, dtype=float).flatten()
    mol = qchem.Molecule(symbols, geom, charge=charge, basis_name=basis)
    n_electrons = mol.n_electrons
    print("[run_vqe] n_electrons:", n_electrons)

    # 3) HF state
    hf_state = np.array(qchem.hf_state(n_electrons, n_wires), dtype=int)
    print("[run_vqe] hf_state:", hf_state)

    # 4) Parameter shape
    shape = qml.kUpCCGSD.shape(k=k, n_wires=n_wires, delta_sz=0)
    n_params = shape[0] * shape[1]
    print(f"[run_vqe] kUpCCGSD shape: {shape}, total params: {n_params}")

    # 5) Build ansatz
    ansatz_fn = make_ansatz(k, wires=list(range(n_wires)), init_state=hf_state, shape=shape)

    # 6) QNode
    @qml.qnode(dev, interface="autograd")
    def circuit(params_flat):
        print(" [circuit] params_flat shape:", params_flat.shape)
        ansatz_fn(params_flat)
        return qml.expval(H)

    # 7) Init params
    params = np.zeros(n_params)
    print("[run_vqe] initial params vector length:", len(params))

    # 8) Pick optimizer
    if optimizer_cls is BCQNSO:
        blocks = [slice(i * shape[1], (i + 1) * shape[1]) for i in range(shape[0])]
        opt = BCQNSO(blocks=blocks, init_shots=init_shots, lr=lr)
    else:
        if optimizer_cls is qml.SPSAOptimizer:
            opt = optimizer_cls(maxiter=steps, a=lr)
        elif optimizer_cls is qml.QNGOptimizer:
            opt = optimizer_cls(stepsize=lr)
        else:
            opt = optimizer_cls(stepsize=lr)
    print("[run_vqe] optimizer:", opt)

    # 9) Optimization loop with timing
    start_total = time.time()
    for i in range(steps):
        print(f"\n-- Step {i+1}/{steps}: calling opt.step() --")
        t0 = time.time()
        params = opt.step(circuit, params)
        t1 = time.time()
        print(f"-- Step {i+1} returned in {t1-t0:.3f} s; params shape now {getattr(params, 'shape', None)} --")
        params = np.array(params)  # enforce array
    energy = circuit(params)
    total_time = time.time() - start_total
    print(f"\n=== run_vqe end: energy={energy:.6f}, total time={total_time:.3f}s ===")
    return energy, total_time

if __name__ == "__main__":
    tests = {
        "H4":  (["H"] * 4, [[0,0,0],[0,0,1],[1,0,0],[1,1,0]]),
        "H2O": (["O","H","H"], [[0,0,0],[0,0.757,0.586],[0,-0.757,0.586]]),
        "N2":  (["N","N"], [[0,0,0],[0,0,1.097]])
    }
    optimizers = [BCQNSO, qml.AdamOptimizer, qml.SPSAOptimizer, qml.QNGOptimizer]

    for name, (symbols, coords) in tests.items():
        print(f"\n=== Testing {name} ===")
        for opt_cls in optimizers:
            label = opt_cls.__name__ if opt_cls is not BCQNSO else "BCQNSO"
            try:
                e, t = run_vqe(symbols, coords, k=2,
                               optimizer_cls=opt_cls, lr=0.1)
                print(f"  {label}: Energy = {e:.6f}, Time = {t:.3f}s")
            except Exception as exc:
                print(f"  {label} failed: {exc}")
