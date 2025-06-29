#!/usr/bin/env python3
import json

import yaml
from qiskit import QuantumCircuit
from qiskit.primitives import (  # <— Estimator import :contentReference[oaicite:3]{index=3}
    StatevectorEstimator,
)
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper

# 1. Load and strip transpiled circuit
qc = QuantumCircuit.from_qasm_file("circuits/vqe_baseline_unmapped.qasm")
qc_nom = QuantumCircuit(qc.num_qubits)
for inst in qc.data:
    op = inst.operation if hasattr(inst, 'operation') else inst[0]
    if op.name != "measure":
        qc_nom.append(op, inst.qubits, [])

# 2. Prepend Hartree–Fock initial state
with open("logs/best_config_25Jun.yaml") as f:
    cfg = yaml.safe_load(f)
num_qubits = qc_nom.num_qubits
num_spin_orbitals = qc_nom.num_qubits
num_spatial_orbitals = num_spin_orbitals // 2
num_particles = (cfg.get("num_alpha", 8), cfg.get("num_beta", 8))
hf_circ = HartreeFock(
    num_spatial_orbitals=num_spatial_orbitals,
    num_particles=num_particles,
    qubit_mapper=JordanWignerMapper(),
)

full_qc = hf_circ.compose(qc_nom)

# 3. Rebuild Hamiltonian with full-length, little-endian labels
with open("data/dbt/qubit_hamiltonian.json") as f:
    raw = json.load(f)
pauli_list = []
for short_label, coeff in raw.items():
    full = ['I'] * num_qubits
    if short_label != "I":
        for term in short_label.split():
            p, idx = term[0], int(term[1:])
            full[idx] = p
    label = "".join(full)[::-1]
    pauli_list.append((label, coeff))
op = SparsePauliOp.from_list(pauli_list)
print(f"Rebuilt Hamiltonian with {len(pauli_list)} terms on {num_qubits} qubits")

# 4. Use the V2 StatevectorEstimator (no backend arg needed)
est = StatevectorEstimator()  # <— Setup Estimator :contentReference[oaicite:4]{index=4}
job = est.run(
    [(full_qc, op)]
)  # <— Single-call expectation :contentReference[oaicite:5]{index=5}
result = job.result()
pub = result[0]
energy = float(pub.data.evs)
print(f"Noise-free transpiled energy (StatevectorEstimator): {energy:.12f} Ha")


# 5. (Optional) HF reference for sanity check
#    Note: You could also use Estimator here instead of Statevector for consistency
job_hf = est.run([(hf_circ, op)])
res_hf = job_hf.result()
pub_hf = res_hf[0]
hf_energy = float(pub_hf.data.evs)
print(f"Hartree–Fock reference energy (StatevectorEstimator): {hf_energy:.12f} Ha")

# 6. Sanity assertion
assert (
    abs(energy - cfg["final_energy"]) < 1e-5
), f"Energy mismatch: saw {energy}, expected {cfg['final_energy']}"
print("✅ Energy matches the optimized value within numerical tolerance.")
