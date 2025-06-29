from qiskit import QuantumCircuit

# Load the QASM file
qc = QuantumCircuit.from_qasm_file("circuits/vqe_baseline_unmapped.qasm")

# Gate counts and depth
counts = qc.count_ops()
depth = qc.depth()

# Focus on 1‑ and 2‑qubit gates
one_qb = sum(counts.get(g, 0) for g in ["u1", "u2", "u3", "x", "h", "s", "sx", "t", "tdg"] )
two_qb = counts.get("cx", 0)

print(counts)
print(f"1‑qubit gates: {one_qb}")
print(f"2‑qubit gates (CNOTs): {two_qb}")
print(f"Depth: {depth}")
