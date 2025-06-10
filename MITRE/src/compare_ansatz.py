import csv
import os
import datetime
import numpy as np
from run_pyscf_rhf import run_rhf_get_integrals_qubit_hamiltonian, run_vqe_pennylane


def compare_ansatzes(xyz_file="dibenzothiophene.xyz", basis="sto-3g", orbitals=8, electrons=8, iterations=100):
    mf_obj, qubit_h, E_nuc, active_idx, n_elec, n_orb, *_ = run_rhf_get_integrals_qubit_hamiltonian(
        xyz_file,
        basis_set=basis,
        n_active_orbitals_target=orbitals,
        n_active_electrons_target=electrons,
    )
    if qubit_h is None:
        print("Failed to generate Hamiltonian")
        return

    ansatzes = [
        ("uccsd", {"ansatz_type": "uccsd"}),
        ("hea", {"ansatz_type": "hea", "hea_layers": 2}),
        ("gucc", {"ansatz_type": "gucc", "hea_layers": 2}),
        ("hva", {"ansatz_type": "hva", "hea_layers": 2}),
        ("adapt", {"ansatz_type": "adapt", "hea_layers": 2}),
        ("ala", {"ansatz_type": "ala", "hea_layers": 2}),
        ("brickwork", {"ansatz_type": "brickwork", "hea_layers": 2}),
        ("qaoa", {"ansatz_type": "qaoa", "hea_layers": 2}),
        ("topology", {"ansatz_type": "topology", "hea_layers": 2}),
        ("tensor", {"ansatz_type": "tensor", "hea_layers": 2}),
        ("genetic", {"ansatz_type": "genetic", "hea_layers": 2}),
        ("basic", {"ansatz_type": "basic", "hea_layers": 2}),
        ("random", {"ansatz_type": "random", "hea_layers": 2}),
    ]

    results = []
    for name, kwargs in ansatzes:
        energy = run_vqe_pennylane(
            qubit_h,
            n_elec,
            n_orb,
            basis,
            max_iterations=iterations,
            **kwargs,
        )
        results.append((name, energy))

    baseline_energy = results[0][1]
    qubits = 2 * n_orb
    print("\nComparison of Ansatzes")
    print(f"{'Ansatz':<10}{'Energy (Ha)':>15}{'Improvement':>15}{'Qubits':>10}")
    for name, energy in results:
        improvement = energy - baseline_energy
        print(f"{name:<10}{energy:>15.8f}{improvement:>15.8f}{qubits:>10}")

    # Save results to a CSV file under logs/
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_file = f"logs/ansatz_comparison_{timestamp}.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ansatz", "energy_ha", "improvement", "qubits"])
        for name, energy in results:
            writer.writerow([name, f"{energy:.8f}", f"{(energy - baseline_energy):.8f}", qubits])
    print(f"\nSaved comparison results to {csv_file}")


if __name__ == "__main__":
    compare_ansatzes()
