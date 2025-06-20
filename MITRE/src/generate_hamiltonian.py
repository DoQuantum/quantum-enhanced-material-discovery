import json
import os
import time

import numpy as np
from openfermion.ops.operators import FermionOperator, QubitOperator
from openfermion.transforms.op_conversions import jordan_wigner
from pyscf import gto, scf

from .utils import (
    check_if_molecule_is_too_big,
    get_fermion_operator,
)


def get_active_space_integrals(mol, mf, n_active_orbitals, n_active_electrons):
    """
    Selects the active space and returns the corresponding integrals.
    This version uses the robust integral transformation from the legacy script.
    """
    n_mo_full = mf.mo_coeff.shape[1]
    n_electrons_full = mol.nelectron
    n_occupied_full = n_electrons_full // 2
    n_occupied_active = n_active_electrons // 2
    n_virtual_active = n_active_orbitals - n_occupied_active

    homo_idx = n_occupied_full - 1
    lumo_idx = n_occupied_full

    occupied_indices = list(range(homo_idx - n_occupied_active + 1, homo_idx + 1))
    virtual_indices = list(range(lumo_idx, lumo_idx + n_virtual_active))
    active_indices = sorted(occupied_indices + virtual_indices)

    print(f"Selected active space MO indices: {active_indices}")

    # --- Robust Integral Transformation (from legacy script) ---
    mo_coeff = mf.mo_coeff
    hcore_ao = mf.get_hcore()
    h1_mo_full = np.einsum("pi,pq,qj->ij", mo_coeff, hcore_ao, mo_coeff, optimize=True)

    eri_ao_s8 = mf._eri
    eri_ao_s1 = ao2mo.restore(1, eri_ao_s8, mol.nao_nr())
    h2_mo_full = ao2mo.general(eri_ao_s1, (mo_coeff,) * 4, compact=False)

    h2_mo_full = h2_mo_full.reshape(n_mo_full, n_mo_full, n_mo_full, n_mo_full)
    h2_mo_full_phys = h2_mo_full.transpose(0, 2, 1, 3)

    # Extract active-space integrals
    idx = np.array(active_indices)
    h1_active = h1_mo_full[np.ix_(idx, idx)]
    h2_active_phys = h2_mo_full_phys[np.ix_(idx, idx, idx, idx)]

    # Calculate the core energy contribution
    core_indices = [i for i in range(n_occupied_full) if i not in active_indices]
    core_energy = 2 * np.sum(h1_mo_full[core_indices, core_indices])
    for i in core_indices:
        for j in core_indices:
            core_energy += 2 * h2_mo_full_phys[i, i, j, j] - h2_mo_full_phys[i, j, j, i]

    return h1_active, h2_active_phys, core_energy


def generate_dbt_hamiltonian():
    """
    Generates and saves the 16-qubit Hamiltonian and active-space integrals
    for DBT. This version uses a manual mapping from spatial to spin-orbital
    integrals to ensure the correct 16-qubit operator is generated.
    """
    # --- Configuration ---
    xyz_filepath = "data/dbt/dibenzothiophene.xyz"
    charge = 0
    active_electrons = 8
    active_orbitals = 8
    basis = "sto-3g"
    num_qubits = 2 * active_orbitals

    print(
        "--- Generating 16-Qubit Hamiltonian and Integrals for DBT (High-Performance Mode) ---"
    )
    print(
        f"Using '{basis}' basis. Active Electrons: {active_electrons}, Active Orbitals: {active_orbitals}"
    )

    # --- Step 1: Perform the expensive SCF calculation using direct PySCF ---
    print("\nStarting the high-performance Hartree-Fock calculation with PySCF...")
    start_time = time.time()

    mol = gto.Mole()
    with open(xyz_filepath, "r") as f:
        xyz_lines = f.readlines()
        mol.atom = "".join(xyz_lines[2:])
    mol.basis = basis
    mol.charge = charge
    mol.build()

    mf = scf.RHF(mol).run()

    end_time = time.time()
    print(f"PySCF calculation completed in {end_time - start_time:.2f} seconds.")
    print(f"Total RHF Energy: {mf.e_tot:.8f} Hartrees")

    # --- Step 2: Get active space integrals (spatial) ---
    one_body_spatial, two_body_spatial, core_energy = get_active_space_integrals(
        mol, mf, active_orbitals, active_electrons
    )

    # --- Step 3: Manually map spatial integrals to spin-orbital integrals ---
    n_spatial = active_orbitals
    h1_spin = np.zeros((num_qubits, num_qubits))
    h2_spin = np.zeros((num_qubits, num_qubits, num_qubits, num_qubits))

    for p in range(n_spatial):
        for q in range(n_spatial):
            h1_spin[2 * p, 2 * q] = one_body_spatial[p, q]
            h1_spin[2 * p + 1, 2 * q + 1] = one_body_spatial[p, q]
            for r in range(n_spatial):
                for s in range(n_spatial):
                    val = two_body_spatial[p, r, s, q]
                    h2_spin[2 * p, 2 * r, 2 * s, 2 * q] = val
                    h2_spin[2 * p + 1, 2 * r + 1, 2 * s + 1, 2 * q + 1] = val
                    h2_spin[2 * p, 2 * r + 1, 2 * s + 1, 2 * q] = val
                    h2_spin[2 * p + 1, 2 * r, 2 * s, 2 * q + 1] = val

    # --- Step 4: Build the qubit Hamiltonian from spin-orbital integrals ---
    interaction_op = InteractionOperator(
        constant=0.0, one_body_tensor=h1_spin, two_body_tensor=h2_spin
    )
    fermion_hamiltonian = get_fermion_operator(interaction_op)
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)

    total_offset = mol.energy_nuc() + core_energy
    qubit_hamiltonian += QubitOperator((), total_offset)

    # --- Verification Step ---
    max_qubit_idx = -1
    if qubit_hamiltonian.terms:
        for term in qubit_hamiltonian.terms.keys():
            if term:
                max_qubit_idx = max(max_qubit_idx, *[op[0] for op in term])

    print(
        f"\nVerification: The generated Hamiltonian acts on {max_qubit_idx + 1} qubits (max index found: {max_qubit_idx})."
    )
    assert (
        max_qubit_idx + 1 == num_qubits
    ), "Error: Generated qubit count does not match target."

    print(f"Successfully generated Hamiltonian for {num_qubits} qubits.")
    print(f"Hamiltonian offset (nuclear repulsion + core): {total_offset:.8f} Ha")

    # --- Step 5: Save Artifacts ---
    output_dir = "data/dbt"
    os.makedirs(output_dir, exist_ok=True)

    integrals_path = os.path.join(output_dir, "dbt_integrals.npz")
    np.savez(
        integrals_path,
        one_body_integrals=one_body_spatial,
        two_body_integrals=two_body_spatial,
    )
    print(f"\nSaved active space integrals to {integrals_path}")

    hamiltonian_path = os.path.join(output_dir, "qubit_hamiltonian.json")
    hamiltonian_dict = {}
    for term, coeff in qubit_hamiltonian.terms.items():
        if not term:
            key = "I"
        else:
            key = " ".join([f"{pauli}{qubit}" for qubit, pauli in term])
        hamiltonian_dict[key] = coeff.real

    with open(hamiltonian_path, "w") as f:
        json.dump(hamiltonian_dict, f, indent=4)
    print(f"Saved 16-qubit Hamiltonian to {hamiltonian_path}")

    print("\n--- Artifact Generation Successful ---")


if __name__ == "__main__":
    generate_dbt_hamiltonian()
