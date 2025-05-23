from pyscf import gto, scf, ao2mo
import numpy as np

# OpenFermion imports
from openfermion import InteractionOperator, FermionOperator, jordan_wigner
from openfermion.transforms import get_fermion_operator

# PennyLane imports
import pennylane as qml
from pennylane import numpy as pnp # Use PennyLane's wrapped numpy for parameters
from pennylane import qchem 
import inspect # For inspecting function signatures

def run_rhf_calculation_and_get_integrals(xyz_filepath, basis_set='sto-3g',
                                        n_active_orbitals_target=8, n_active_electrons_target=8):
    try:
        mol = gto.Mole()
        mol.atom = xyz_filepath
        mol.basis = basis_set
        mol.build()

        print(f"Molecule: {xyz_filepath}")
        print(f"Basis set: {basis_set}")
        print(f"Number of basis functions (MOs): {mol.nao_nr()}")
        
        mf = scf.RHF(mol)
        mf.kernel()

        print(f"\nSCF converged: {mf.converged}")
        print(f"Total RHF Energy: {mf.e_tot:.8f} Hartrees")

        mo_coeff = mf.mo_coeff 
        hcore_ao = mf.get_hcore() 
        h1_mo_full = np.einsum('pi,pq,qj->ij', mo_coeff, hcore_ao, mo_coeff, optimize=True)

        eri_ao_chemists = mf._eri 
        if eri_ao_chemists is None: 
            eri_ao_chemists = mol.intor('int2e', aosym='s8')

        eri_ao_s1 = ao2mo.restore(1, eri_ao_chemists, mol.nao_nr()) 
        h2_mo_chemists_full = ao2mo.general(eri_ao_s1, (mo_coeff,)*4, compact=False) 
        h2_mo_full_physicists = h2_mo_chemists_full.transpose(0, 2, 1, 3) 
        
        h1_active, h2_active_phys, core_energy_offset, active_idx = get_active_space_integrals(
            mf, h1_mo_full, h2_mo_full_physicists,
            n_active_orbitals_target, n_active_electrons_target
        )

        if h1_active is None: 
            return mf, None, None, None, None, None, None, None

        return mf, h1_mo_full, h2_mo_full_physicists, \
               h1_active, h2_active_phys, core_energy_offset, active_idx, n_active_electrons_target

    except Exception as e:
        print(f"An error occurred in RHF/Integrals: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None, None, None

def get_active_space_integrals(mf, h1_mo_full, h2_mo_full_phys, n_active_orbitals, n_active_electrons):
    n_mo_full = mf.mo_coeff.shape[1]
    n_electrons_full = mf.mol.nelectron
    n_occupied_full = n_electrons_full // 2

    if n_active_electrons % 2 != 0:
        print("Error: Number of active electrons must be even for RHF/CAS.")
        return None, None, None, None

    n_occupied_active = n_active_electrons // 2

    if n_active_orbitals < n_occupied_active:
        print("Error: Number of active orbitals cannot be less than number of occupied active orbitals.")
        return None, None, None, None

    homo_idx = n_occupied_full - 1
    lumo_idx = n_occupied_full
    n_virtual_active = n_active_orbitals - n_occupied_active
        
    occupied_indices_for_as = list(range(homo_idx - n_occupied_active + 1, homo_idx + 1))
    virtual_indices_for_as = list(range(lumo_idx, lumo_idx + n_virtual_active))

    if not occupied_indices_for_as or occupied_indices_for_as[0] < 0:
        print("Error: Not enough occupied orbitals for the requested active space.")
        return None, None, None, None
    if not virtual_indices_for_as and n_virtual_active > 0:
        if not virtual_indices_for_as or (virtual_indices_for_as and virtual_indices_for_as[-1] >= n_mo_full) :
             print("Error: Not enough virtual orbitals for the requested active space.")
             return None, None, None, None
            
    active_space_indices = sorted(occupied_indices_for_as + virtual_indices_for_as)
    
    if len(active_space_indices) != n_active_orbitals:
            print(f"Error: Final active space selection resulted in {len(active_space_indices)} orbitals, expected {n_active_orbitals}.")
            return None, None, None, None

    print(f"\n--- Active Space Selection ---")
    print(f"Full MOs: {n_mo_full}, Full Occupied MOs: {n_occupied_full}")
    print(f"Requested active orbitals: {n_active_orbitals}, Requested active electrons: {n_active_electrons}")
    print(f"Selected active space MO indices: {active_space_indices}")
    print(f"Number of MOs in active space: {len(active_space_indices)}")
    print(f"HOMO index (0-based): {homo_idx}, LUMO index (0-based): {lumo_idx}")
    print(f"MO energies (Hartree) for active space: {mf.mo_energy[active_space_indices]}")

    core_indices = [i for i in range(n_occupied_full) if i not in active_space_indices]
    e_offset_for_interaction_op = mf.energy_nuc() 
    
    if core_indices:
        print(f"Core MO indices: {core_indices}")
    else:
        print(f"No core orbitals.")
    print(f"Nuclear repulsion energy (used as E_offset for InteractionOperator): {e_offset_for_interaction_op:.8f} Hartrees")

    idx = np.array(active_space_indices)
    h1_active = h1_mo_full[np.ix_(idx, idx)]
    h2_active_phys = h2_mo_full_phys[np.ix_(idx, idx, idx, idx)]
    
    print(f"Shape of active one-electron integrals: {h1_active.shape}")
    print(f"Shape of active two-electron integrals (physicist's): {h2_active_phys.shape}")

    return h1_active, h2_active_phys, e_offset_for_interaction_op, active_space_indices

def run_rhf_get_integrals_qubit_hamiltonian(xyz_filepath, basis_set='sto-3g',
                                            n_active_orbitals_target=8, n_active_electrons_target=8):
    try:
        mf, h1_mo_full, h2_mo_full_phys, \
        h1_act, h2_act_phys, E_offset_nuc_rep, act_indices, n_elec_act_from_prev = \
            run_rhf_calculation_and_get_integrals(
                xyz_filepath, basis_set,
                n_active_orbitals_target, n_active_electrons_target
            )

        if mf is None or h1_act is None: 
            print("Aborting Hamiltonian construction due to RHF/active space selection error.")
            return None, None, None, None, None, None, None, None

        print(f"\n--- Hamiltonian Construction (Active Space: {n_active_orbitals_target} orbitals) ---")
        num_active_mos_spatial = h1_act.shape[0]
        
        interaction_op = InteractionOperator(
            constant=E_offset_nuc_rep, 
            one_body_tensor=h1_act,
            two_body_tensor=h2_act_phys
        )
        fermion_hamiltonian = get_fermion_operator(interaction_op)
        print(f"Number of terms in Fermionic Hamiltonian: {len(fermion_hamiltonian.terms)}")
        
        qubit_hamiltonian_openfermion = jordan_wigner(fermion_hamiltonian)
        n_spin_orbitals = 2 * num_active_mos_spatial
        print(f"Number of qubits for JW mapping: {n_spin_orbitals}")
        print(f"Number of terms in OpenFermion Qubit Hamiltonian (Jordan-Wigner): {len(qubit_hamiltonian_openfermion.terms)}")

        return mf, qubit_hamiltonian_openfermion, E_offset_nuc_rep, act_indices, \
               n_elec_act_from_prev, num_active_mos_spatial, h1_mo_full, h2_mo_full_phys

    except Exception as e:
        print(f"An error occurred during Hamiltonian generation: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None, None, None

def run_vqe_pennylane(openfermion_qubit_op, num_active_electrons, num_active_spatial_orbitals, max_iterations=200, verbose=False):
    print(f"\n--- Starting VQE Calculation (PennyLane runtime version: v{qml.__version__}) ---")
    num_qubits = 2 * num_active_spatial_orbitals

    try:
        H_pennylane_imported = qml.import_operator(openfermion_qubit_op, format='openfermion')
        if isinstance(H_pennylane_imported, qml.Hamiltonian):
            is_complex_coeff = np.any([np.iscomplex(c) for c in H_pennylane_imported.coeffs])
            is_complex_offset = H_pennylane_imported.offset is not None and np.iscomplex(H_pennylane_imported.offset)
            if is_complex_coeff or is_complex_offset:
                print("Warning: PennyLane Hamiltonian from import_operator had complex parts. Taking real components.")
                new_coeffs_import = [c.real for c in H_pennylane_imported.coeffs]
                new_offset_import = H_pennylane_imported.offset.real if H_pennylane_imported.offset is not None else 0.0
                H_pennylane = qml.Hamiltonian(new_coeffs_import, H_pennylane_imported.ops, offset=new_offset_import)
            else:
                H_pennylane = H_pennylane_imported
        else: 
            H_pennylane = H_pennylane_imported
    except Exception as e_import:
        print(f"Error importing OpenFermion operator to PennyLane: {e_import}. Using manual fallback.")
        coeffs_list = []
        obs_list = []
        offset_val = 0.0 
        if not openfermion_qubit_op.terms:
             H_pennylane = qml.Hamiltonian([0.0],[qml.Identity(0)]) if num_qubits > 0 else qml.Hamiltonian([],[])
        else:
            for term, val in openfermion_qubit_op.terms.items():
                if not term: 
                    offset_val += val.real 
                    continue
                pauli_word = qml.Identity(0)
                first_op = True
                for qubit_idx, pauli_char in term:
                    op_map = {'X': qml.PauliX, 'Y': qml.PauliY, 'Z': qml.PauliZ}
                    if pauli_char not in op_map: continue
                    op = op_map[pauli_char](qubit_idx)
                    if first_op:
                        pauli_word = op
                        first_op = False
                    else:
                        pauli_word = pauli_word @ op
                if not first_op: 
                    coeffs_list.append(val.real)
                    obs_list.append(pauli_word)
            if not obs_list and offset_val != 0.0 :
                 H_pennylane = qml.Hamiltonian([0.0],[qml.Identity(0)], offset=offset_val) if num_qubits > 0 else qml.Hamiltonian([],[], offset=offset_val)
            elif not obs_list and offset_val == 0.0:
                 H_pennylane = qml.Hamiltonian([0.0],[qml.Identity(0)]) if num_qubits > 0 else qml.Hamiltonian([],[])
            else: 
                 H_pennylane = qml.Hamiltonian(coeffs_list, obs_list, offset=offset_val)

    if verbose:
        print(f"PennyLane Hamiltonian num_wires: {H_pennylane.wires.tolist() if H_pennylane.wires else 'N/A'}")
        if hasattr(H_pennylane, 'terms'):
            _coeffs, _ops = H_pennylane.terms()
            print(f"Number of terms in PennyLane Hamiltonian (excluding offset): {len(_ops)}")
            if isinstance(H_pennylane, qml.Hamiltonian): print(f"Offset of PennyLane Hamiltonian: {H_pennylane.offset}")
        else: print(f"PennyLane Hamiltonian object type: {type(H_pennylane)}")

    dev = qml.device("lightning.qubit", wires=num_qubits)
    hf_state_config = np.zeros(num_qubits, dtype=int)
    hf_state_config[:num_active_electrons] = 1
    if verbose: print(f"Hartree-Fock state configuration (for init_state): {hf_state_config}")

    if verbose: print(f"Generating excitations for {num_active_electrons} electrons in {num_qubits} spin-orbitals (using qml.qchem.excitations).")
    raw_s_excitations, raw_d_excitations = qchem.excitations(electrons=num_active_electrons, orbitals=num_qubits, delta_sz=0)

    if verbose:
        print(f"Generated {len(raw_s_excitations)} raw single excitations.")
        if raw_s_excitations: print(f"  Format of first few raw single excitations: {raw_s_excitations[:3]}")
        print(f"Generated {len(raw_d_excitations)} raw double excitations.")
        if raw_d_excitations: print(f"  Format of first few raw double excitations: {raw_d_excitations[:3]}")
    
    s_wires, d_wires = qchem.excitations_to_wires(raw_s_excitations, raw_d_excitations)
    if verbose:
        print(f"Processed {len(s_wires)} single excitation wire-sets using excitations_to_wires.")
        if s_wires: print(f"  Format of first few processed s_wires: {s_wires[:3]}")
        print(f"Processed {len(d_wires)} double excitation wire-sets using excitations_to_wires.")
        if d_wires: print(f"  Format of first few processed d_wires: {d_wires[:3]}")

    if verbose:
        print("\nInspecting the qml.UCCSD template that will be called:")
        print(f"  qml.UCCSD module: {qml.UCCSD.__module__}")
        print(f"  qml.UCCSD qualified name: {qml.UCCSD.__qualname__}")
        try:
            print(f"  File path for qml.UCCSD: {inspect.getfile(qml.UCCSD)}")
            print(f"  Signature of qml.UCCSD.__init__: {inspect.signature(qml.UCCSD.__init__)}")
        except Exception as e_inspect: print(f"  Could not get file/signature for qml.UCCSD: {e_inspect}")

    num_uccsd_params = len(raw_s_excitations) + len(raw_d_excitations)
    if verbose: print(f"Total number of UCCSD parameters calculated (based on raw excitations): {num_uccsd_params}")

    @qml.qnode(dev)
    def cost_function(weights):
        qml.UCCSD(weights, wires=range(num_qubits), s_wires=s_wires, d_wires=d_wires, init_state=hf_state_config)
        return qml.expval(H_pennylane)

    opt = qml.AdamOptimizer(stepsize=0.1)
    params = pnp.random.normal(0, np.pi / 4, size=num_uccsd_params, requires_grad=True)

    print(f"\nUsing {num_uccsd_params} UCCSD parameters for optimization.")
    print("Attempting to run VQE optimization with PennyLane...")
    
    energies_over_iterations = [] # List to store energies
    
    # Calculate and store initial energy
    initial_energy = cost_function(params)
    energies_over_iterations.append(initial_energy)
    print(f"Iter {-1:3d}:  E = {initial_energy:.8f} Ha (Initial)") # -1 for initial state before any step

    vqe_energy = initial_energy 

    for it in range(max_iterations):
        params, prev_E = opt.step_and_cost(cost_function, params) 
        # prev_E is the energy *before* the step of iteration `it`
        # which is the energy of the state at the beginning of this iteration
        energies_over_iterations.append(prev_E) # Store energy for this iteration

        if (it + 1) % 20 == 0 or it == 0: # Iter 0 is the first step
            print(f"Iter {it:3d}:  E = {prev_E:.8f} Ha") 
        
        if it == max_iterations - 1: # After the last step
            vqe_energy = cost_function(params) # Get the final energy
            energies_over_iterations.append(vqe_energy) # Append the very final energy
            print(f"Iter {it:3d}:  E = {vqe_energy:.8f} Ha (After final step)")


    if max_iterations == 0: 
        vqe_energy = initial_energy # Already stored
    # The final print of vqe_energy is already handled by the loop's last iteration or initial state

    # Save energies to a file
    np.savetxt("vqe_energies.dat", np.array(energies_over_iterations), header="Iteration Energy(Ha)")
    print("\nSaved VQE energies to vqe_energies.dat")

    print(f"Final VQE Energy (raw value from PennyLane for H_qubit): {vqe_energy:.8f} Hartrees")
    return vqe_energy

if __name__ == "__main__":
    print(f"--- Script starting with PennyLane version: {qml.__version__} ---")
    xyz_file = "dibenzothiophene.xyz"
    num_active_orbitals_config = 8 
    num_active_electrons_config = 8 
    vqe_iterations = 500

    print("--- Starting Full Pipeline Step with PennyLane VQE ---")
    
    mf_obj, openfermion_h_qubit, E_nuc_rep_interaction_op_const, \
    active_idx, n_elec_active, n_orb_active_spatial, \
    h1_mo_full, h2_mo_full_phys = \
        run_rhf_get_integrals_qubit_hamiltonian(
            xyz_file, basis_set='sto-3g',
            n_active_orbitals_target=num_active_orbitals_config,
            n_active_electrons_target=num_active_electrons_config
        )

    if mf_obj and openfermion_h_qubit and h1_mo_full is not None and h2_mo_full_phys is not None:
        print("\n--- Hamiltonian Generation Successful ---")
        print(f"Target active space: {n_elec_active} electrons in {n_orb_active_spatial} spatial orbitals.")
        print(f"Number of spin-orbitals / qubits: {2 * n_orb_active_spatial}")
        print(f"Nuclear Repulsion Energy (used as constant in InteractionOperator): {E_nuc_rep_interaction_op_const:.8f} Hartrees")
        
        qubit_ham_const_from_of = 0.0
        for term_tuple, coeff_val in openfermion_h_qubit.terms.items():
            if not term_tuple: 
                qubit_ham_const_from_of = coeff_val.real 
                break
        print(f"Constant term in OpenFermion Qubit Hamiltonian (offset of H_qubit): {qubit_ham_const_from_of:.8f}")

        vqe_computed_energy_pl = run_vqe_pennylane(
            openfermion_h_qubit, 
            n_elec_active, 
            n_orb_active_spatial,
            max_iterations=vqe_iterations,
            verbose=False 
        )

        if vqe_computed_energy_pl is not None:
            print("\n--- PennyLane VQE Calculation Complete ---")
            
            core_indices = [i for i in range(mf_obj.mol.nelectron // 2) if i not in active_idx]
            E_core_elec_hf = 0.0
            if core_indices:
                # E_core_elec_HF = 2 * sum_{i in core} h_ii + sum_{i,j in core} (2*J_ij - K_ij)
                for i in core_indices:
                    E_core_elec_hf += 2 * h1_mo_full[i, i]
                for i in core_indices:
                    for j in core_indices:
                        J_ij = h2_mo_full_phys[i, i, j, j] # <ii|jj>
                        K_ij = h2_mo_full_phys[i, j, j, i] # <ij|ji>
                        E_core_elec_hf += (2 * J_ij - K_ij)
            
            print(f"\n--- Energy Summary (PennyLane VQE) ---")
            print(f"RHF Total Energy (for comparison): {mf_obj.e_tot:.8f} Hartrees")
            print(f"VQE Computed Value (eigenvalue of active space H_qubit): {vqe_computed_energy_pl:.8f} Hartrees")
            print(f"  Note: H_qubit's constant includes E_nuc_rep ({E_nuc_rep_interaction_op_const:.8f} Ha) from InteractionOperator.")
            
            print(f"Hartree-Fock Electronic Energy of Core Orbitals (E_core_elec_hf): {E_core_elec_hf:.8f} Hartrees")
            
            # Total Energy = vqe_computed_energy_pl (which is E_nuc + E_active_VQE) + E_core_elec_hf
            total_energy_from_vqe_pl = vqe_computed_energy_pl + E_core_elec_hf

            print(f"Approx. Total Molecular Energy from PennyLane VQE: {total_energy_from_vqe_pl:.8f} Hartrees")
            print(f"  (Calculated as: VQE_H_qubit_eigenvalue + E_core_elec_hf)")
            print(f"  (VQE_H_qubit_eigenvalue = E_nuc_rep + E_active_VQE_electronic)")
    else:
        print("\n--- Full Pipeline Step Failed (Hamiltonian or Integral Generation Error) ---")
