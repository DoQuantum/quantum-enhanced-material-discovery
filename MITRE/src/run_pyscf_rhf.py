from pyscf import gto, scf, ao2mo, mcscf
import numpy as np

# OpenFermion imports
from openfermion import InteractionOperator, FermionOperator, jordan_wigner
from openfermion.transforms import get_fermion_operator

# PennyLane imports
import pennylane as qml
from pennylane import numpy as pnp # Use PennyLane's wrapped numpy for parameters
from pennylane import qchem 
import inspect # For inspecting function signatures
import datetime # Added for timestamp
import matplotlib.pyplot as plt # Added for plotting directly
import sys # Added for stdout redirection
import traceback # Ensure traceback is imported
import os # Added for directory creation

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
        h1_mo_full = np.einsum('pi,pq,qj->ij', mo_coeff, hcore_ao, mo_coeff, optimize='optimal')

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
        # import traceback # No need to re-import if imported at the top
        traceback.print_exc(file=sys.stdout) # Direct traceback to the current stdout (log file)
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

        # print(f"DEBUG: E_offset_nuc_rep passed to InteractionOperator: {E_offset_nuc_rep:.8f}")
        # print(f"DEBUG: interaction_op.constant = {interaction_op.constant:.8f}")
        
        fh_const_to_check = 0.0
        if hasattr(fermion_hamiltonian, 'constant'): 
             fh_const_to_check = fermion_hamiltonian.constant
             if isinstance(fh_const_to_check, complex):
                 fh_const_to_check = fh_const_to_check.real
            #  print(f"DEBUG: fermion_hamiltonian.constant attribute = {fh_const_to_check:.8f}")
        elif isinstance(fermion_hamiltonian, FermionOperator) and () in fermion_hamiltonian.terms: 
             fh_const_to_check = fermion_hamiltonian.terms[()].real 
            #  print(f"DEBUG: fermion_hamiltonian constant term value [()] = {fh_const_to_check:.8f}")
        else:
             if isinstance(fermion_hamiltonian, (int, float, complex)):
                 fh_const_to_check = fermion_hamiltonian.real if isinstance(fermion_hamiltonian, complex) else float(fermion_hamiltonian)
                #  print(f"DEBUG: fermion_hamiltonian is a scalar constant = {fh_const_to_check:.8f}")
            #  else:
                # print(f"DEBUG: Could not determine fermion_hamiltonian constant directly. Type: {type(fermion_hamiltonian)}")


        qh_const_from_terms = 0.0
        if () in qubit_hamiltonian_openfermion.terms: 
            qh_const_from_terms = qubit_hamiltonian_openfermion.terms[()].real
        # print(f"DEBUG: qubit_hamiltonian_openfermion constant term value [()] = {qh_const_from_terms:.8f}")

        if isinstance(fh_const_to_check, float):
            # if not np.isclose(interaction_op.constant, fh_const_to_check):
                # print("WARNING: InteractionOperator constant does NOT match FermionOperator constant!")
            if not np.isclose(fh_const_to_check, qh_const_from_terms):
                print("INFO: FermionOperator constant does not match QubitOperator constant (known OpenFermion JW behavior).")
        # else:
            # print("WARNING: Could not reliably compare constants as FermionOperator constant was not extracted as float.")


        return mf, qubit_hamiltonian_openfermion, E_offset_nuc_rep, act_indices, \
               n_elec_act_from_prev, num_active_mos_spatial, h1_mo_full, h2_mo_full_phys

    except Exception as e:
        print(f"An error occurred during Hamiltonian generation: {e}")
        # import traceback # No need to re-import
        traceback.print_exc(file=sys.stdout) # Direct traceback to the current stdout (log file)
        return None, None, None, None, None, None, None, None

def run_vqe_pennylane(openfermion_qubit_op, num_active_electrons, num_active_spatial_orbitals, basis_set_name, max_iterations=200, verbose=False): # Added basis_set_name
    print(f"\n--- Starting VQE Calculation (PennyLane runtime version: v{qml.__version__}) ---")
    num_qubits = 2 * num_active_spatial_orbitals

    # Generate timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_filename = f"dbt_{basis_set_name}_8e8o_{timestamp}"
    energy_data_filename = f"logs/{base_filename}_energies.dat"
    plot_filename = f"logs/{base_filename}_convergence.png"

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

    @qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
    def cost_function(weights):
        qml.UCCSD(weights, wires=range(num_qubits), s_wires=s_wires, d_wires=d_wires, init_state=hf_state_config)
        return qml.expval(H_pennylane)

    opt = qml.AdamOptimizer(stepsize=0.01, beta1=0.9, beta2=0.999)
    params = pnp.random.normal(0, 0.1, size=num_uccsd_params, requires_grad=True)
    
    print(f"\nUsing {num_uccsd_params} UCCSD parameters for optimization.")
    print("Attempting to run VQE optimization with PennyLane...")
    
    energies_for_plot = []
    current_energy = cost_function(params)
    energies_for_plot.append(current_energy)
    best_energy_so_far = current_energy
    best_params = pnp.copy(params)
    
    print(f"Iter {-1:3d}:  E = {current_energy:.8f} Ha (Initial) | Best E: {best_energy_so_far:.8f} Ha")

    # convergence parameters
    conv_window = 5
    conv_threshold = 1e-6
    no_improvement_count = 0
    
    for it in range(max_iterations):
        gradients = qml.grad(cost_function)(params)
        params = opt.apply_grad(gradients, params) # update params
        
        current_energy = cost_function(params)
        energies_for_plot.append(current_energy)
        
        # update best energy and parameters only if improved
        if current_energy < best_energy_so_far:
            best_energy_so_far = current_energy
            best_params = pnp.copy(params)
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # optimize step size if no improvement for 5 iterations
        if no_improvement_count >= 5:
            opt.stepsize *= 0.95
            no_improvement_count = 0    
        
        if (it + 1) % 20 == 0 or it == 0:
            print(f"Iter {it:3d}:  E = {current_energy:.8f} Ha | Best E: {best_energy_so_far:.8f} Ha | Step size: {opt.stepsize:.6f}")
        
        # check convergence and if recent energies are stable
        if len(energies_for_plot) >= conv_window:
            recent_energies = energies_for_plot[-conv_window:]
            if np.std(recent_energies) < conv_threshold and it > conv_window:
                print(f"\nConverged at iteration {it} with energy std dev < {conv_threshold}")
                break
    
    # restore best parameters from every iteration
    params = best_params

    np.savetxt(energy_data_filename, np.array(energies_for_plot), header="Iteration Energy(Ha)") # Use unique filename
    print(f"\nSaved VQE energies (optimizer path) to {energy_data_filename}")

    # Plotting logic integrated here
    num_energy_points_plot = len(energies_for_plot)
    if num_energy_points_plot > 0:
        if num_energy_points_plot == 1: # Only initial energy
            iterations_plot = np.array([-1])
        else:
            iterations_plot = np.array([-1] + list(range(num_energy_points_plot - 2)) + [num_energy_points_plot - 2])
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations_plot, energies_for_plot, marker='o', linestyle='-')
        plt.xlabel("Optimization Iteration (Adam Step)")
        plt.ylabel("Energy (Hartree)")
        plt.title(f"VQE Energy Convergence for DBT ({basis_set_name}, {num_active_electrons}e, {num_active_spatial_orbitals}o)")
        plt.grid(True)
        plt.xticks(np.linspace(min(iterations_plot), max(iterations_plot), num=min(20, len(iterations_plot)), dtype=int))
        plt.tight_layout()
        plt.savefig(plot_filename) # Use unique filename
        print(f"Saved convergence plot to {plot_filename}")
        # plt.show() # Optionally show plot, but for batch runs, saving is often preferred.
        plt.close() # Close the plot figure to free memory
    else:
        print("No energy data to plot.")


    print(f"Final VQE Energy (raw value from PennyLane, best found): {best_energy_so_far:.8f} Hartrees")
    return best_energy_so_far

if __name__ == "__main__":
    # === Configuration for the current run (MUST be defined before log setup) ===
    current_basis_set = 'sto-3g' 
    # To run with STO-3G again, change above to 'sto-3g'
    # To run with 6-31G*, change above to '6-31g*' (or '6-31g(d)')

    xyz_file = "dibenzothiophene.xyz"
    num_active_orbitals_config = 8 
    num_active_electrons_config = 8 
    vqe_iterations = 300 # You can adjust this

    # --- Setup Log File ---
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    timestamp_log = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = f"logs/dbt_{current_basis_set}_8e8o_{timestamp_log}_run.log"
    original_stdout = sys.stdout  # Save a reference to the original standard output

    # Ensure the log file is opened with 'utf-8' encoding for broader character support
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        sys.stdout = log_file  # Redirect stdout to log_file
        try:
            # Try a very simple initial print and flush
            log_file.write(f"Log file opened: {log_filename} at {timestamp_log}\n")
            log_file.flush()

            print(f"--- Script starting with PennyLane version: {qml.__version__} ---")
            print(f"Run configured for basis set: {current_basis_set}")
            print(f"Active space: {num_active_electrons_config} electrons in {num_active_orbitals_config} orbitals.")
            print(f"VQE iterations: {vqe_iterations}")
            print(f"Output is being logged to this file: {log_filename}")
            print(f"Timestamp: {timestamp_log}\n")
            
            print(f"--- Starting Full Pipeline Step with PennyLane VQE using basis: {current_basis_set} ---")
            
            mf_obj, openfermion_h_qubit, E_nuc_rep_interaction_op_const, \
            active_idx, n_elec_active, n_orb_active_spatial, \
            h1_mo_full, h2_mo_full_phys = \
                run_rhf_get_integrals_qubit_hamiltonian(
                    xyz_file, basis_set=current_basis_set,
                    n_active_orbitals_target=num_active_orbitals_config,
                    n_active_electrons_target=num_active_electrons_config
                )

            if mf_obj and openfermion_h_qubit and h1_mo_full is not None and h2_mo_full_phys is not None:
                print("\n--- Hamiltonian Generation Successful ---")
                print(f"Target active space: {n_elec_active} electrons in {n_orb_active_spatial} spatial orbitals.")
                print(f"Number of spin-orbitals / qubits: {2 * n_orb_active_spatial}")
                print(f"Nuclear Repulsion Energy (E_nuc_rep_actual): {E_nuc_rep_interaction_op_const:.8f} Hartrees")
                
                qubit_ham_const_from_of = 0.0
                if () in openfermion_h_qubit.terms:
                    qubit_ham_const_from_of = openfermion_h_qubit.terms[()].real 
                print(f"Constant in OpenFermion Qubit Hamiltonian (C_qubit_jw): {qubit_ham_const_from_of:.8f} Hartrees")

                vqe_computed_energy_pl = run_vqe_pennylane(
                    openfermion_h_qubit, 
                    n_elec_active, 
                    n_orb_active_spatial,
                    current_basis_set, 
                    max_iterations=vqe_iterations,
                    verbose=False 
                )

                if vqe_computed_energy_pl is not None:
                    print("\n--- PennyLane VQE Calculation Complete ---")
                    
                    E_active_VQE_electronic_ops_part = vqe_computed_energy_pl - qubit_ham_const_from_of
                    
                    rhf_total_electronic_energy_pyscf = mf_obj.e_tot - E_nuc_rep_interaction_op_const
                    
                    active_occupied_indices = sorted([i for i in active_idx if i < mf_obj.mol.nelectron // 2])
                    
                    E_rhf_elec_active_occ = 0.0
                    if active_occupied_indices:
                        sum_2h_ii_act_occ = 0.0
                        sum_ee_act_occ = 0.0
                        for i in active_occupied_indices:
                            sum_2h_ii_act_occ += 2 * h1_mo_full[i, i]
                        for i in active_occupied_indices:
                            for j in active_occupied_indices:
                                J_ij_act = h2_mo_full_phys[i, i, j, j]
                                K_ij_act = h2_mo_full_phys[i, j, j, i]
                                sum_ee_act_occ += (2 * J_ij_act - K_ij_act)
                        E_rhf_elec_active_occ = sum_2h_ii_act_occ + sum_ee_act_occ
                    
                    core_hf_electronic_energy = rhf_total_electronic_energy_pyscf - E_rhf_elec_active_occ
                    
                    print(f"\n--- Final VQE Energy Summary ---")
                    print(f"RHF Total Energy (PySCF): {mf_obj.e_tot:.8f} Hartrees")
                    print(f"Nuclear Repulsion Energy (E_nuc_rep_actual): {E_nuc_rep_interaction_op_const:.8f} Hartrees")
                    print(f"Core HF Electronic Energy (E_core_elec_hf): {core_hf_electronic_energy:.8f} Hartrees")
                    print(f"VQE Eigenvalue of Qubit Hamiltonian (E_VQE_raw): {vqe_computed_energy_pl:.8f} Hartrees")
                    print(f"Constant in Qubit Hamiltonian (C_qubit_jw): {qubit_ham_const_from_of:.8f} Hartrees")
                    print(f"Active Space Electronic Ops Part from VQE (E_VQE_raw - C_qubit_jw): {E_active_VQE_electronic_ops_part:.8f} Hartrees")
                    
                    final_total_vqe_energy = E_nuc_rep_interaction_op_const + core_hf_electronic_energy + E_active_VQE_electronic_ops_part
                    print(f"Final Total VQE Molecular Energy (E_nuc_rep + E_core_elec_hf + E_active_ops_part): {final_total_vqe_energy:.8f} Hartrees")

                    correlation_energy_captured_by_vqe = final_total_vqe_energy - mf_obj.e_tot
                    print(f"VQE Correlation Energy (E_VQE_total - E_RHF_total): {correlation_energy_captured_by_vqe:.8f} Hartrees")
                    if correlation_energy_captured_by_vqe > 0:
                        print("INFO: VQE total energy is higher than RHF total energy.")
                    else:
                        print("INFO: VQE total energy is lower than or equal to RHF total energy.")

            else:
                print(f"\n--- Full Pipeline Step Failed for basis {current_basis_set} (Hamiltonian or Integral Generation Error) ---")

            print(f"\n--- Full Pipeline Step Completed for basis {current_basis_set} ---")
            print(f"All output logged to: {log_filename}")

        except Exception as main_execution_error:
            print(f"\nCRITICAL ERROR IN MAIN EXECUTION BLOCK: {main_execution_error}")
            traceback.print_exc(file=sys.stdout) # Log this critical error
        finally:
            if log_file: # Check if log_file is defined (it should be)
                log_file.flush() # Explicitly flush before restoring stdout
            sys.stdout = original_stdout # Reset stdout to its original value
            # The 'with open' statement ensures log_file is closed automatically.

    print(f"Script execution finished. Full log available in: {log_filename}") # This print goes to the terminal
