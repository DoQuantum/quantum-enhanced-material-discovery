import datetime  # Added for timestamp
import inspect  # For inspecting function signatures
import json
import os  # Added for directory creation
import sys  # Added for stdout redirection
import traceback  # Ensure traceback is imported

import matplotlib.pyplot as plt  # Added for plotting directly
import numpy as np
# PennyLane imports
import pennylane as qml
# OpenFermion imports
from openfermion import FermionOperator, InteractionOperator, jordan_wigner
from openfermion.transforms import get_fermion_operator
from pennylane import \
    numpy as pnp  # Use PennyLane's wrapped numpy for parameters
from pennylane import qchem
from pyscf import ao2mo, gto, mcscf, scf


def run_rhf_calculation_and_get_integrals(
    xyz_filepath,
    basis_set="sto-3g",
    n_active_orbitals_target=8,
    n_active_electrons_target=8,
):
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
        h1_mo_full = np.einsum(
            "pi,pq,qj->ij", mo_coeff, hcore_ao, mo_coeff, optimize=True
        )

        eri_ao_chemists = mf._eri
        if eri_ao_chemists is None:
            eri_ao_chemists = mol.intor("int2e", aosym="s8")

        eri_ao_s1 = ao2mo.restore(1, eri_ao_chemists, mol.nao_nr())
        h2_mo_chemists_full = ao2mo.general(eri_ao_s1, (mo_coeff,) * 4, compact=False)
        h2_mo_full_physicists = h2_mo_chemists_full.transpose(0, 2, 1, 3)

        h1_active, h2_active_phys, core_energy_offset, active_idx = (
            get_active_space_integrals(
                mf,
                h1_mo_full,
                h2_mo_full_physicists,
                n_active_orbitals_target,
                n_active_electrons_target,
            )
        )

        if h1_active is None:
            return mf, None, None, None, None, None, None, None

        return (
            mf,
            h1_mo_full,
            h2_mo_full_physicists,
            h1_active,
            h2_active_phys,
            core_energy_offset,
            active_idx,
            n_active_electrons_target,
        )

    except Exception as e:
        print(f"An error occurred in RHF/Integrals: {e}")
        # import traceback # No need to re-import if imported at the top
        traceback.print_exc(
            file=sys.stdout
        )  # Direct traceback to the current stdout (log file)
        return None, None, None, None, None, None, None, None


def get_active_space_integrals(
    mf, h1_mo_full, h2_mo_full_phys, n_active_orbitals, n_active_electrons
):
    n_mo_full = mf.mo_coeff.shape[1]
    n_electrons_full = mf.mol.nelectron
    n_occupied_full = n_electrons_full // 2

    if n_active_electrons % 2 != 0:
        print("Error: Number of active electrons must be even for RHF/CAS.")
        return None, None, None, None

    n_occupied_active = n_active_electrons // 2

    if n_active_orbitals < n_occupied_active:
        print(
            "Error: Number of active orbitals cannot be less than number of occupied active orbitals."
        )
        return None, None, None, None

    homo_idx = n_occupied_full - 1
    lumo_idx = n_occupied_full
    n_virtual_active = n_active_orbitals - n_occupied_active

    occupied_indices_for_as = list(
        range(homo_idx - n_occupied_active + 1, homo_idx + 1)
    )
    virtual_indices_for_as = list(range(lumo_idx, lumo_idx + n_virtual_active))

    if not occupied_indices_for_as or occupied_indices_for_as[0] < 0:
        print("Error: Not enough occupied orbitals for the requested active space.")
        return None, None, None, None
    if not virtual_indices_for_as and n_virtual_active > 0:
        if not virtual_indices_for_as or (
            virtual_indices_for_as and virtual_indices_for_as[-1] >= n_mo_full
        ):
            print("Error: Not enough virtual orbitals for the requested active space.")
            return None, None, None, None

    active_space_indices = sorted(occupied_indices_for_as + virtual_indices_for_as)

    if len(active_space_indices) != n_active_orbitals:
        print(
            f"Error: Final active space selection resulted in {len(active_space_indices)} orbitals, expected {n_active_orbitals}."
        )
        return None, None, None, None

    print(f"\n--- Active Space Selection ---")
    print(f"Full MOs: {n_mo_full}, Full Occupied MOs: {n_occupied_full}")
    print(
        f"Requested active orbitals: {n_active_orbitals}, Requested active electrons: {n_active_electrons}"
    )
    print(f"Selected active space MO indices: {active_space_indices}")
    print(f"Number of MOs in active space: {len(active_space_indices)}")
    print(f"HOMO index (0-based): {homo_idx}, LUMO index (0-based): {lumo_idx}")
    print(
        f"MO energies (Hartree) for active space: {mf.mo_energy[active_space_indices]}"
    )

    core_indices = [i for i in range(n_occupied_full) if i not in active_space_indices]
    e_offset_for_interaction_op = mf.energy_nuc()

    if core_indices:
        print(f"Core MO indices: {core_indices}")
    else:
        print(f"No core orbitals.")
    print(
        f"Nuclear repulsion energy (used as E_offset for InteractionOperator): {e_offset_for_interaction_op:.8f} Hartrees"
    )

    idx = np.array(active_space_indices)
    h1_active = h1_mo_full[np.ix_(idx, idx)]
    h2_active_phys = h2_mo_full_phys[np.ix_(idx, idx, idx, idx)]

    print(f"Shape of active one-electron integrals: {h1_active.shape}")
    print(
        f"Shape of active two-electron integrals (physicist's): {h2_active_phys.shape}"
    )

    return h1_active, h2_active_phys, e_offset_for_interaction_op, active_space_indices


def run_rhf_get_integrals_qubit_hamiltonian(
    xyz_filepath,
    basis_set="sto-3g",
    n_active_orbitals_target=8,
    n_active_electrons_target=8,
):
    try:
        (
            mf,
            h1_mo_full,
            h2_mo_full_phys,
            h1_act,
            h2_act_phys,
            E_offset_nuc_rep,
            act_indices,
            n_elec_act_from_prev,
        ) = run_rhf_calculation_and_get_integrals(
            xyz_filepath, basis_set, n_active_orbitals_target, n_active_electrons_target
        )

        if mf is None or h1_act is None:
            print(
                "Aborting Hamiltonian construction due to RHF/active space selection error."
            )
            return None, None, None, None, None, None, None, None

        print(
            f"\n--- Hamiltonian Construction (Active Space: {n_active_orbitals_target} orbitals) ---"
        )
        num_active_mos_spatial = h1_act.shape[0]

        interaction_op = InteractionOperator(
            constant=E_offset_nuc_rep,
            one_body_tensor=h1_act,
            two_body_tensor=h2_act_phys,
        )
        fermion_hamiltonian = get_fermion_operator(interaction_op)
        print(
            f"Number of terms in Fermionic Hamiltonian: {len(fermion_hamiltonian.terms)}"
        )

        qubit_hamiltonian_openfermion = jordan_wigner(fermion_hamiltonian)
        n_spin_orbitals = 2 * num_active_mos_spatial
        print(f"Number of qubits for JW mapping: {n_spin_orbitals}")
        print(
            f"Number of terms in OpenFermion Qubit Hamiltonian (Jordan-Wigner): {len(qubit_hamiltonian_openfermion.terms)}"
        )

        # --- START: ADDED CODE TO SAVE ARTIFACTS ---
        # 1. Save DBT integrals (active space one- and two-body integrals)
        np.savez(
            "dbt_integrals.npz",
            one_body_integrals=h1_act,
            two_body_integrals=h2_act_phys,
        )
        print("Saved active space integrals to dbt_integrals.npz")

        # 2. Save qubit Hamiltonian to JSON
        hamiltonian_dict = {
            str(term): coeff.real
            for term, coeff in qubit_hamiltonian_openfermion.terms.items()
        }
        with open("qubit_hamiltonian.json", "w") as f:
            json.dump(hamiltonian_dict, f, indent=4)
        print("Saved qubit Hamiltonian to qubit_hamiltonian.json")
        # --- END: ADDED CODE TO SAVE ARTIFACTS ---

        # print(f"DEBUG: E_offset_nuc_rep passed to InteractionOperator: {E_offset_nuc_rep:.8f}")
        # print(f"DEBUG: interaction_op.constant = {interaction_op.constant:.8f}")

        fh_const_to_check = 0.0
        if hasattr(fermion_hamiltonian, "constant"):
            fh_const_to_check = fermion_hamiltonian.constant
            if isinstance(fh_const_to_check, complex):
                fh_const_to_check = fh_const_to_check.real
        #  print(f"DEBUG: fermion_hamiltonian.constant attribute = {fh_const_to_check:.8f}")
        elif (
            isinstance(fermion_hamiltonian, FermionOperator)
            and () in fermion_hamiltonian.terms
        ):
            fh_const_to_check = fermion_hamiltonian.terms[()].real
        #  print(f"DEBUG: fermion_hamiltonian constant term value [()] = {fh_const_to_check:.8f}")
        else:
            if isinstance(fermion_hamiltonian, (int, float, complex)):
                fh_const_to_check = (
                    fermion_hamiltonian.real
                    if isinstance(fermion_hamiltonian, complex)
                    else float(fermion_hamiltonian)
                )
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
                print(
                    "INFO: FermionOperator constant does not match QubitOperator constant (known OpenFermion JW behavior)."
                )
        # else:
        # print("WARNING: Could not reliably compare constants as FermionOperator constant was not extracted as float.")

        return (
            mf,
            qubit_hamiltonian_openfermion,
            E_offset_nuc_rep,
            act_indices,
            n_elec_act_from_prev,
            num_active_mos_spatial,
            h1_mo_full,
            h2_mo_full_phys,
        )

    except Exception as e:
        print(f"An error occurred during Hamiltonian generation: {e}")
        # import traceback # No need to re-import
        traceback.print_exc(
            file=sys.stdout
        )  # Direct traceback to the current stdout (log file)
        return None, None, None, None, None, None, None, None


def run_vqe_pennylane(
    openfermion_qubit_op,
    num_active_electrons,
    num_active_spatial_orbitals,
    basis_set_name,
    max_iterations=200,
    verbose=False,
):  # Added basis_set_name
    print(
        f"\n--- Starting VQE Calculation (PennyLane runtime version: v{qml.__version__}) ---"
    )
    num_qubits = 2 * num_active_spatial_orbitals

    # Generate timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_filename = f"dbt_{basis_set_name}_8e8o_{timestamp}"
    energy_data_filename = f"logs/{base_filename}_energies.dat"
    plot_filename = f"logs/{base_filename}_convergence.png"

    try:
        H_pennylane_imported = qml.import_operator(
            openfermion_qubit_op, format="openfermion"
        )
        if isinstance(H_pennylane_imported, qml.Hamiltonian):
            is_complex_coeff = np.any(
                [np.iscomplex(c) for c in H_pennylane_imported.coeffs]
            )
            is_complex_offset = (
                H_pennylane_imported.offset is not None
                and np.iscomplex(H_pennylane_imported.offset)
            )
            if is_complex_coeff or is_complex_offset:
                print(
                    "Warning: PennyLane Hamiltonian from import_operator had complex parts. Taking real components."
                )
                new_coeffs_import = [c.real for c in H_pennylane_imported.coeffs]
                new_offset_import = (
                    H_pennylane_imported.offset.real
                    if H_pennylane_imported.offset is not None
                    else 0.0
                )
                H_pennylane = qml.Hamiltonian(
                    new_coeffs_import,
                    H_pennylane_imported.ops,
                    offset=new_offset_import,
                )
            else:
                H_pennylane = H_pennylane_imported
        else:
            H_pennylane = H_pennylane_imported
    except Exception as e_import:
        print(
            f"Error importing OpenFermion operator to PennyLane: {e_import}. Using manual fallback."
        )
        coeffs_list = []
        obs_list = []
        offset_val = 0.0
        if not openfermion_qubit_op.terms:
            H_pennylane = (
                qml.Hamiltonian([0.0], [qml.Identity(0)])
                if num_qubits > 0
                else qml.Hamiltonian([], [])
            )
        else:
            for term, val in openfermion_qubit_op.terms.items():
                if not term:
                    offset_val += val.real
                    continue
                pauli_word = qml.Identity(0)
                first_op = True
                for qubit_idx, pauli_char in term:
                    op_map = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
                    if pauli_char not in op_map:
                        continue
                    op = op_map[pauli_char](qubit_idx)
                    if first_op:
                        pauli_word = op
                        first_op = False
                    else:
                        pauli_word = pauli_word @ op
                if not first_op:
                    coeffs_list.append(val.real)
                    obs_list.append(pauli_word)
            if not obs_list and offset_val != 0.0:
                H_pennylane = (
                    qml.Hamiltonian([0.0], [qml.Identity(0)], offset=offset_val)
                    if num_qubits > 0
                    else qml.Hamiltonian([], [], offset=offset_val)
                )
            elif not obs_list and offset_val == 0.0:
                H_pennylane = (
                    qml.Hamiltonian([0.0], [qml.Identity(0)])
                    if num_qubits > 0
                    else qml.Hamiltonian([], [])
                )
            else:
                H_pennylane = qml.Hamiltonian(coeffs_list, obs_list, offset=offset_val)

    if verbose:
        print(
            f"PennyLane Hamiltonian num_wires: {H_pennylane.wires.tolist() if H_pennylane.wires else 'N/A'}"
        )
        if hasattr(H_pennylane, "terms"):
            _coeffs, _ops = H_pennylane.terms()
            print(
                f"Number of terms in PennyLane Hamiltonian (excluding offset): {len(_ops)}"
            )
            if isinstance(H_pennylane, qml.Hamiltonian):
                print(f"Offset of PennyLane Hamiltonian: {H_pennylane.offset}")
        else:
            print(f"PennyLane Hamiltonian object type: {type(H_pennylane)}")

    dev = qml.device("lightning.qubit", wires=num_qubits)
    hf_state_config = np.zeros(num_qubits, dtype=int)
    hf_state_config[:num_active_electrons] = 1
    if verbose:
        print(f"Hartree-Fock state configuration (for init_state): {hf_state_config}")

    if verbose:
        print(
            f"Generating excitations for {num_active_electrons} electrons in {num_qubits} spin-orbitals (using qml.qchem.excitations)."
        )
    raw_s_excitations, raw_d_excitations = qchem.excitations(
        electrons=num_active_electrons, orbitals=num_qubits, delta_sz=0
    )

    if verbose:
        print(f"Generated {len(raw_s_excitations)} raw single excitations.")
        if raw_s_excitations:
            print(
                f"  Format of first few raw single excitations: {raw_s_excitations[:3]}"
            )
        print(f"Generated {len(raw_d_excitations)} raw double excitations.")
        if raw_d_excitations:
            print(
                f"  Format of first few raw double excitations: {raw_d_excitations[:3]}"
            )

    s_wires, d_wires = qchem.excitations_to_wires(raw_s_excitations, raw_d_excitations)
    if verbose:
        print(
            f"Processed {len(s_wires)} single excitation wire-sets using excitations_to_wires."
        )
        if s_wires:
            print(f"  Format of first few processed s_wires: {s_wires[:3]}")
        print(
            f"Processed {len(d_wires)} double excitation wire-sets using excitations_to_wires."
        )
        if d_wires:
            print(f"  Format of first few processed d_wires: {d_wires[:3]}")

    if verbose:
        print("\nInspecting the qml.UCCSD template that will be called:")
        print(f"  qml.UCCSD module: {qml.UCCSD.__module__}")
        print(f"  qml.UCCSD qualified name: {qml.UCCSD.__qualname__}")
        try:
            print(f"  File path for qml.UCCSD: {inspect.getfile(qml.UCCSD)}")
            print(
                f"  Signature of qml.UCCSD.__init__: {inspect.signature(qml.UCCSD.__init__)}"
            )
        except Exception as e_inspect:
            print(f"  Could not get file/signature for qml.UCCSD: {e_inspect}")

    num_uccsd_params = len(raw_s_excitations) + len(raw_d_excitations)
    if verbose:
        print(
            f"Total number of UCCSD parameters calculated (based on raw excitations): {num_uccsd_params}"
        )

    @qml.qnode(dev)
    def cost_function(weights):
        qml.UCCSD(
            weights,
            wires=range(num_qubits),
            s_wires=s_wires,
            d_wires=d_wires,
            init_state=hf_state_config,
        )
        return qml.expval(H_pennylane)

    opt = qml.AdamOptimizer(
        stepsize=0.1
    )  # Consider making stepsize configurable or trying other optimizers
    params = pnp.random.normal(0, np.pi / 4, size=num_uccsd_params, requires_grad=True)

    print(f"\nUsing {num_uccsd_params} UCCSD parameters for optimization.")
    print("Attempting to run VQE optimization with PennyLane...")

    energies_for_plot = []

    current_energy_after_step = cost_function(params)
    energies_for_plot.append(current_energy_after_step)  # Initial energy
    best_energy_so_far = current_energy_after_step
    # best_params_so_far = pnp.copy(params, requires_grad=False) # If you need to store best parameters

    print(
        f"Iter {-1:3d}:  E = {current_energy_after_step:.8f} Ha (Initial) | Best E: {best_energy_so_far:.8f} Ha"
    )

    for it in range(max_iterations):
        # params are updated in-place by opt.step_and_cost
        # cost_before_step is the energy evaluated with params *before* this optimization step
        params, cost_before_step = opt.step_and_cost(cost_function, params)

        energies_for_plot.append(
            cost_before_step
        )  # Store energy *before* the step (consistent with old plotting)

        current_energy_after_step = cost_function(
            params
        )  # Energy with the *newly updated* params

        if current_energy_after_step < best_energy_so_far:
            best_energy_so_far = current_energy_after_step
            # best_params_so_far = pnp.copy(params, requires_grad=False) # Update best params

        if (it + 1) % 20 == 0 or it == 0:
            print(
                f"Iter {it:3d}:  E_before_step = {cost_before_step:.8f} Ha | E_after_step = {current_energy_after_step:.8f} Ha | Best E: {best_energy_so_far:.8f} Ha"
            )

        if it == max_iterations - 1:  # After the last step
            # The last energy to complete the plot data, corresponding to the state after the final iteration
            energies_for_plot.append(current_energy_after_step)
            print(
                f"Iter {it:3d}:  E_before_step = {cost_before_step:.8f} Ha | E_after_step = {current_energy_after_step:.8f} Ha | Best E: {best_energy_so_far:.8f} Ha (Optimization finished)"
            )

    if max_iterations == 0:  # Only initial energy was calculated
        pass  # best_energy_so_far is already initial_energy
    elif not energies_for_plot:  # Should not happen if max_iterations >=0
        energies_for_plot.append(best_energy_so_far)

    np.savetxt(
        energy_data_filename, np.array(energies_for_plot), header="Iteration Energy(Ha)"
    )  # Use unique filename
    print(f"\nSaved VQE energies (optimizer path) to {energy_data_filename}")

    # Plotting logic integrated here
    num_energy_points_plot = len(energies_for_plot)
    if num_energy_points_plot > 0:
        if num_energy_points_plot == 1:  # Only initial energy
            iterations_plot = np.array([-1])
        else:
            iterations_plot = np.array(
                [-1]
                + list(range(num_energy_points_plot - 2))
                + [num_energy_points_plot - 2]
            )

        plt.figure(figsize=(10, 6))
        plt.plot(iterations_plot, energies_for_plot, marker="o", linestyle="-")
        plt.xlabel("Optimization Iteration (Adam Step)")
        plt.ylabel("Energy (Hartree)")
        plt.title(
            f"VQE Energy Convergence for DBT ({basis_set_name}, {num_active_electrons}e, {num_active_spatial_orbitals}o)"
        )
        plt.grid(True)
        plt.xticks(
            np.linspace(
                min(iterations_plot),
                max(iterations_plot),
                num=min(20, len(iterations_plot)),
                dtype=int,
            )
        )
        plt.tight_layout()
        plt.savefig(plot_filename)  # Use unique filename
        print(f"Saved convergence plot to {plot_filename}")
        # plt.show() # Optionally show plot, but for batch runs, saving is often preferred.
        plt.close()  # Close the plot figure to free memory
    else:
        print("No energy data to plot.")

    print(
        f"Final VQE Energy (raw value from PennyLane, best found): {best_energy_so_far:.8f} Hartrees"
    )
    return best_energy_so_far


if __name__ == "__main__":
    # === This block is modified to only generate artifacts and then exit ===
    # === Configuration for the current run ===
    current_basis_set = "sto-3g"
    xyz_file = "dibenzothiophene.xyz"
    num_active_orbitals_config = 8
    num_active_electrons_config = 8

    # --- Setup Log File ---
    os.makedirs("logs", exist_ok=True)
    timestamp_log = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = (
        f"logs/dbt_{current_basis_set}_8e8o_{timestamp_log}_artifact_generation.log"
    )
    original_stdout = sys.stdout

    with open(log_filename, "w", encoding="utf-8") as log_file:
        sys.stdout = log_file
        try:
            print(f"--- Script starting to generate artifacts ---")
            print(f"Run configured for basis set: {current_basis_set}")
            print(f"Output is being logged to this file: {log_filename}\n")

            # --- Run the function that calculates and saves the artifacts ---
            mf_obj, openfermion_h_qubit, _, _, _, _, _, _ = (
                run_rhf_get_integrals_qubit_hamiltonian(
                    xyz_file,
                    basis_set=current_basis_set,
                    n_active_orbitals_target=num_active_orbitals_config,
                    n_active_electrons_target=num_active_electrons_config,
                )
            )

            if mf_obj and openfermion_h_qubit:
                print("\n--- Artifact Generation Successful ---")
                print(
                    "Files 'dbt_integrals.npz' and 'qubit_hamiltonian.json' have been generated in the project root."
                )
                print("Script will now exit as requested.")
            else:
                print(
                    f"\n--- Artifact Generation Failed for basis {current_basis_set} ---"
                )

        except Exception as main_execution_error:
            print(f"\nCRITICAL ERROR IN MAIN EXECUTION BLOCK: {main_execution_error}")
            traceback.print_exc(file=sys.stdout)
        finally:
            sys.stdout = original_stdout

    print(f"Script execution finished. Full log available in: {log_filename}")
