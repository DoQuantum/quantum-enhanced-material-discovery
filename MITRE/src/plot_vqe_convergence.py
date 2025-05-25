import numpy as np
import matplotlib.pyplot as plt

def plot_energy_convergence(energy_file="vqe_energies.dat"):
    """
    Plots VQE energy convergence from a data file.
    The first energy is initial (iteration -1), subsequent are after each step.
    """
    try:
        energies = np.loadtxt(energy_file)
    except Exception as e:
        print(f"Error loading energy file '{energy_file}': {e}")
        return

    # Iteration numbers: start from -1 (initial) then 0 to N-1 for steps
    # If energies_over_iterations stores N+2 values (initial, N steps, final after Nth step)
    # then iterations should be -1, 0, 1, ..., N-1, N
    # If energies_over_iterations stores initial + N values from prev_E
    # then iterations are -1, 0, ..., N-1
    
    # Based on the saving logic:
    # energies[0] is initial (iter -1)
    # energies[1] is E after step 0 (iter 0)
    # ...
    # energies[max_iterations] is E after step max_iterations-1 (iter max_iterations-1)
    # energies[max_iterations+1] is E after final step (if max_iterations > 0)
    
    num_energy_points = len(energies)
    if num_energy_points == 0:
        print("No energy data to plot.")
        return

    # iterations = np.arange(-1, num_energy_points - 1) # if energies are [initial, E_iter0, E_iter1, ..., E_iter_N-1_final_step]
    
    # Correcting iteration axis based on the modified saving logic:
    # energies_over_iterations will have:
    # 1 (initial) + max_iterations (prev_E from each step) + 1 (final after last step, if max_iter > 0)
    # = max_iterations + 2 points if max_iterations > 0
    # = 1 point if max_iterations == 0
    
    if num_energy_points == 1: # Only initial energy
        iterations = np.array([-1])
    else:
        # Initial (-1), then iter 0, 1, ..., max_iterations-1 (from prev_E), then final (effectively iter max_iterations)
        iterations = np.array([-1] + list(range(num_energy_points - 2)) + [num_energy_points - 2])


    plt.figure(figsize=(10, 6))
    plt.plot(iterations, energies, marker='o', linestyle='-')
    plt.xlabel("Optimization Iteration (Adam Step)")
    plt.ylabel("Energy (Hartree)")
    plt.title("VQE Energy Convergence for Dibenzothiophene (STO-3G, 8e,8o)")
    plt.grid(True)
    plt.xticks(np.linspace(min(iterations), max(iterations), num=min(20, len(iterations)), dtype=int)) # Adjust for better tick display
    plt.tight_layout()
    plt.savefig("vqe_convergence_plot.png")
    print("Saved convergence plot to vqe_convergence_plot.png")
    plt.show()

if __name__ == "__main__":
    plot_energy_convergence()