import os

import matplotlib.pyplot as plt
import pandas as pd
from adapt_vqe_driver import run_adapt_vqe_baseline


def save_and_plot_results(trace_data):
    """Saves trace data to CSV and generates plots."""
    if not trace_data:
        print("No data to save or plot.")
        return

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Create DataFrame with the 4 columns returned by the driver
    df = pd.DataFrame(
        trace_data, columns=["iteration", "energy", "num_params", "cnot_depth"]
    )
    csv_path = os.path.join("results", "adapt_trace.csv")
    df.to_csv(csv_path, index=False)
    print(f"Trace data saved to {csv_path}")

    # --- Generate Plots ---
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Energy Convergence Plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df["iteration"], df["energy"], marker="o", linestyle="-", color="b")
    ax1.set_xlabel("ADAPT Iteration")
    ax1.set_ylabel("Energy (Ha)")
    ax1.set_title("ADAPT-VQE Energy Convergence")
    ax1.grid(True)
    plot1_path = os.path.join("results", "convergence_plot.png")
    fig1.savefig(plot1_path, bbox_inches="tight")
    print(f"Convergence plot saved to {plot1_path}")
    plt.close(fig1)

    # 2. CNOT Depth Plot (Bar Plot)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(df["iteration"], df["cnot_depth"], color="c")
    ax2.set_xlabel("ADAPT Iteration")
    ax2.set_ylabel("Total CNOT Depth")
    ax2.set_title("Circuit CNOT Depth per Iteration")
    ax2.grid(axis="y")
    if len(df["iteration"]) < 25:
        ax2.set_xticks(df["iteration"])
    plot2_path = os.path.join("results", "depth_plot.png")
    fig2.savefig(plot2_path, bbox_inches="tight")
    print(f"CNOT depth plot saved to {plot2_path}")
    plt.close(fig2)


if __name__ == "__main__":
    # Run the simulation
    final_trace_data = run_adapt_vqe_baseline()

    # Save and plot the results
    if final_trace_data:
        save_and_plot_results(final_trace_data)
        print(
            "\nBaseline run complete. Artifacts generated in 'results/' and 'circuits/'."
        )
    else:
        print("Simulation did not produce data.")
