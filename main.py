import os
import numpy as np
import time
from datetime import datetime
from core.environment import generate_grid, gaussian_density
from core.lloyd import run_lloyd
from core.annealing import run_simulated_annealing
from visual.plotter import plot_grid_and_density, plot_voronoi_partition, animate_lloyd, animate_simulated_annealing, plot_costs, plot_acceptance_over_time
from config import settings
import logging
from core.environment import validate_settings as validate_environment_settings
import matplotlib.pyplot as plt  # Import for showing plots
import platform
import psutil

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)         # Suppress PIL debug logs

def create_results_folder(experiment_name):
    """
    Creates a unique folder for saving results based on experimental parameters.
    Returns the folder path.

    Args:
        experiment_name (str): Custom name for the experiment.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = (
        f"{experiment_name}_results_{timestamp}_agents{settings.NUM_AGENTS}_"
        f"density{settings.DENSITY_SIGMA}_T{settings.SA_T_INIT}_alpha{settings.SA_ALPHA}_"
        f"lloydIters{settings.LLOYD_MAX_ITERS}_saIters{settings.SA_MAX_ITERS}"
    )
    folder_path = os.path.join("results", folder_name)
    os.makedirs(folder_path, exist_ok=True)
    logging.info("Results folder created: %s", folder_path)
    return folder_path

def save_plot(fig, folder_path, filename):
    """
    Saves a matplotlib figure to the specified folder.

    Args:
        fig (matplotlib.figure.Figure): The figure to save.
        folder_path (str): The folder path where the figure will be saved.
        filename (str): The name of the file (e.g., "grid_density.png").
    """
    file_path = os.path.join(folder_path, filename)
    fig.savefig(file_path)
    logging.info("Plot saved: %s", file_path)

def save_results_summary(folder_path, algorithm_name, final_cost, iterations, time_taken):
    """
    Saves a summary of the results (final cost, iterations, time taken) to a text file.

    Args:
        folder_path (str): The folder path where the summary will be saved.
        algorithm_name (str): Name of the algorithm (e.g., "Lloyd's Algorithm").
        final_cost (float): Final coverage cost.
        iterations (int): Number of iterations completed.
        time_taken (float): Total time taken in seconds.
    """
    summary_file = os.path.join(folder_path, f"{algorithm_name.lower().replace(' ', '_')}_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Algorithm: {algorithm_name}\n")
        f.write(f"Final Coverage Cost: {final_cost:.6f}\n")
        f.write(f"Iterations: {iterations}\n")
        f.write(f"Time Taken: {time_taken:.2f} seconds\n")
    logging.info("Results summary saved: %s", summary_file)

def save_experiment_details(folder_path):
    """
    Saves all programmable settings and system specifications to a text file.

    Args:
        folder_path (str): The folder path where the details will be saved.
    """
    details_file = os.path.join(folder_path, "experiment_details.txt")
    with open(details_file, "w") as f:
        # Save environment settings
        f.write("=== Environment Settings ===\n")
        f.write(f"DOMAIN_SIZE: {settings.DOMAIN_SIZE}\n")
        f.write(f"GRID_RES: {settings.GRID_RES}\n")
        f.write(f"NUM_AGENTS: {settings.NUM_AGENTS}\n")
        f.write(f"DENSITY_CENTER: {settings.DENSITY_CENTER}\n")
        f.write(f"DENSITY_SIGMA: {settings.DENSITY_SIGMA}\n")
        f.write(f"DENSITY_MODE: {settings.DENSITY_MODE}\n\n")

        # Save Lloyd's Algorithm parameters
        f.write("=== Lloyd's Algorithm Parameters ===\n")
        f.write(f"LLOYD_MAX_ITERS: {settings.LLOYD_MAX_ITERS}\n")
        f.write(f"LLOYD_TOL: {settings.LLOYD_TOL}\n\n")

        # Save Simulated Annealing parameters
        f.write("=== Simulated Annealing Parameters ===\n")
        f.write(f"SA_T_INIT: {settings.SA_T_INIT}\n")
        f.write(f"SA_ALPHA: {settings.SA_ALPHA}\n")
        f.write(f"SA_MAX_ITERS: {settings.SA_MAX_ITERS}\n")
        f.write(f"SA_MOVE_SCALE: {settings.SA_MOVE_SCALE}\n\n")

        # Save agent initialization mode
        f.write("=== Agent Initialization ===\n")
        f.write(f"AGENT_INIT_MODE: {settings.AGENT_INIT_MODE}\n")
        if settings.AGENT_INIT_MODE == 'manual':
            f.write(f"MANUAL_AGENT_POSITIONS: {settings.MANUAL_AGENT_POSITIONS}\n")
        elif settings.AGENT_INIT_MODE == 'clustered':
            f.write(f"CLUSTER_CENTER: {settings.CLUSTER_CENTER}\n")
            f.write(f"CLUSTER_SPREAD: {settings.CLUSTER_SPREAD}\n")

        # Save system specifications
        f.write("\n=== System Specifications ===\n")
        f.write(f"OS: {platform.system()} {platform.release()} ({platform.version()})\n")
        f.write(f"Processor: {platform.processor()}\n")
        f.write(f"CPU Cores: {psutil.cpu_count(logical=False)}\n")
        f.write(f"Logical CPUs: {psutil.cpu_count(logical=True)}\n")
        f.write(f"Total Memory: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB\n")
        f.write(f"Available Memory: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB\n")
        f.write(f"Python Version: {platform.python_version()}\n")

    logging.info("Experiment details saved: %s", details_file)

def initialize_agent_positions():
    """
    Initializes agent positions based on the selected mode in settings.

    Returns:
        ndarray: Array of agent positions with shape (NUM_AGENTS, 2).
    """
    if settings.AGENT_INIT_MODE == 'random':
        np.random.seed(40)
        return np.random.rand(settings.NUM_AGENTS, 2)
    elif settings.AGENT_INIT_MODE == 'manual':
        return np.array(settings.MANUAL_AGENT_POSITIONS)
    elif settings.AGENT_INIT_MODE == 'clustered':
        np.random.seed(42)
        cluster_center = np.array(settings.CLUSTER_CENTER)
        return cluster_center + settings.CLUSTER_SPREAD * (np.random.rand(settings.NUM_AGENTS, 2) - 0.5)
    else:
        raise ValueError(f"Unknown AGENT_INIT_MODE: {settings.AGENT_INIT_MODE}")

def main():
    """
    Main function to run the coverage optimization algorithms (Lloyd's and Simulated Annealing)
    and visualize their results.
    """
    try:
        logging.info("Starting the coverage optimization process.")

        save_results = input("Do you want to save the results? (yes/no): ").strip().lower()
        if save_results == "yes":
            experiment_name = input("Enter a custom name for this experiment: ").strip()
            folder_path = create_results_folder(experiment_name)
            save_experiment_details(folder_path)
        else:
            folder_path = None

        # Step 1: Validate settings
        validate_environment_settings()

        # Step 2: Generate the grid and compute the density function
        logging.info("Generating the grid and computing the density function.")
        X, Y, grid = generate_grid()
        density = gaussian_density(grid, mode=settings.DENSITY_MODE)
        logging.info("Grid and density function generated successfully.")

        # Step 3: Visualize the grid and density
        logging.info("Visualizing the grid and density function.")
        fig_density = plot_grid_and_density(X, Y, density)
        if folder_path:
            save_plot(fig_density, folder_path, "grid_density.png")

        # Step 4: Initialize agent positions
        logging.info("Initializing agent positions.")
        initial_positions = initialize_agent_positions()

        # Step 5: Visualize the initial Voronoi partition
        logging.info("Visualizing the initial Voronoi partition.")
        fig_voronoi = plot_voronoi_partition(initial_positions, grid, title="Initial Voronoi Partition")
        if folder_path:
            save_plot(fig_voronoi, folder_path, "initial_voronoi_partition.png")

        # Step 6: Run and animate Lloyd's Algorithm
        logging.info("Running and animating Lloyd's Algorithm.")
        print("Running and animating Lloyd's Algorithm...")
        final_positions_lloyd, _, cost_history_lloyd = animate_lloyd(
            initial_positions, grid, density, run_lloyd_func=run_lloyd
        )
        if folder_path:
            save_results_summary(folder_path, "Lloyd's Algorithm", cost_history_lloyd[-1], len(cost_history_lloyd), 0)
            # Save the final Voronoi partition for Lloyd's Algorithm
            logging.info("Saving final Voronoi partition for Lloyd's Algorithm.")
            fig_final_voronoi_lloyd = plot_voronoi_partition(final_positions_lloyd, grid, title="Final Voronoi Partition (Lloyd's Algorithm)")
            save_plot(fig_final_voronoi_lloyd, folder_path, "final_voronoi_partition_lloyd.png")

        # Step 7: Run Simulated Annealing (with acceptance tracking)
        logging.info("Running Simulated Annealing.")
        print("Running Simulated Annealing...")

        final_positions_sa, cost_history_sa, acceptance_history, accepted, rejected = animate_simulated_annealing(
            initial_positions, grid, density, run_sa_func=run_simulated_annealing
        )


        if folder_path:
            save_results_summary(folder_path, "Simulated Annealing", cost_history_sa[-1], len(cost_history_sa), 0)

            # Save final Voronoi
            logging.info("Saving final Voronoi partition for Simulated Annealing.")
            fig_final_voronoi_sa = plot_voronoi_partition(final_positions_sa, grid, title="Final Voronoi Partition (Simulated Annealing)")
            save_plot(fig_final_voronoi_sa, folder_path, "final_voronoi_partition_sa.png")

            # Save acceptance stats
            acceptance_summary_path = os.path.join(folder_path, "experiment_details.txt")
            with open(acceptance_summary_path, "a") as f:
                f.write("\n=== Simulated Annealing Acceptance Stats ===\n")
                f.write(f"Accepted Moves: {accepted}\n")
                f.write(f"Rejected Moves: {rejected}\n")
                f.write(f"Total Moves: {accepted + rejected}\n")
                f.write(f"Acceptance Rate: {accepted / (accepted + rejected):.4f}\n")

        # Plot acceptance stats
        plot_acceptance_over_time(acceptance_history)
        print(f"SA Acceptance Stats — Accepted: {accepted}, Rejected: {rejected}, Total: {accepted + rejected}")


        # Step 8: Plot and show convergence graph
        logging.info("Plotting convergence graph.")
        fig_costs = plot_costs(cost_history_lloyd, cost_history_sa)
        if folder_path:
            save_plot(fig_costs, folder_path, "convergence_graph.png")
        plt.show()

        logging.info("Coverage optimization process completed.")
    except Exception as e:
        logging.error("An error occurred: %s", str(e))
        raise

if __name__ == "__main__":
    main()
