import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import logging
from matplotlib.animation import FuncAnimation

def plot_costs(cost_lloyd, cost_sa):
    """
    Plots the convergence of Lloyd's Algorithm and Simulated Annealing.

    Args:
        cost_lloyd (list): Coverage cost history for Lloyd's Algorithm.
        cost_sa (list): Coverage cost history for Simulated Annealing.

    Returns:
        matplotlib.figure.Figure: The figure object for the plot.
    """
    logging.debug("Plotting convergence comparison for Lloyd's Algorithm and Simulated Annealing.")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cost_lloyd, label="Lloyd's Algorithm", color='blue')
    ax.plot(cost_sa, label="Simulated Annealing", color='red')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Coverage Cost")
    ax.set_title("Convergence Comparison")
    ax.legend()
    ax.grid(True)
    
    logging.debug("Convergence comparison plot completed.")
    return fig  # Return the figure object

def plot_voronoi(points, title="Voronoi Diagram"):
    vor = Voronoi(points)
    voronoi_plot_2d(vor, show_vertices=False, show_points=True)
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal')
    plt.show()

def plot_grid_and_density(X, Y, density):
    """
    Visualizes the grid points and overlays the Gaussian density function as a heatmap.

    Args:
        X (ndarray): 2D array representing the x-coordinates of the grid.
        Y (ndarray): 2D array representing the y-coordinates of the grid.
        density (ndarray): 1D array of density values corresponding to each grid point.

    Returns:
        matplotlib.figure.Figure: The figure object for the plot.
    """
    logging.debug("Plotting grid and density heatmap.")
    
    # Reshape the density array to match the grid dimensions
    density_reshaped = density.reshape(X.shape)
    logging.debug("Reshaped density array to match grid dimensions: %s", density_reshaped.shape)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X, Y, s=1, color='blue', label='Grid Points', alpha=0.5)
    heatmap = ax.imshow(density_reshaped, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', 
                        cmap='hot', alpha=0.6)
    fig.colorbar(heatmap, ax=ax, label='Density')
    ax.set_title("Grid Points and Gaussian Density Heatmap")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True)
    logging.debug("Finished plotting grid and density heatmap.")
    return fig  # Return the figure object

def plot_voronoi_partition(agent_positions, grid_points, title="Voronoi Partition"):
    """
    Visualizes the Voronoi partition and agent positions.

    Args:
        agent_positions (ndarray): Array of agent positions with shape (M, 2).
        grid_points (ndarray): Array of grid points with shape (N, 2).
        title (str): Title of the plot.

    Returns:
        matplotlib.figure.Figure: The figure object for the plot.
    """
    logging.debug("Plotting Voronoi partition for %d agents.", agent_positions.shape[0])
    
    # Compute Voronoi diagram
    vor = Voronoi(agent_positions)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False)
    
    # Plot agent positions
    ax.scatter(agent_positions[:, 0], agent_positions[:, 1], c='red', label='Agents', zorder=2)
    
    # Plot grid points
    ax.scatter(grid_points[:, 0], grid_points[:, 1], c='blue', s=1, label='Grid Points', alpha=0.5, zorder=1)
    
    # Set fixed axis limits and aspect ratio
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')  # Keep the aspect ratio square
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True)
    logging.debug("Voronoi partition plot completed.")
    return fig  # Return the figure object

def animate_lloyd(agent_positions, grid_points, density, run_lloyd_func, interval=500):
    """
    Animates Lloyd's algorithm, showing the agents' positions, Voronoi regions, and coverage cost convergence.
    """
    logging.info("Starting animation for Lloyd's algorithm.")

    # Run Lloyd's algorithm and capture intermediate states
    positions_history = []
    cost_history = []

    def wrapped_run_lloyd(*args, **kwargs):
        nonlocal positions_history, cost_history
        def callback(positions, cost):
            positions_history.append(positions.copy())
            cost_history.append(cost)
        return run_lloyd_func(*args, **kwargs, callback=callback)

    # Run the algorithm with the callback to capture states
    final_positions, cost_history = wrapped_run_lloyd(agent_positions, grid_points, density)

    # Set up the figure
    fig, (ax_voronoi, ax_cost) = plt.subplots(1, 2, figsize=(12, 6))
    ax_voronoi.set_title("Voronoi Diagram")
    ax_voronoi.set_xlim(0, 1)
    ax_voronoi.set_ylim(0, 1)
    ax_voronoi.set_aspect('equal')  # Fix aspect ratio
    ax_cost.set_title("Coverage Cost Convergence")
    ax_cost.set_xlabel("Iteration")
    ax_cost.set_ylabel("Coverage Cost")

    # Initialize plots
    voronoi_plot = None
    cost_line, = ax_cost.plot([], [], label='Coverage Cost', color='blue')
    ax_cost.legend()

    def update(frame):
        nonlocal voronoi_plot

        # Clear previous Voronoi regions
        for collection in list(ax_voronoi.collections):  # Iterate over a copy of the list
            collection.remove()

        # Update Voronoi diagram
        positions = positions_history[frame]
        vor = Voronoi(positions)
        voronoi_plot_2d(vor, ax=ax_voronoi, show_vertices=False, show_points=False)
        scatter = ax_voronoi.scatter(positions[:, 0], positions[:, 1], c='red', label='Agents')

        # Update coverage cost plot
        cost_line.set_data(range(frame + 1), cost_history[:frame + 1])
        ax_cost.set_xlim(0, len(cost_history))
        ax_cost.set_ylim(0, max(cost_history) * 1.1)

        return scatter, cost_line  # Return updated artists

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(positions_history), interval=interval, blit=False)
    plt.show()

    logging.info("Animation completed.")
    return final_positions, len(cost_history), cost_history


def animate_simulated_annealing(agent_positions, grid_points, density, run_sa_func, interval=50):
    """
    Animates the simulated annealing process, showing the agents' positions, Voronoi partitions,
    and coverage cost convergence.
    """
    logging.info("Starting animation for simulated annealing.")

    # Run simulated annealing and capture intermediate states
    positions_history = []
    cost_history = []

    def wrapped_run_sa(*args, **kwargs):
        nonlocal positions_history, cost_history
        def callback(positions, cost):
            positions_history.append(positions.copy())
            cost_history.append(cost)
        return run_sa_func(*args, **kwargs, callback=callback)

    # Run the algorithm with the callback to capture states
    final_positions, cost_history, acceptance_history, accepted, rejected = wrapped_run_sa(agent_positions, grid_points, density)

    # Set up the figure
    fig, (ax_voronoi, ax_cost) = plt.subplots(1, 2, figsize=(12, 6))
    ax_voronoi.set_title("Voronoi Diagram")
    ax_voronoi.set_xlim(0, 1)
    ax_voronoi.set_ylim(0, 1)
    ax_voronoi.set_aspect('equal')  # Fix aspect ratio
    ax_cost.set_title("Coverage Cost Convergence")
    ax_cost.set_xlabel("Iteration")
    ax_cost.set_ylabel("Coverage Cost")

    # Initialize plots
    voronoi_plot = None
    cost_line, = ax_cost.plot([], [], label='Coverage Cost', color='blue')
    ax_cost.legend()

    def update(frame):
        nonlocal voronoi_plot

        # Clear previous Voronoi regions
        for collection in list(ax_voronoi.collections):  # Iterate over a copy of the list
            collection.remove()

        # Update Voronoi diagram
        positions = positions_history[frame]
        vor = Voronoi(positions)
        voronoi_plot_2d(vor, ax=ax_voronoi, show_vertices=False, show_points=False)
        scatter = ax_voronoi.scatter(positions[:, 0], positions[:, 1], c='red', label='Agents')

        # Update coverage cost plot
        cost_line.set_data(range(frame + 1), cost_history[:frame + 1])
        ax_cost.set_xlim(0, len(cost_history))
        ax_cost.set_ylim(0, max(cost_history) * 1.1)

        return scatter, cost_line  # Return updated artists

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(positions_history), interval=interval, blit=False)
    plt.show()

    logging.info("Animation completed.")
    return final_positions, cost_history, acceptance_history, accepted, rejected

def plot_acceptance_over_time(acceptance_history):
    """
    Plots accepted vs. rejected solutions over time.
    """
    import matplotlib.pyplot as plt
    accepted = np.cumsum(acceptance_history)
    total = np.arange(1, len(acceptance_history) + 1)
    rejected = total - accepted

    plt.figure(figsize=(10, 5))
    plt.plot(total, accepted, label='Cumulative Accepted', color='green')
    plt.plot(total, rejected, label='Cumulative Rejected', color='red')
    plt.xlabel("Iteration")
    plt.ylabel("Count")
    plt.title("Accepted vs. Rejected Solutions Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

