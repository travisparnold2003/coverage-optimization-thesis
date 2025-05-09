import numpy as np
from .coverage_utils import compute_centroids, coverage_cost
import logging
from config import settings

def default_callback(positions, cost):
    logging.debug("Intermediate state captured. Cost: %.6f", cost)

def run_lloyd(agent_positions, grid_points, density,
              max_iters=settings.LLOYD_MAX_ITERS, tol=settings.LLOYD_TOL, callback=None):
    """
    Runs Lloyd's algorithm to optimize agent positions for coverage.

    Args:
        agent_positions (ndarray): Initial positions of agents with shape (M, 2), where M is the number of agents.
        grid_points (ndarray): Array of grid points with shape (N, 2), where N is the number of grid points.
        density (ndarray): Array of density values with shape (N,).
        max_iters (int): Maximum number of iterations to run the algorithm (default: 100).
        tol (float): Convergence tolerance for agent movement (default: 1e-4).
        callback (function): Optional callback function to capture intermediate states.

    Returns:
        tuple:
            - ndarray: Final optimized agent positions with shape (M, 2).
            - list: History of coverage costs over iterations.
    """
    logging.info("Starting Lloyd's algorithm with %d agents and %d grid points.", agent_positions.shape[0], grid_points.shape[0])
    logging.info("Maximum iterations: %d, Tolerance: %.6f", max_iters, tol)

    # Use the default callback if none is provided
    if callback is None:
        callback = default_callback

    # Initialize a list to store the history of coverage costs
    history = []

    for iteration in range(max_iters):
        logging.debug("Iteration %d started.", iteration + 1)

        # Compute new agent positions (density-weighted centroids)
        new_positions = compute_centroids(agent_positions, grid_points, density)
        logging.debug("New agent positions computed: %s", new_positions)

        # Compute the maximum movement of agents
        movement = np.max(np.linalg.norm(agent_positions - new_positions, axis=1))
        logging.debug("Maximum agent movement: %.6f", movement)

        # Compute the coverage cost for the new positions
        cost = coverage_cost(new_positions, grid_points, density)
        logging.debug("Coverage cost computed: %.6f", cost)

        # Append the cost to the history
        history.append(cost)

        # Call the callback function if provided
        if callback:
            callback(new_positions, cost)

        # Update agent positions
        agent_positions = new_positions

        # Check for convergence
        if movement < tol:
            logging.info("Convergence achieved after %d iterations. Maximum movement: %.6f", iteration + 1, movement)
            break
    else:
        logging.warning("Maximum iterations reached without convergence.")

    logging.info("Lloyd's algorithm completed. Final coverage cost: %.6f", history[-1])
    return agent_positions, history
