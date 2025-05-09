import numpy as np
import logging
from .coverage_utils import coverage_cost, compute_voronoi_partition, compute_centroids
from config import settings
from scipy.spatial import cKDTree

def validate_annealing_params(T_init, alpha, move_scale):
    if T_init <= 0:
        raise ValueError("T_init must be greater than 0.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")
    if move_scale <= 0:
        raise ValueError("move_scale must be greater than 0.")
    logging.info("Simulated annealing parameters validated successfully.")

def perturb_agents(current, T, move_scale, grid_points, density):
    """
    All agents are perturbed every iteration.
    Each move is a random direction OR biased toward the centroid.
    Probability of centroid increases as T decreases.
    Move distance also shrinks over time.
    """
    M = current.shape[0]
    candidate = current.copy()

    # Compute centroids for all agents
    centroids = compute_centroids(current, grid_points, density)
    
    # Linearly scale toward centroid as T cools
    bias_prob = 1.0 - T / settings.SA_T_INIT
    max_move = move_scale * (T / settings.SA_T_INIT)  # Shrinks over time

    for i in range(M):
        direction_to_centroid = centroids[i] - current[i]
        direction_to_centroid /= np.linalg.norm(direction_to_centroid) + 1e-8

        random_direction = np.random.uniform(-1, 1, 2)
        random_direction /= np.linalg.norm(random_direction) + 1e-8

        if np.random.rand() < bias_prob:
            direction = direction_to_centroid
        else:
            direction = random_direction

        step_size = np.random.uniform(0, max_move)
        move = direction * step_size
        candidate[i] = np.clip(current[i] + move, 0, 1)

    return candidate

# def perturb_agents(current, move_scale):
#     """
#     Perturb all agents in purely random directions.
#     """
#     M = current.shape[0]
#     candidate = current.copy()

#     for i in range(M):
#         random_direction = np.random.uniform(-1, 1, 2)
#         random_direction /= np.linalg.norm(random_direction) + 1e-8

#         step_size = np.random.uniform(0, move_scale/10)
#         move = random_direction * step_size

#         # Apply move and clip to unit square bounds [0, 1]
#         candidate[i] = np.clip(current[i] + move, 0, 1)

#     return candidate


def run_simulated_annealing(agent_positions, grid_points, density,
                            T_init=settings.SA_T_INIT, alpha=settings.SA_ALPHA,
                            max_iters=settings.SA_MAX_ITERS, move_scale=settings.SA_MOVE_SCALE,
                            callback=None):
    """
    Simplified Simulated Annealing: all agents move each iteration with biased randomness.
    Tracks and returns acceptance statistics.
    """
    validate_annealing_params(T_init, alpha, move_scale)
    logging.info("Starting Simulated Annealing with %d agents, %d grid points.", agent_positions.shape[0], grid_points.shape[0])

    current = agent_positions.copy()
    cost = coverage_cost(current, grid_points, density)
    costs = [cost]
    T = T_init

    accepted_moves = 0
    rejected_moves = 0
    acceptance_history = []

    for iter_num in range(max_iters):
        candidate = perturb_agents(current, T, move_scale, grid_points, density)
        #candidate = perturb_agents(current, move_scale)
        new_cost = coverage_cost(candidate, grid_points, density)
        dE = new_cost - cost

        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            current = candidate
            cost = new_cost
            accepted_moves += 1
            acceptance_history.append(1)
        else:
            rejected_moves += 1
            acceptance_history.append(0)

        costs.append(cost)

        if callback:
            callback(current, cost)

        T *= alpha
        if T < 1e-5:
            logging.info("Temperature threshold reached; stopping at iteration %d.", iter_num)
            break

    logging.info("SA completed. Accepted: %d | Rejected: %d | Total: %d",
                 accepted_moves, rejected_moves, accepted_moves + rejected_moves)

    return current, costs, acceptance_history, accepted_moves, rejected_moves


