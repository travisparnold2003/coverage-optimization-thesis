import numpy as np
from scipy.spatial import cKDTree
import logging

def compute_voronoi_partition(agent_positions, grid_points):
    """
    Assigns each grid point to the nearest agent using a KD-tree.

    Args:
        agent_positions (ndarray): Array of agent positions with shape (M, 2), where M is the number of agents.
        grid_points (ndarray): Array of grid points with shape (N, 2), where N is the number of grid points.

    Returns:
        ndarray: Array of indices indicating the nearest agent for each grid point.
    """
    logging.debug("Computing Voronoi partition for %d agents and %d grid points.", agent_positions.shape[0], grid_points.shape[0])
    
    # Build a KD-tree for efficient nearest-neighbor search
    tree = cKDTree(agent_positions)
    logging.debug("KD-tree built for agent positions.")
    
    # Query the KD-tree to find the nearest agent for each grid point
    _, indices = tree.query(grid_points)
    logging.debug("Voronoi partition computed. Assigned grid points to nearest agents.")
    
    return indices

def compute_centroids(agent_positions, grid_points, density):
    """
    Computes the density-weighted centroids of Voronoi regions.

    Args:
        agent_positions (ndarray): Array of agent positions with shape (M, 2).
        grid_points (ndarray): Array of grid points with shape (N, 2).
        density (ndarray): Array of density values with shape (N,).

    Returns:
        ndarray: Array of updated agent positions (centroids) with shape (M, 2).
    """
    logging.debug("Computing centroids for %d agents.", agent_positions.shape[0])
    
    # Compute Voronoi partition
    indices = compute_voronoi_partition(agent_positions, grid_points)
    logging.debug("Voronoi partition obtained for centroid computation.")
    
    # Initialize centroids array
    centroids = np.zeros_like(agent_positions)
    
    # Compute centroids for each agent
    for i in range(agent_positions.shape[0]):
        logging.debug("Processing agent %d.", i)
        
        # Mask to identify grid points in the current agent's region
        mask = indices == i
        region_points = grid_points[mask]
        weights = density[mask]
        mass = np.sum(weights)
        
        if mass > 0:
            # Compute the density-weighted centroid
            centroids[i] = np.sum(region_points * weights[:, None], axis=0) / mass
            logging.debug("Centroid for agent %d computed: %s", i, centroids[i])
        else:
            # If no mass, keep the agent's position unchanged
            centroids[i] = agent_positions[i]
            logging.debug("No mass in region for agent %d. Keeping original position: %s", i, centroids[i])
    
    logging.debug("Centroid computation completed.")
    return centroids

def coverage_cost(agent_positions, grid_points, density):
    """
    Computes the total coverage cost as the weighted sum of squared distances.

    Args:
        agent_positions (ndarray): Array of agent positions with shape (M, 2).
        grid_points (ndarray): Array of grid points with shape (N, 2).
        density (ndarray): Array of density values with shape (N,).

    Returns:
        float: Total coverage cost.
    """
    logging.debug("Computing coverage cost for %d agents and %d grid points.", agent_positions.shape[0], grid_points.shape[0])
    
    # Build a KD-tree for efficient nearest-neighbor search
    tree = cKDTree(agent_positions)
    logging.debug("KD-tree built for agent positions.")
    
    # Query the KD-tree to find the nearest agent for each grid point
    _, indices = tree.query(grid_points)
    logging.debug("Assigned grid points to nearest agents for coverage cost computation.")
    
    # Compute squared distances between grid points and their assigned agents
    distances = np.linalg.norm(grid_points - agent_positions[indices], axis=1)**2
    logging.debug("Squared distances computed for all grid points.")
    
    # Compute the total coverage cost as the weighted sum of squared distances
    total_cost = np.sum(density * distances)
    logging.debug("Total coverage cost computed: %f", total_cost)

    # Normalize by total density mass
    normalized_cost = total_cost / np.sum(density)
    return normalized_cost
