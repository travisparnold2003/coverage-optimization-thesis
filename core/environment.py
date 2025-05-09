import numpy as np
import logging
from config import settings

def validate_settings():
    """
    Validates the settings to ensure they are within reasonable ranges.
    """
    if settings.GRID_RES <= 0 or settings.GRID_RES >= settings.DOMAIN_SIZE:
        raise ValueError("GRID_RES must be greater than 0 and less than DOMAIN_SIZE.")
    if settings.DENSITY_SIGMA <= 0:
        raise ValueError("DENSITY_SIGMA must be greater than 0.")
    logging.info("Settings validated successfully.")

# Function to generate a 2D grid based on the domain size and grid resolution
def generate_grid():
    """
    Generates a 2D grid of points within the domain.

    Returns:
        X (ndarray): 2D array representing the x-coordinates of the grid.
        Y (ndarray): 2D array representing the y-coordinates of the grid.
        grid_points (ndarray): Flattened array of grid points with shape (N, 2),
                               where N is the total number of grid points.
    """
    logging.debug("Generating grid with DOMAIN_SIZE=%s and GRID_RES=%s", settings.DOMAIN_SIZE, settings.GRID_RES)

    # Create 1D arrays for x and y coordinates based on the domain size and resolution
    x = np.arange(0, settings.DOMAIN_SIZE, settings.GRID_RES)
    y = np.arange(0, settings.DOMAIN_SIZE, settings.GRID_RES)
    logging.debug("Generated x-coordinates: %s", x)
    logging.debug("Generated y-coordinates: %s", y)

    # Create a 2D meshgrid from the x and y coordinates
    X, Y = np.meshgrid(x, y)
    logging.debug("Generated 2D meshgrid X: %s, Y: %s", X, Y)

    # Flatten the grid into a list of points (N, 2) where each row is a point (x, y)
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T
    logging.debug("Flattened grid points: %s", grid_points)

    return X, Y, grid_points

# Function to compute a Gaussian density distribution over the grid
def gaussian_density(grid, center=settings.DENSITY_CENTER, sigma=settings.DENSITY_SIGMA, mode=settings.DENSITY_MODE):
    """
    Computes normalized density values over a 2D grid.

    Args:
        grid (ndarray): Shape (N, 2), where each row is a point (x, y).
        center (tuple): Center of the Gaussian for 'single' mode.
        sigma (float): Standard deviation of the Gaussian.
        mode (str): One of {'single', 'multi', 'uniform', 'solid_circles', 'solid_squares'}.

    Returns:
        ndarray: Normalized density values (shape: N,).
    """
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")

    # Normalizing constant for 2D Gaussian
    norm_const = 1 / (2 * np.pi * sigma**2)

    if mode == 'single':
        # Single Gaussian centered at `center`
        diff = grid - np.array(center)
        squared_dist = np.sum(diff**2, axis=1)
        density = norm_const * np.exp(-squared_dist / (2 * sigma**2))

    elif mode == 'multi':
        # Two Gaussians centered at fixed positions
        centers = [np.array([0.2, 0.7]), np.array([0.2, 0.3])]
        density = np.zeros(grid.shape[0])
        for c in centers:
            diff = grid - c
            squared_dist = np.sum(diff**2, axis=1)
            density += norm_const * np.exp(-squared_dist / (2 * sigma**2))

    elif mode == 'uniform':
        # Uniform density over grid
        density = np.ones(grid.shape[0])

    elif mode == 'solid_circles':
        # Two solid circles on the left side of the grid
        centers = [np.array([0.2, 0.7]), np.array([0.2, 0.3])]
        radius = 0.1  # Radius of the circles
        density = np.zeros(grid.shape[0])
        for c in centers:
            squared_dist = np.sum((grid - c)**2, axis=1)
            density += (squared_dist <= radius**2).astype(float)  # Add 1 for points inside the circle

    elif mode == 'solid_squares':
        # Two solid squares on the left side of the grid
        centers = [np.array([0.2, 0.7]), np.array([0.2, 0.3])]
        side_length = 0.2  # Side length of the squares
        density = np.zeros(grid.shape[0])
        for c in centers:
            in_x_range = (grid[:, 0] >= c[0] - side_length / 2) & (grid[:, 0] <= c[0] + side_length / 2)
            in_y_range = (grid[:, 1] >= c[1] - side_length / 2) & (grid[:, 1] <= c[1] + side_length / 2)
            density += (in_x_range & in_y_range).astype(float)  # Add 1 for points inside the square

    else:
        raise ValueError(f"Unknown density mode: {mode}")

    # Normalize the density to sum to 1
    density /= np.sum(density)
    return density