# Environment Settings
DOMAIN_SIZE = 1.0
GRID_RES = 0.01
NUM_AGENTS = 10
DENSITY_CENTER = (0.5, 0.5)
DENSITY_SIGMA = 0.1
DENSITY_MODE = 'uniform'  # Options: 'single', 'multi', 'uniform', 'solid_circles', 'solid_squares'

# Lloyd's Algorithm Parameters
LLOYD_MAX_ITERS = 100
LLOYD_TOL = 1e-4

# Simulated Annealing Parameters

# SA_T_INIT: Initial temperature for the annealing process.
# A higher initial temperature allows the algorithm to explore the solution space more freely,
# accepting worse solutions early on to escape local minima. However, if it's too high, it may
# waste time exploring poor solutions. A typical value is between 1.0 and 10.0.
SA_T_INIT = 1.0  # Recommended: Start with 1.0 for moderate exploration.

# SA_ALPHA: Cooling rate, which determines how quickly the temperature decreases.
# A value close to 1.0 (e.g., 0.99 or 0.995) ensures a slow cooling process, allowing more
# thorough exploration. Faster cooling (e.g., 0.9) may lead to premature convergence.
SA_ALPHA = 0.995  # Recommended: Use 0.995 for gradual cooling and better convergence.

# SA_MAX_ITERS: Maximum number of iterations for the algorithm.
# Higher values allow the algorithm to explore the solution space more thoroughly, but at the
# cost of increased computation time. For large problems, this should be balanced with runtime.
SA_MAX_ITERS = 10000  # Recommended: Use 10,000 for a balance between thoroughness and runtime.

# SA_MOVE_SCALE: Maximum scale of random moves for agents.
# Larger values allow agents to explore the solution space more broadly, but may result in
# inefficient exploration if the moves are too large. Smaller values focus on fine-tuning.
SA_MOVE_SCALE = 0.5 

# Agent Initialization Mode
AGENT_INIT_MODE = 'manual'  # Options: 'random', 'manual', 'clustered'
MANUAL_AGENT_POSITIONS = [  # Used only if AGENT_INIT_MODE is 'manual'
    [0.95, 0.1],
    [0.95, 0.2],
    [0.8, 0.3],
    [0.95, 0.4],
    [0.95, 0.5],
    [0.95, 0.6],
    [0.95, 0.7],
    [0.8, 0.8],
    [0.95, 0.9],
    [0.95, 0.95],
]
CLUSTER_CENTER = (0.5, 0.5)  # Used only if AGENT_INIT_MODE is 'clustered'
CLUSTER_SPREAD = 0.1  # Spread for clustered initialization
