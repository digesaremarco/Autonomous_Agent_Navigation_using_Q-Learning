import numpy as np

# World dimensions
NX = 100
NY = 100

# Orientation
DELTA_THETA_DEG = 5.0
DELTA_THETA_RAD = np.deg2rad(DELTA_THETA_DEG)
N_THETA = int(360.0 / DELTA_THETA_DEG)

# Robot footprint
ROBOT_LENGTH = 3.0
ROBOT_WIDTH = 2.0

# Actions
ACTIONS = {
    'TURN_LEFT': 0,
    'TURN_RIGHT': 1,
    'MOVE_FORWARD': 2
}
N_ACTIONS = len(ACTIONS)
STEP_SIZE = 1.0

# Goal
GOAL_POS = (82, 95)
GOAL_THETA_IDX = int(270.0 / DELTA_THETA_DEG) # 270 degrees
GOAL_STATE = (GOAL_POS[0], GOAL_POS[1], GOAL_THETA_IDX)

# Obstacles (polygons)
OBSTACLES_VERTICES = [
    [(0, 30), (70, 30), (70, 35), (0, 35)],      # bottom wall
    [(100, 60), (30, 60), (30, 65), (100, 65)],  # top wall
    
    # Garage structure
    [(80, 0), (80, 40), (85, 40), (85, 0)],
    [(80, 50), (80, 90), (85, 90), (85, 50)],
    [(85, 0), (85, 100), (100, 100), (100, 0)],
]

# Rewards
R_GOAL = 100.0
R_COLLISION = -100.0
R_STEP = -0.1
R_ROTATE = -0.5
R_DRIFT_PENALTY = -90.0

# RL params
GAMMA = 0.99
VI_CONVERGENCE_THRESHOLD = 1e-4

ALPHA = 0.1
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 10000
N_EPISODES = 50000
MAX_STEPS_PER_EPISODE = 500