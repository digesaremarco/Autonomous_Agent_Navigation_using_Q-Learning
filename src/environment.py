import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
from src import config

class Environment:
    def __init__(self):
        self.config = config
        self.nx = config.NX
        self.ny = config.NY
        self.n_theta = config.N_THETA
        self.goal_state = config.GOAL_STATE
        self.obstacles = [Polygon(v) for v in config.OBSTACLES_VERTICES]

        # Base robot footprint at (0,0) with 0 rotation
        l, w = config.ROBOT_LENGTH, config.ROBOT_WIDTH
        self.base_robot_footprint = Polygon([
            (-l/2, -w/2), (l/2, -w/2), (l/2, w/2), (-l/2, w/2)
        ])

        self.world_boundary = Polygon([(0, 0), (self.nx, 0), (self.nx, self.ny), (0, self.ny)])

    def _get_robot_footprint(self, state):
        x, y, theta_idx = state
        theta_rad = theta_idx * config.DELTA_THETA_RAD
        
        # rotate around its own center before translating
        rotated = rotate(self.base_robot_footprint, theta_rad, origin='center', use_radians=True)
        return translate(rotated, x, y)

    def is_collision(self, state):
        x, y, _ = state
        # Quick pre-check: if center is out, definitely a collision 
        # avoid expensive polygon checks
        if x < 0 or x >= self.nx or y < 0 or y >= self.ny:
            return True

        robot_poly = self._get_robot_footprint(state)
        # Full footprint check (e.g. center is inside, but corners might be out)
        if not self.world_boundary.contains(robot_poly):
            return True

        for obs in self.obstacles:
            if robot_poly.intersects(obs):
                return True

        return False

    def is_goal(self, state):
        x, y, theta = state
        gx, gy, gtheta = self.goal_state

        check_x = int(np.round(x))
        check_y = int(np.round(y))

        return (check_x == gx) and (check_y == gy) and (theta == gtheta)

    def get_sensors(self, state):

        x, y, theta_idx = state
        theta = theta_idx * self.config.DELTA_THETA_RAD

        directions = [
            theta,  # front
            theta + np.pi / 2,  # left
            theta - np.pi / 2  # right
        ]

        distances = []

        for d in directions:

            for r in np.linspace(0, self.config.SENSOR_RANGE, 20):

                sx = x + r * np.cos(d)
                sy = y + r * np.sin(d)

                if self.is_collision((sx, sy, theta_idx)):
                    distances.append(r / self.config.SENSOR_RANGE)
                    break
            else:
                distances.append(1.0)

        return np.array(distances, dtype=np.float32)

    def step(self, state, action, continuous=False):
        if self.is_goal(state):
            return state, 0.0, True

        x, y, theta_idx = state
        next_x, next_y = float(x), float(y)
        next_theta = theta_idx
        reward = 0.0
        terminated = False

        if action == self.config.ACTIONS['TURN_LEFT']:
            next_theta = (theta_idx - 1) % self.n_theta
            reward = self.config.R_ROTATE
        elif action == self.config.ACTIONS['TURN_RIGHT']:
            next_theta = (theta_idx + 1) % self.n_theta
            reward = self.config.R_ROTATE
        elif action == self.config.ACTIONS['MOVE_FORWARD']:
            theta_rad = theta_idx * self.config.DELTA_THETA_RAD
            #compute continuous next position
            cont_x = x + self.config.STEP_SIZE * np.cos(theta_rad)
            cont_y = y + self.config.STEP_SIZE * np.sin(theta_rad)
            if continuous:
                next_x, next_y = cont_x, cont_y
                reward = self.config.R_STEP
            else:
                next_x = int(round(cont_x))
                next_y = int(round(cont_y))
                reward = self.config.R_STEP
        else:
            raise ValueError(f"Invalid action: {action}")

        next_state = (next_x, next_y, next_theta)

        # --- Distance shaping ---
        goal_x, goal_y, _ = self.config.GOAL_STATE

        prev_distance = np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)
        new_distance = np.sqrt((next_x - goal_x) ** 2 + (next_y - goal_y) ** 2)

        # Angle shaping: reward for facing towards the goal
        goal_theta = np.arctan2(goal_y - y, goal_x - x)
        robot_theta = theta_idx * self.config.DELTA_THETA_RAD

        angle_diff = abs(goal_theta - robot_theta)
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)

        reward += 0.1 * (np.pi - angle_diff)

        # Positive reward if moving closer
        reward += 5 * (prev_distance - new_distance)

        # Terminal checks
        if self.is_collision(next_state):
            reward += self.config.R_COLLISION
            terminated = True
            #next_state = state  # stay in place if collision, but allow learning to recover instead of ending episode immediately
            #terminated = False  # don't end episode on collision to allow learning recovery strategies

        elif self.is_goal(next_state):
            reward += self.config.R_GOAL
            terminated = True

        '''next_state = (next_x, next_y, next_theta)

        if self.is_collision(next_state):
            reward = self.config.R_COLLISION
            terminated = True
            # state *in* collision, not the previous one.
        elif self.is_goal(next_state):
            reward = self.config.R_GOAL
            terminated = True'''

        return next_state, reward, terminated


