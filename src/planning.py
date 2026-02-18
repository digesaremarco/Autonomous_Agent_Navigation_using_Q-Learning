import numpy as np
import time
import os
from src import config

class ValueIterationPlanner:
    def __init__(self, environment):
        self.env = environment
        self.config = config
        self.nx, self.ny = config.NX, config.NY
        self.n_theta, self.n_actions = config.N_THETA, config.N_ACTIONS
        
        # Initialize tables. V starts at 0, policy at -1 (no action).
        self.V = np.zeros((self.nx, self.ny, self.n_theta))
        self.policy = np.full((self.nx, self.ny, self.n_theta), -1, dtype=int)
        
        # Pre-allocate maps for quick lookups during VI loops
        self.collision_map = np.zeros((self.nx, self.ny, self.n_theta), dtype=bool)
        self.goal_map = np.zeros((self.nx, self.ny, self.n_theta), dtype=bool)
        
        gx, gy, gtheta = config.GOAL_STATE
        self.goal_map[gx, gy, gtheta] = True

    def precompute_collision_map(self):
        # This might take a minute, but it saves massive time during the VI loop 
        # by avoiding repeated complex polygon intersection checks.
        print("Pre-calculating Collision Map")
        start_time = time.time()
        count = 0
        for x in range(self.nx):
            if (x + 1) % 10 == 0:
                print(f"Processing... {x + 1}/{self.nx} (collisions: {count})")
            for y in range(self.ny):
                for theta_idx in range(self.n_theta):
                    if self.env.is_collision((x, y, theta_idx)):
                        self.collision_map[x, y, theta_idx] = True
                        count += 1
        print(f"Done in {time.time() - start_time:.2f}s. Total collisions: {count}")

    def _get_next_state_reward(self, state, action):
        x, y, theta_idx = state
        # If already at goal, stay there with 0 reward
        if self.goal_map[x, y, theta_idx]:
            return state, 0.0, True

        next_x, next_y, next_theta_idx = x, y, theta_idx
        drift_penalty = 0.0
        base_reward = 0.0

        if action == self.config.ACTIONS['TURN_LEFT']:
            next_theta_idx = (theta_idx - 1) % self.n_theta
            base_reward = self.config.R_ROTATE
        elif action == self.config.ACTIONS['TURN_RIGHT']:
            next_theta_idx = (theta_idx + 1) % self.n_theta
            base_reward = self.config.R_ROTATE
        elif action == self.config.ACTIONS['MOVE_FORWARD']:
            base_reward = self.config.R_STEP
            theta_rad = theta_idx * self.config.DELTA_THETA_RAD
            
            # Calculate continuous next position and snap to grid
            cont_x = x + self.config.STEP_SIZE * np.cos(theta_rad)
            cont_y = y + self.config.STEP_SIZE * np.sin(theta_rad)
            next_x, next_y = int(np.round(cont_x)), int(np.round(cont_y))

            # Drift penalty: Snapping to grid can cause misalignment between intended 
            # angle and actual move. We penalize this to avoid "weird" zig-zag paths.
            if next_x != x or next_y != y:
                move_vec = np.array([next_x - x, next_y - y])
                move_norm = move_vec / np.linalg.norm(move_vec)
                intended_vec = np.array([np.cos(theta_rad), np.sin(theta_rad)])
                
                # Dot product is 1.0 if perfectly aligned, less otherwise.
                drift_error = max(0.0, 1.0 - np.dot(intended_vec, move_norm))
                drift_penalty = self.config.R_DRIFT_PENALTY * drift_error
        
        next_state = (next_x, next_y, next_theta_idx)

        # Check boundary and collisions
        if not (0 <= next_x < self.nx and 0 <= next_y < self.ny):
            return next_state, self.config.R_COLLISION, True
        if self.collision_map[next_state]:
            return next_state, self.config.R_COLLISION, True
        if self.goal_map[next_state]:
            return next_state, self.config.R_GOAL, True

        return next_state, base_reward + drift_penalty, False

    def run_value_iteration(self):
        print("\nStarting Value Iteration")
        start_time = time.time()
        iteration = 0
        
        while True:
            iteration += 1
            delta = 0.0
            # Use a copy for synchronous updates (standard VI stability)
            V_old = np.copy(self.V)
            
            for x in range(self.nx):
                for y in range(self.ny):
                    for theta_idx in range(self.n_theta):
                        state = (x, y, theta_idx)
                        
                        # Force terminal state values
                        if self.collision_map[state]:
                            self.V[state] = self.config.R_COLLISION
                            continue
                        if self.goal_map[state]:
                            self.V[state] = self.config.R_GOAL
                            continue

                        # Bellman update
                        q_values = np.full(self.n_actions, -np.inf)
                        for action in range(self.n_actions):
                            next_s, r, term = self._get_next_state_reward(state, action)
                            q_values[action] = r if term else r + self.config.GAMMA * V_old[next_s]
                        
                        best_value = np.max(q_values)
                        delta = max(delta, np.abs(best_value - V_old[state]))
                        self.V[state] = best_value

            print(f"Iteration {iteration}: Max Delta = {delta:.6f}")
            if delta < self.config.VI_CONVERGENCE_THRESHOLD:
                break
        
        print(f"Value Iteration finished in {iteration} iterations ({time.time() - start_time:.2f}s)")
        self.extract_policy()
        self.save_model()

    def extract_policy(self):
        print("Extracting optimal policy...")
        # Re-run one pass of Q-value calculation to find the best action for each state
        for x in range(self.nx):
            for y in range(self.ny):
                for theta_idx in range(self.n_theta):
                    state = (x, y, theta_idx)
                    if self.collision_map[state] or self.goal_map[state]:
                        self.policy[state] = -1
                        continue
                    
                    q_values = np.full(self.n_actions, -np.inf)
                    for action in range(self.n_actions):
                        next_s, r, term = self._get_next_state_reward(state, action)
                        q_values[action] = r if term else r + self.config.GAMMA * self.V[next_s]
                    
                    # argmax gives us the action index with the highest expected return
                    self.policy[state] = np.argmax(q_values)
        print("Policy extracted.")

    def save_model(self, v_file='v.npy', policy_file='policy.npy'):
        print(f"Saving model to {v_file} and {policy_file}...")
        np.save(v_file, self.V)
        np.save(policy_file, self.policy)
        print("Save complete.")

    def load_model(self, v_file='v.npy', policy_file='policy.npy'):
        if os.path.exists(v_file) and os.path.exists(policy_file):
            print(f"Loading model from {v_file} and {policy_file}...")
            self.V = np.load(v_file)
            self.policy = np.load(policy_file)
            print("Load complete.")
            return True
        print("Model files not found. Starting fresh training.")
        return False