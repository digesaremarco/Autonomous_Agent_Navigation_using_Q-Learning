import numpy as np
from src import config

class QLearningTabular:
    def __init__(self, env):
        self.env = env

        self.gamma = config.GAMMA
        self.alpha = config.ALPHA

        self.epsilon = config.EPSILON_START
        self.epsilon_start = config.EPSILON_START
        self.epsilon_end = config.EPSILON_END
        self.epsilon_decay_steps = config.EPSILON_DECAY_STEPS

        # Q-table 4D
        self.Q = np.zeros(
            (config.NX, config.NY, config.N_THETA, config.N_ACTIONS),
            dtype=np.float32
        )

        self.total_steps = 0

    def select_action(self, state):
        '''
        Epsilon-greedy action selection
        param state: tuple (x, y, theta_idx)
        return: action index
        '''

        x, y, theta = state

        if np.random.rand() < self.epsilon:
            return np.random.choice(config.N_ACTIONS)
        else:
            return np.argmax(self.Q[x, y, theta])

    '''def update_epsilon(self):
        
        Linear decay of epsilon over time
        

        fraction = min(self.total_steps / self.epsilon_decay_steps, 1.0)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)'''

    def update_epsilon_episode(self, episode):
        fraction = min(episode / config.N_EPISODES, 1.0)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def update_epsilon(self, episode):
        """
        Linear decay of epsilon over time
        """
        epsilon_decay = (config.EPSILON_START - config.EPSILON_END) / config.N_EPISODES
        self.epsilon = max(config.EPSILON_END, self.epsilon - epsilon_decay)

    def train(self):
        """
        Q-learning training loop with tracking of goals and collisions
        """
        goals_reached = 0
        collisions_count = 0

        for episode in range(1, config.N_EPISODES + 1):

            # --- Sample a valid initial state ---
            while True:
                x = np.random.randint(0, config.NX)
                y = np.random.randint(0, config.NY)
                theta = np.random.randint(0, config.N_THETA)

                state = (int(x), int(y), int(theta))

                if not self.env.is_collision(state):
                    break
            #state = (20, 20, 0)  # start state

            episode_reward = 0.0
            done = False

            # --- Episode loop ---
            for step in range(config.MAX_STEPS_PER_EPISODE):

                # Select action (epsilon-greedy)
                action = self.select_action(state)

                # Environment step
                next_state, reward, done = self.env.step(state, action)
                next_state = tuple(map(int, next_state))

                episode_reward += reward

                x, y, theta = state
                nx, ny, ntheta = next_state

                # --- Q-learning update ---
                best_next_q = 0.0 if done else np.max(self.Q[nx, ny, ntheta])

                td_target = reward + self.gamma * best_next_q
                td_error = td_target - self.Q[x, y, theta, action]

                self.Q[x, y, theta, action] += self.alpha * td_error

                # Move to next state
                state = next_state
                self.total_steps += 1

                # Update epsilon after each step
                self.update_epsilon(self.total_steps)

                if done:
                    # Track the outcome
                    if reward == config.R_GOAL:
                        goals_reached += 1
                    elif reward == config.R_COLLISION:
                        collisions_count += 1
                    break

            # --- Logging with diagnostics ---
            if episode % 1000 == 0:
                success_rate = (goals_reached / episode) * 100
                print(
                    f"Episode {episode:5d}/{config.N_EPISODES} | "
                    f"Reward: {episode_reward:7.2f} | "
                    f"Epsilon: {self.epsilon:.4f} | "
                    f"Success Rate: {success_rate:5.1f}% ({goals_reached:5d} goals) | "
                    f"Collisions: {collisions_count:5d}"
                )

            # Update epsilon after each episode
            self.update_epsilon_episode(episode)

        print(f"\nTraining Complete!")
        print(f"Final Statistics - Goals: {goals_reached}, Collisions: {collisions_count}")
        print(f"Final Success Rate: {(goals_reached / config.N_EPISODES) * 100:.2f}%")

    def get_greedy_action(self, state):
        '''
        Get the action with the highest Q-value for a given state
        '''

        x, y, theta = state
        x, y, theta = int(x), int(y), int(theta)
        return np.argmax(self.Q[x, y, theta])

    def extract_policy(self):
        '''
        Extract the greedy policy from the Q-table
        '''

        return np.argmax(self.Q, axis=3)  # shape (NX, NY, N_THETA) with action indices

    def evaluate(self, n_episodes=100):
        '''
        Evaluate the learned policy by running episodes and measuring success rate
        '''

        success_count = 0

        for episode in range(n_episodes):
            state = (0, 0, 0)  # start state
            done = False

            for step in range(config.MAX_STEPS_PER_EPISODE):
                action = self.get_greedy_action(state)
                next_state, reward, done = self.env.step(state, action)

                state = next_state

                if done:
                    if self.env.is_goal(state):
                        success_count += 1
                    break

        success_rate = success_count / n_episodes
        print(f"Evaluation: Success Rate = {success_rate:.2%} ({success_count}/{n_episodes})")

    def save_model(self, filename='q_learning_model.npz'):
        '''
        Save the Q-table to a file
        '''

        np.savez_compressed(filename, Q=self.Q)
        print(f"Model saved to {filename}")

    def load_model(self, filename='q_learning_model.npz'):
        '''
        Load the Q-table from a file
        param filename: path to the saved model file
        return: True if load successful, False otherwise
        '''

        try:
            data = np.load(filename)
            self.Q = data['Q']
            print(f"Model loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"Model file {filename} not found. Starting with empty Q-table.")
            return False
        except Exception as e:
            print(f"Error loading model from {filename}: {e}")
            return False