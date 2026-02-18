# Autonomous Agent Navigation using Value Iteration

This project implements an autonomous agent capable of navigating a 2D grid environment with obstacles, using a tabular approach based on **Value Iteration**. The agent controls a non-point robot (with physical footprint and orientation) and learns an optimal policy to reach a target position and orientation `(x, y, theta)`, avoiding collisions and respecting movement constraints.

## Key Features

* **Grid Environment:** Discrete `NX x NY` world with polygonal obstacles.
* **Agent State:** The agent's state is defined by `(x, y, theta_idx)`, where `theta_idx` represents the discrete orientation (72 possible angles).
* **Agent Actions:** The agent has three discrete actions: `TURN_LEFT`, `TURN_RIGHT`, `MOVE_FORWARD`.
* **Sim-to-Real Validation:** The project implements an evaluation pipeline that tests the discrete policy in a Continuous Environment, highlighting the "Sim-to-Real" gap.
* **Collision Detection:** Uses the `Shapely` library to calculate the exact intersection between the robot's rotated footprint and obstacles, including map boundaries as impassable barriers.
* **Advanced Reward Shaping:** Uses a system of rewards and penalties to guide learning:
    * `R_GOAL`: Positive reward for reaching the target.
    * `R_COLLISION`: Severe penalty for collisions with obstacles or boundaries.
    * `R_STEP`: Cost for each step to incentivize short paths.
    * `R_ROTATE`: Cost for rotations to avoid unnecessary movements.
    * `R_DRIFT_PENALTY`: Specific penalty to discourage physically unrealistic movements ("drift") caused by grid discretization, forcing clean driving maneuvers.

## Project Structure

    autonomous_agent_pjwk/
    ├── src/
    │   ├── config.py        # Configuration parameters (map, robot, rewards)
    │   ├── environment.py   # World logic, physics, and collisions
    │   ├── planning.py      # Value Iteration algorithm
    │   ├── visualizer.py    # Functions for generating plots and animations    
    │   └── main.py          # Main script for training and simulation
    ├── README.md            # Project documentation (English Version)
    ├── results/             # Images and animations of multiple experiments
    │   ├── Value Iteration  # Value Iteration results 
    │   └── Q-Learning       # Q-Learning results 
    └── requirements.txt     # Python dependencies

## Implementation Details

### 1. Environment and Physics (`src/environment.py`)
The core of the simulation is the `Environment` class, which handles discrete world physics.
* **State:** Each state is a tuple `(x, y, theta_idx)`.
* **Transitions:** The `step(state, action)` function calculates the next state by applying the robot's kinematics. Continuous movements are discretized ("snapped") to the nearest grid cell.
* **Collisions:** The `is_collision(state)` function uses `Shapely` to create a rotated polygon representing the robot's exact footprint and checks for intersection with obstacles or exiting map boundaries.
* **Evaluation Mode (Continuous):** Movements use floating-point precision. The robot moves in continuous space, simulating real-world physics where grid snapping does not exist.

### 2. Planning with Value Iteration (`src/planning.py`)
The `ValueIterationPlanner` solves the navigation problem by discretely calculating the Value Function $V(s)$ for each state.
* **Pre-computation:** For efficiency, a boolean `collision_map` is pre-computed for all 720,000 possible states, drastically speeding up training.
* **Bellman Equation:** The algorithm iterates through all states applying the update:
    $$V_{k+1}(s) = \max_a [ R(s,a,s') + \gamma V_k(s') ]$$
    until convergence.
* **Drift Penalty:** During reward calculation $R(s,a,s')$, the dot product between the intended movement vector (based on angle) and the actual movement vector (based on grid) is calculated. A penalty is applied if these vectors diverge, discouraging "dirty" movements.

### 3. Visualization (`src/visualizer.py`)
Uses `Matplotlib` to create graphical representations of policies.
* **Static Plots:** Draws the full path, obstacles, and goal on a 2D grid.
* **Animations:** Generates animated GIFs showing the robot moving step by step.

## 4. Configuration (`src/config.py`)

This file centralizes all modifiable parameters.

* **World Dimensions (`NX`, `NY`):** Define grid resolution.
* **Robot Parameters:** `ROBOT_LENGTH` and `ROBOT_WIDTH` define physical footprint, while `DELTA_THETA_DEG` defines possible rotation granularity (e.g., 5°).
* **Map:** `OBSTACLES_VERTICES` contains the list of polygons forming obstacles, and `GOAL_STATE` defines the target position and orientation.
* **Reward System (`R_*`):** Defines scalar weights for all event types (goal, collisions, steps, rotations, drift), which directly determine agent behavior.

### 5. Main Script (`main.py`)
The application entry point that orchestrates the entire process.
* **Initialization:** Creates instances of `Environment` and `ValueIterationPlanner`.
* **Model Management:** Checks if pre-trained models (`v.npy`, `policy.npy`) exist. If not found, it automatically starts pre-computation and training.
* **Testing and Validation:** Tests the agent in the ideal grid world where it learned.
* **Continuous Simulation:**Tests the agent in a realistic continuous world. The script handles the translation between the continuous robot position and the discrete policy lookup (using nearest-neighbor rounding).

## How to Run

1.  **Install dependencies:**
    Run the following instruction (Python needed):
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start the simulation:**
    Run the main script:
    ```bash
    python main.py
    ```
    * On the first run, the script will perform the **collision map pre-computation** (may take a few minutes) and the **Value Iteration** algorithm (on a 7th gen Intel Core i3 CPU, full training took approximately 4 hours).
    * Trained models (`v.npy`, `policy.npy`) will be saved for faster future executions.
    * Several test simulations will be run, saving results as static images (`.png`) and animations (`.gif`).

## Results

### Final Results (Optimized Policy)
These results show the agent's behavior with the final configuration, which includes a severe penalty for drift (`R_DRIFT_PENALTY = -90.0`) and correct map boundary checks. The agent demonstrates clean and safe driving.

|                               Simulation from (10, 10, 0°) - Static                               | Simulation from (10, 10, 0°) - Animation |
|:-------------------------------------------------------------------------------------------------:| :---: |
| <img src="results/Value Iteration/results_without_drift/Simulazione_da_10_10_0.png" width="100%"> | <img src="results/Value Iteration/results_without_drift/Animazione_da_10_10_0.gif" width="100%"> |

| Simulation from (50, 50, 90°) - Static | Simulation from (50, 50, 90°) - Animation |
| :---: | :---: |
| <img src="results/Value Iteration/results_without_drift/Simulazione_da_50_50_18.png" width="100%"> | <img src="results/Value Iteration/results_without_drift/Animazione_da_50_50_18.gif" width="100%"> |

| Simulation from (70, 72, 0°) - Static | Simulation from (70, 72, 0°) - Animation |
| :---: | :---: |
| <img src="results/Value Iteration/results_without_drift/Simulazione_da_70_72_0.png" width="100%"> | <img src="results/Value Iteration/results_without_drift/Animazione_da_70_72_0.gif" width="100%"> |

---

### Analysis of Solved Problems
During development, two critical challenges were addressed that compromised simulation realism.

#### 1. Exiting Map Boundaries
Initially, the system only checked if the robot's *center* was within the grid. This allowed the robot, having a physical footprint, to partially "break through" outer map walls with its perimeter.
* **Solution:** A rigorous geometric check was implemented in `environment.py` that verifies if the entire robot polygon (footprint) is contained within world boundaries, treating map edges as impassable walls.

#### 2. "Drifting" and Unrealistic Movements
Due to grid discretization, the agent could move in one direction (e.g., North) while oriented slightly differently (e.g., 85°), creating an unrealistic "drift" or sideways sliding effect.
* **Solution:** A **Drift Penalty** (`R_DRIFT_PENALTY`) was introduced. This penalty calculates the dot product between the intended movement vector (based on angle) and the actual one (on grid). By setting a very high penalty (-90.0), the agent was forced to learn that only perfectly aligned movements are acceptable, completely eliminating drift behavior.

### Behavior Before Corrections
This example shows agent behavior *before* the above corrections. Note how the robot tends to "slide" sideways in tight turns to avoid complex rotations and how it can partially exit top map boundaries in some cases.

|                                           Simulation with Issues - Static                                            |                                           Simulation with Issues - Animation                                           |
|:--------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------:|
| <img src="results/Value Iteration/breaking_boundries_and_low_drift_penalty/Simulazione_da_10_10_0.png" width="100%"> | <img src="results/Value Iteration/breaking_boundries_and_low_drift_penalty/Animazione_da_10_10_0(1).gif" width="100%"> |

---

### Sim-to-Real Gap Analysis
A key objective of this project is to evaluate how a policy learned in a discrete world performs in a continuous environment.

* **Success Cases:** In open areas (e.g., starting from `50, 50` or `70, 72`), the agent successfully navigates in continuous mode. The "Drift Penalty" effectively taught the agent to align itself with the grid axes, minimizing trajectory errors.
* **Edge Cases:** In tight spaces (e.g., starting from `10, 10` near obstacles), the continuous simulation may fail where the discrete one succeeds.

|                          Continuous Simulation from (10, 10, 0°) - Static                           | Continuous Simulation from (10, 10, 0°) - Animation |
|:---------------------------------------------------------------------------------------------------:| :---: |
| <img src="results\Value Iteration\continous_simulations\Sim_1_Continuous_10_10_0.png" width="100%"> | <img src="results\Value Iteration\continous_simulations\Anim_1_Continuous_10_10_0.gif" width="100%"> |

| Continuous Simulation from (50, 50, 90°) - Static | Continuous Simulation from (50, 50, 90°) - Animation |
| :---: | :---: |
| <img src="results\Value Iteration\continous_simulations\Sim_2_Continuous_50_50_18.png" width="100%"> | <img src="results\Value Iteration\continous_simulations\Anim_2_Continuous_50_50_18.gif" width="100%"> |

| Continuous Simulation from (70, 72, 0°) - Static | Continuous Simulation from (70, 72, 0°) - Animation |
| :---: | :---: |
| <img src="results\Value Iteration\continous_simulations\Sim_3_Continuous_70_72_0.png" width="100%"> | <img src="results\Value Iteration\continous_simulations\Anim_3_Continuous_70_72_0.gif" width="100%"> |
