from src.environment import Environment
from src.planning import ValueIterationPlanner
from src.visualizer import plot_static_path, animate_path
from src import config 
import numpy as np

def simulate_policy(planner, env, start_state, continuous_mode=False):
    mode_str = "CONTINUOUS" if continuous_mode else "DISCRETE"
    print(f"\n Starting {mode_str} Simulation from {start_state} ")

    path = [start_state]
    current_state = start_state
    max_steps = config.NX * config.NY 

    for i in range(max_steps):
        x, y, theta = current_state

        ix = int(np.round(x))
        iy = int(np.round(y))
        ix = max(0, min(ix, config.NX - 1))
        iy = max(0, min(iy, config.NY - 1))

        action = planner.policy[ix, iy, theta]

        if action == -1:
            if env.is_goal(current_state):
                print(f"RESULT: SUCCESS! Goal reached in {i} steps.")
            else:
                print(f"RESULT: FAILURE. Policy -1 (Stall/Collision) at {current_state} in {i} steps.")
            break 

        next_state, reward, terminated = env.step(current_state, action, continuous=continuous_mode)
        path.append(next_state)
        current_state = next_state

        if terminated:
            if env.is_goal(current_state):
                print(f"RESULT: SUCCESS! Goal reached in {i+1} steps.")
            else:
                print(f"RESULT: COLLISION! (State {current_state}) in {i+1} steps.")
            break
    else:
        print(f"RESULT: TIMEOUT! {max_steps} steps limit reached.")

    # Path summary for quick debugging
    if len(path) > 10:
        print("Path (first 5):")
        for j in range(5): print(f"  {j}: {path[j]}")
        print("  ...")
        print("Path (last 5):")
        for j in range(len(path) - 5, len(path)): print(f"  {j}: {path[j]}")
    else:
        print("Path:")
        for j, state in enumerate(path): print(f"  {j}: {state}")
            
    return path

def run_policy_tests(planner):
    print("\nStarting Optimal Policy Tests")
    
    # Test cases format: (name, state, assertion_lambda)
    test_cases = [
        ("Collision State (30, 30, 0)", (30, 30, 0), lambda p: p == -1),
        ("Goal State", config.GOAL_STATE, lambda p: p == -1),
        ("Safe State (10, 10, 0)", (10, 10, 0), lambda p: p != -1),
        ("Safe State (50, 50, 18)", (50, 50, 18), lambda p: p != -1),
        # Specific scenarios near goal might need adjustment if goal coords change
        ("Pre-Goal (forward)", (82, 94, config.GOAL_STATE[2]), lambda p: p == config.ACTIONS['MOVE_FORWARD']),
    ]
    
    success_count = 0
    for name, state, test_lambda in test_cases:
        try:
            x, y, theta = state
            # Handle potential out-of-bounds if test cases are bad
            if not (0 <= x < config.NX and 0 <= y < config.NY):
                 print(f"[ERROR]   Test '{name}': State {state} out of bounds!")
                 continue

            policy_action = planner.policy[x, y, theta]
            if test_lambda(policy_action):
                print(f"[SUCCESS] Test '{name}': Correct (Action: {policy_action})")
                success_count += 1
            else:
                print(f"[FAILED]  Test '{name}': Failed (Action: {policy_action})")
        except Exception as e:
            print(f"[ERROR]   Test '{name}': Exception! {e}")
            
    print("Policy Tests Completed")
    print(f"Result: {success_count} / {len(test_cases)} tests passed.")
    
    unique, counts = np.unique(planner.policy, return_counts=True)
    print("\nPolicy Action Summary:")
    for action, count in zip(unique, counts):
        print(f"  Action {action}: {count} states")

if __name__ == "__main__":
    env = Environment()
    print("Environment created.")

    planner = ValueIterationPlanner(env)
    print("Planner created.")

    # Try loading first, otherwise train from scratch
    if not planner.load_model():
        print("No saved model found. Starting pre-computation and training...")
        planner.precompute_collision_map()
        planner.run_value_iteration()
    else:
        print("Models loaded. Re-running pre-computation for collision map...")
        # We need the collision map for simulation, even if V is loaded
        planner.precompute_collision_map()

    run_policy_tests(planner)

    print("\n RUNNING SIMULATIONS AND VISUALIZATION")

    # Simulation scenarios
    start_states = [
        (10, 10, 0),
        (50, 50, 18),
        (70, 72, 0)
    ]

    for i, start_state in enumerate(start_states, 1):
        if planner.policy[start_state] != -1:
            
            #SIMULAZIONE 1: DISCRETA
            print(f"--- Running Discrete Sim {i} ---")
            path_disc = simulate_policy(planner, env, start_state, continuous_mode=False)
            plot_static_path(env, path_disc, title=f"Sim_{i}_Discrete_{start_state}")
            animate_path(env, path_disc, title=f"Anim_{i}_Discrete_{start_state}")

            # SIMULAZIONE 2: CONTINUA
            print(f"--- Running Continuous Sim {i} ---")
            path_cont = simulate_policy(planner, env, start_state, continuous_mode=True)
            plot_static_path(env, path_cont, title=f"Sim_{i}_Continuous_{start_state}")
            animate_path(env, path_cont, title=f"Anim_{i}_Continuous_{start_state}")

        else:
            print(f"\nSkipping simulation from {start_state} (Invalid starting policy -1)")