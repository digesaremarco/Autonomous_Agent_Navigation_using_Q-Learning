import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import matplotlib.animation as animation

def plot_static_path(env, path, title="Simulation Path"):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.set_xlim(0, env.nx)
    ax.set_ylim(0, env.ny)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Draw obstacles
    for obstacle in env.obstacles:
        ax.add_patch(MplPolygon(
            obstacle.exterior.coords, 
            closed=True, 
            color='gray', 
            alpha=0.8
        ))
        
    # Draw goal position
    goal_footprint = env._get_robot_footprint(env.goal_state)
    ax.add_patch(MplPolygon(
        goal_footprint.exterior.coords, 
        closed=True, 
        color='green', 
        alpha=0.7, 
        label='Goal'
    ))
    
    # Draw start position
    start_footprint = env._get_robot_footprint(path[0])
    ax.add_patch(MplPolygon(
        start_footprint.exterior.coords, 
        closed=True, 
        color='blue', 
        alpha=0.7, 
        label='Start'
    ))
    
    # Draw path center line
    x_coords = [state[0] for state in path]
    y_coords = [state[1] for state in path]
    ax.plot(x_coords, y_coords, 'r--', label='Path (Center)')
    
    ax.legend()
    ax.set_title(title)
    
    # Save to file
    filename = f"{title.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.png"
    plt.savefig(filename)
    print(f"Static path image saved: {filename}")
    
    plt.close(fig)

def animate_path(env, path, title="Animation"):
    # Note: Using brute-force MplPolygon update (blit=False) for simplicity and compatibility.
    # Saves only as GIF to avoid external dependencies like FFMpeg.
    print(f"Starting GIF animation for: {title}")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, env.nx)
    ax.set_ylim(0, env.ny)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Static elements
    for obstacle in env.obstacles:
        ax.add_patch(MplPolygon(
            obstacle.exterior.coords, closed=True, color='gray', alpha=0.8
        ))
        
    goal_footprint = env._get_robot_footprint(env.goal_state)
    ax.add_patch(MplPolygon(
        goal_footprint.exterior.coords, closed=True, color='green', alpha=0.7, label='Goal'
    ))

    # Dynamic elements setup
    start_state = path[0]
    initial_footprint = env._get_robot_footprint(start_state)
    
    robot_patch = MplPolygon(
        initial_footprint.exterior.coords, closed=True, color='blue', alpha=0.8, label='Robot'
    )
    ax.add_patch(robot_patch)

    path_line, = ax.plot([], [], 'r--', label='Path (Center)')
    
    ax.legend(loc='upper left')
    ax.set_title(title)

    def update_frame(frame_index):
        current_state = path[frame_index]
        ax.set_title(f"{title} (Step: {frame_index+1}/{len(path)})")
        
        # Update robot polygon vertices
        current_footprint = env._get_robot_footprint(current_state)
        robot_patch.set_xy(current_footprint.exterior.coords)
        
        # Update path line data
        x_data = [s[0] for s in path[:frame_index+1]]
        y_data = [s[1] for s in path[:frame_index+1]]
        path_line.set_data(x_data, y_data)
        
        return robot_patch, path_line

    interval_ms = 100 
    ani = animation.FuncAnimation(
        fig, update_frame, frames=len(path),
        interval=interval_ms, blit=False, repeat=False
    )
    
    clean_title = title.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
    filename_gif = f"{clean_title}.gif"
    
    print(f"Saving GIF (this might take a while): {filename_gif}")
    try:
        ani.save(filename_gif, writer='pillow', fps=1000/interval_ms)
        print(f"GIF saved successfully: {filename_gif}")
    except Exception as e:
        print(f"!!! ERROR saving GIF: {e}")

    plt.close(fig)