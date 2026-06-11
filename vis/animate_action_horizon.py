import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset, concatenate_datasets, Value
from data.utils import build_episode_lookup
from vis.dataset_mapper import map_horizon_to_g1

def main():
    # 1. Load dataset (similar to your visualization script)
    datasets_root_dir = Path('data/datasets')
    
    if not datasets_root_dir.exists():
        print(f"Error: Dataset directory {datasets_root_dir} not found.")
        return

    dataset_repos = sorted([str(p) for p in datasets_root_dir.iterdir() if p.is_dir()])
    
    if not dataset_repos:
        print("No datasets found in data/datasets.")
        return

    all_datasets = []
    for repo in dataset_repos:
        ds = load_dataset(repo)['train']
        if ds.features["timestamp"].dtype == "float32":
            ds = ds.cast_column("timestamp", Value("float64"))
        all_datasets.append(ds)

    combined_dataset = concatenate_datasets(all_datasets)
    
    ep_ranges = build_episode_lookup(combined_dataset)
    
    # Pick the episode to visualize
    ep_id = combined_dataset[10]["episode_index"]
    ep_start, ep_end = ep_ranges[ep_id]
    
    action_horizon = 30

    actions = []
    for k in range(action_horizon):
        target_idx = min(ep_start + k, ep_end)
        actions.append(combined_dataset[target_idx]["action"])
    
    actions = np.array(actions) # Shape: (H, 30)
    print(f"Loaded episode {ep_id}. Action shape before mapping: {actions.shape}")
    
    # Map the dataset horizon to the G1 43-actuator space
    actions = map_horizon_to_g1(actions)
    print(f"Action shape after mapping: {actions.shape}")
    
    H, A = actions.shape
    
    dt = 0.0333
    time_steps = np.arange(H) * dt
    
    # 2. Setup Matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Give some padding to the limits
    ax.set_xlim(0, time_steps[-1])
    ax.set_ylim(np.min(actions) - 0.5, np.max(actions) + 0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint Position (rad)")
    ax.set_title(f"Action Horizon Time Series (Episode {ep_id})")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # We will use a colormap to differentiate lines
    cmap = plt.get_cmap('tab20')
    lines = [ax.plot([], [], lw=1.5, color=cmap(i % 20))[0] for i in range(A)]
    
    def init():
        for line in lines:
            line.set_data([], [])
        return lines
        
    def update(frame):
        # 'frame' goes from 0 to H-1. 
        # Update each line's data to progressively reveal the time series.
        for i, line in enumerate(lines):
            line.set_data(time_steps[:frame+1], actions[:frame+1, i])
        return lines
        
    # 3. Create the animation
    ani = FuncAnimation(
        fig, 
        update, 
        frames=H, 
        init_func=init, 
        blit=True, 
        interval=int(dt * 1000), # Interval in ms (approx 33ms)
        repeat=True,
        repeat_delay=1000 # wait 1s before repeating
    )
    
    # Save as GIF
    save_path = Path(__file__).resolve().parent / "action_horizon_timeseries.gif"
    print(f"Saving animation to {save_path} ...")
    try:
        ani.save(save_path, writer='pillow', fps=int(1/dt))
        print(f"Saved successfully to {save_path}!")
    except Exception as e:
        print(f"Could not save animation. Error: {e}")
        
    # Also attempt to display it interactively if running in a supported environment
    plt.show()

if __name__ == "__main__":
    main()
