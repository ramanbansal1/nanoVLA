import sys
import os
import time
from pathlib import Path
import numpy as np

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import genesis as gs
from datasets import load_dataset, concatenate_datasets, Value
from data.utils import build_episode_lookup
from vis.dataset_mapper import map_horizon_to_g1

def main():
    # 1. Initialize Genesis
    # Will fallback to CPU if GPU backend is not available
    gs.init(backend=gs.gpu) 

    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -2.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.0333, # Horizon action taken after .0333 s or 1/30 s
            gravity=(0, 0, -9.81),
        ),
    )
    
    # Add entities
    plane = scene.add_entity(gs.morphs.Plane())
    
    # Load the specified unitree g1 xml
    xml_path = 'data/unitree_g1/g1_mocap_29dof_with_hands.xml'
    robot = scene.add_entity(gs.morphs.MJCF(file=xml_path, pos=(0.0, 0.0, 0.79)))
    
    scene.build()
    print(f"Robot DoFs: {robot.n_dofs}")
    
    # 2. Load dataset
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
    print(f"Loaded {len(combined_dataset)} transitions from dataset.")
    
    ep_ranges = build_episode_lookup(combined_dataset)
    
    # Pick the first episode to visualize
    ep_id = combined_dataset[0]["episode_index"]
    ep_start, ep_end = ep_ranges[ep_id]
    
    episode_length = ep_end - ep_start + 1
    print(f"Visualizing full episode {ep_id} from index {ep_start} to {ep_end} (length: {episode_length})")

    actions = []
    for k in range(episode_length):
        target_idx = ep_start + k
        actions.append(combined_dataset[target_idx]["action"])
    
    actions = np.array(actions)
    print(f"Actions shape before mapping: {actions.shape}")

    # Map the dataset horizon to the G1 43-actuator space
    actions = map_horizon_to_g1(actions)
    print(f"Actions shape after mapping: {actions.shape}")

    dofs_to_control = min(robot.n_dofs, actions.shape[1])
    
    # 3. Simulate the horizon
    print("Starting simulation loop...")
    for step_idx in range(len(actions)):
        target_action = actions[step_idx]
        
        # Take the required number of dofs
        action_np = np.array(target_action[:dofs_to_control])
        
        # For kinematically replaying mocap datasets, setting DoF positions is usually the most stable way 
        # to visualize what the action values are without fighting physics/PID.
        # Alternatively, robot.control_dofs_position(action_np) can be used if PD gains are set.
        try:
            robot.set_dofs_position(action_np, dofs_idx_local=np.arange(dofs_to_control))
        except AttributeError:
            # Fallback if set_dofs_position is not available
            robot.control_dofs_position(action_np, dofs_idx_local=np.arange(dofs_to_control))
            
        scene.step()
        time.sleep(0.0333)
        
    print("Finished horizon visualization. Running empty (idle)...")
    while True:
        scene.step()
        time.sleep(0.0333)

if __name__ == "__main__":
    main()
