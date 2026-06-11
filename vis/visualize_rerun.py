import argparse
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import genesis as gs
import rerun as rr
from datasets import load_dataset
import sys
sys.path.append('vis')
from dataset_mapper import map_horizon_to_g1

def parse_mjcf_visual_geoms(xml_path):
    """
    Parses the MJCF XML file to extract visual geoms for each body.
    Returns a dictionary mapping body name to a list of geoms:
        body_name -> [{"mesh_path": str, "pos": [x,y,z], "quat": [w,x,y,z]}]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Map mesh names to file names
    mesh_assets = {}
    for asset in root.iter('asset'):
        for mesh in asset.findall('mesh'):
            name = mesh.get('name', mesh.get('file').split('.')[0])
            mesh_assets[name] = mesh.get('file')
            
    # Find visual geoms for each body
    body_geoms = {}
    xml_dir = Path(xml_path).parent
    
    # Helper to parse string arrays
    def parse_array(s, default):
        if not s: return default
        return [float(x) for x in s.split()]
        
    for body in root.iter('body'):
        body_name = body.get('name')
        geoms = []
        for geom in body.findall('geom'):
            # Check if it's a visual geom
            if geom.get('class') == 'visual' or geom.get('mesh') is not None:
                mesh_name = geom.get('mesh')
                if not mesh_name:
                    continue
                    
                mesh_file = mesh_assets.get(mesh_name, mesh_name)
                # Ensure the file path ends with .STL
                if not mesh_file.lower().endswith('.stl'):
                    mesh_file += '.STL'
                
                # We assume the meshes are in the same dir as xml or under meshes/
                mesh_path = xml_dir / mesh_file
                if not mesh_path.exists():
                    mesh_path = xml_dir / 'meshes' / mesh_file
                    
                pos = parse_array(geom.get('pos'), [0.0, 0.0, 0.0])
                # MuJoCo quats are (w, x, y, z)
                quat = parse_array(geom.get('quat'), [1.0, 0.0, 0.0, 0.0])
                
                geoms.append({
                    "mesh_path": str(mesh_path),
                    "pos": pos,
                    "quat": quat
                })
        
        if geoms:
            body_geoms[body_name] = geoms
            
    return body_geoms

def main():
    parser = argparse.ArgumentParser(description="Visualize Unitree G1 Action Horizon in Rerun")
    parser.add_argument("--episode_idx", type=int, default=0, help="Episode index to visualize")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    args = parser.parse_args()

    # 1. Initialize Rerun
    rr.init("NanoVLA G1 Horizon")
    # Save the recording to a file so the user can open it with python3 rerun viewer
    rr.save("horizon.rrd")
    rr.set_time("log_time", duration=0.0)

    # 2. Initialize Genesis
    gs.init(backend=gs.cpu)
    scene = gs.Scene(show_viewer=False) # Headless mode since we use Rerun
    
    xml_path = 'data/unitree_g1/g1_mocap_29dof_with_hands.xml'
    robot = scene.add_entity(gs.morphs.MJCF(file=xml_path))
    
    # Add an ego camera to match the XML's `<camera name="ego_head">`
    cam = scene.add_camera(res=(320, 240), fov=90)
    
    scene.build()
    
    # Attach camera to torso_link with the same local pose as the XML
    from scipy.spatial.transform import Rotation as R
    r = R.from_euler('xyz', [0.25, 1.5708, 0])
    offset_T = np.eye(4)
    offset_T[:3, :3] = r.as_matrix()
    offset_T[:3, 3] = [0.08, 0.0, 0.38]
    cam.attach(robot.get_link('torso_link'), offset_T)
    
    # 3. Load Dataset
    datasets_root_dir = Path('data/datasets')
    dataset_repos = sorted([str(p) for p in datasets_root_dir.iterdir() if p.is_dir()])
    
    if not dataset_repos:
        print("No datasets found in data/datasets.")
        return

    print(f"Loading dataset from {dataset_repos[0]}...")
    try:
        dataset = load_dataset(dataset_repos[0])['train']
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Filter to requested episode
    episodes = dataset['episode_index']
    ep_start = None
    ep_end = None
    for i, ep in enumerate(episodes):
        if ep == args.episode_idx and ep_start is None:
            ep_start = i
        elif ep != args.episode_idx and ep_start is not None:
            ep_end = i
            break
    if ep_start is None:
        print(f"Episode {args.episode_idx} not found.")
        return
    if ep_end is None:
        ep_end = len(episodes)
        
    print(f"Visualizing full episode {args.episode_idx} from index {ep_start} to {ep_end-1} (length: {ep_end - ep_start})")
    
    actions = np.array(dataset['action'][ep_start:ep_end])
    g1_actions = map_horizon_to_g1(actions)

    # 4. Parse MJCF for Rerun visual logging
    body_geoms = parse_mjcf_visual_geoms(xml_path)
    
    # Log meshes to Rerun ONCE (static assets)
    rr.set_time("step", sequence=0)
    
    # Log the Pinhole camera geometry and offset statically!
    quat_xyzw = r.as_quat()
    rr.log("world/torso_link/ego_head", rr.Transform3D(
        translation=[0.08, 0.0, 0.38],
        rotation=rr.Quaternion(xyzw=quat_xyzw)
    ), static=True)
    
    rr.log("world/torso_link/ego_head", rr.Pinhole(
        resolution=[320, 240],
        focal_length=160.0
    ), static=True)
    
    for link_name, geoms in body_geoms.items():
        for i, geom_info in enumerate(geoms):
            entity_path = f"world/{link_name}/geom_{i}"
            # Log the mesh
            try:
                rr.log(entity_path, rr.Asset3D(path=geom_info["mesh_path"]))
                # If there's a local offset for the visual geometry relative to the link frame, log it as static
                # Rerun uses (x,y,z,w) for quaternions natively, but `rr.Quaternion` accepts `xyzw` or `wxyz`?
                # Rerun 0.33 rr.Quaternion(xyzw=...) 
                # MuJoCo is wxyz.
                w, x, y, z = geom_info["quat"]
                rr.log(entity_path, rr.Transform3D(
                    translation=geom_info["pos"],
                    rotation=rr.Quaternion(xyzw=[x, y, z, w])
                ), static=True)
            except Exception as e:
                print(f"Failed to log mesh {geom_info['mesh_path']} for {link_name}: {e}")

    print("Starting Rerun simulation loop...")
    
    # Step duration (30Hz dataset)
    dt = 1.0 / 30.0 
    
    for step_idx in range(len(g1_actions)):
        target_action = g1_actions[step_idx]
        
        # Apply the action to Genesis
        # Genesis expects all 49 DoFs
        robot.set_dofs_position(target_action)
        scene.step()
        
        # Update Rerun timeline
        rr.set_time("step", sequence=step_idx)
        rr.set_time("sim_time", duration=step_idx * dt)
        
        # Log link poses
        for link in robot.links:
            # Genesis returns tensors on CPU or GPU
            pos = link.get_pos().cpu().numpy()
            quat = link.get_quat().cpu().numpy() # Genesis quats are (w, x, y, z)
            
            w, x, y, z = quat
            
            # Log the transform for the link
            rr.log(f"world/{link.name}", rr.Transform3D(
                translation=pos,
                rotation=rr.Quaternion(xyzw=[x, y, z, w])
            ))
            
        # Render the ego camera view and log it
        cam_out = cam.render()
        rgb = cam_out[0] if isinstance(cam_out, tuple) else cam_out
        rr.log("world/torso_link/ego_head", rr.Image(rgb))
            
    print("Episode visualization saved to horizon.rrd.")
    print("Run `python3 -m rerun horizon.rrd` to view the animation.")

if __name__ == "__main__":
    main()
