import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset, concatenate_datasets, Value

# Joint names mapping for 0-27
JOINT_NAMES = {
    0: "L_Shoulder_Pitch",
    1: "L_Shoulder_Roll",
    2: "L_Shoulder_Yaw",
    3: "L_Elbow",
    4: "L_Wrist_Roll",
    5: "L_Wrist_Pitch",
    6: "L_Wrist_Yaw",
    7: "R_Shoulder_Pitch",
    8: "R_Shoulder_Roll",
    9: "R_Shoulder_Yaw",
    10: "R_Elbow",
    11: "R_Wrist_Roll",
    12: "R_Wrist_Pitch",
    13: "R_Wrist_Yaw",
    14: "L_Hand_Thumb_0",
    15: "L_Hand_Thumb_1",
    16: "L_Hand_Thumb_2",
    17: "L_Hand_Middle_0",
    18: "L_Hand_Middle_1",
    19: "L_Hand_Index_0",
    20: "L_Hand_Index_1",
    21: "R_Hand_Thumb_0",
    22: "R_Hand_Thumb_1",
    23: "R_Hand_Thumb_2",
    24: "R_Hand_Index_0",
    25: "R_Hand_Index_1",
    26: "R_Hand_Middle_0",
    27: "R_Hand_Middle_1",
    28: "Ignored_28",
    29: "Ignored_29",
}

EEF_NAMES = [
    "L_EEF_X", "L_EEF_Y", "L_EEF_Z", "L_EEF_Rx", "L_EEF_Ry", "L_EEF_Rz",
    "R_EEF_X", "R_EEF_Y", "R_EEF_Z", "R_EEF_Rx", "R_EEF_Ry", "R_EEF_Rz"
]

def main():
    # 1. Load dataset
    datasets_root_dir = Path('data/datasets')
    
    if not datasets_root_dir.exists():
        print(f"Error: Dataset directory {datasets_root_dir} not found.")
        return

    dataset_repos = sorted([str(p) for p in datasets_root_dir.iterdir() if p.is_dir()])
    
    if not dataset_repos:
        print("No datasets found in data/datasets.")
        return

    print("Loading datasets...")
    all_datasets = []
    for repo in dataset_repos:
        ds = load_dataset(repo)['train']
        if ds.features["timestamp"].dtype == "float32":
            ds = ds.cast_column("timestamp", Value("float64"))
        all_datasets.append(ds)

    combined_dataset = concatenate_datasets(all_datasets)
    n_samples = len(combined_dataset)
    print(f"Loaded {n_samples} samples.")

    # 2. Extract arrays
    print("Extracting actions and observations...")
    joint_actions = np.array(combined_dataset["action"])
    eef_actions = np.array(combined_dataset["eef_sim_pose_action"])
    
    joint_observations = np.array(combined_dataset["observation.state"])
    eef_observations = np.array(combined_dataset["eef_sim_pose_state"])

    # 3. Calculate standard deviations
    print("Calculating standard deviations...")
    std_joint_actions = np.std(joint_actions, axis=0)
    std_eef_actions = np.std(eef_actions, axis=0)
    
    std_joint_obs = np.std(joint_observations, axis=0)
    std_eef_obs = np.std(eef_observations, axis=0)

    # Use a modern style/palette
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 4. Plot 1: Joint States (Actions vs Observations)
    fig1, ax1 = plt.subplots(figsize=(15, 8))
    x_indices = np.arange(30)
    bar_width = 0.35

    rects1 = ax1.bar(x_indices - bar_width/2, std_joint_obs, bar_width, label='Observation State', color='#3498db', alpha=0.9)
    rects2 = ax1.bar(x_indices + bar_width/2, std_joint_actions, bar_width, label='Action State', color='#e74c3c', alpha=0.9)

    ax1.set_xlabel('Joint Names / Dimension Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Standard Deviation (rad / unit)', fontsize=12, fontweight='bold')
    ax1.set_title('Joint States: Standard Deviation Comparison (Observation vs Action)', fontsize=14, fontweight='bold', pad=15)
    
    joint_labels = [f"{i}: {JOINT_NAMES.get(i, f'Joint {i}')}" for i in range(30)]
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(joint_labels, rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot1_path = Path(__file__).resolve().parent / "joint_states_std.png"
    plt.savefig(plot1_path, dpi=200)
    print(f"Saved Joint States plot to {plot1_path}")
    plt.close()

    # 5. Plot 2: EEF States (Actions vs Observations)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    x_eef_indices = np.arange(12)
    
    rects1_eef = ax2.bar(x_eef_indices - bar_width/2, std_eef_obs, bar_width, label='Observation EEF State', color='#2ecc71', alpha=0.9)
    rects2_eef = ax2.bar(x_eef_indices + bar_width/2, std_eef_actions, bar_width, label='Action EEF State', color='#9b59b6', alpha=0.9)

    ax2.set_xlabel('End Effector Dimension', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Standard Deviation (unit)', fontsize=12, fontweight='bold')
    ax2.set_title('End Effector States: Standard Deviation Comparison (Observation vs Action)', fontsize=14, fontweight='bold', pad=15)
    
    eef_labels = [f"{i}: {EEF_NAMES[i]}" for i in range(12)]
    ax2.set_xticks(x_eef_indices)
    ax2.set_xticklabels(eef_labels, rotation=30, ha='right', fontsize=10)
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot2_path = Path(__file__).resolve().parent / "eef_states_std.png"
    plt.savefig(plot2_path, dpi=200)
    print(f"Saved EEF States plot to {plot2_path}")
    plt.close()

    # 6. Combined 2x1 Plot for Overview
    fig, (ax_joint, ax_eef) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Joint plot in combined
    ax_joint.bar(x_indices - bar_width/2, std_joint_obs, bar_width, label='Observation Joint State', color='#3498db', alpha=0.9)
    ax_joint.bar(x_indices + bar_width/2, std_joint_actions, bar_width, label='Action Joint State', color='#e74c3c', alpha=0.9)
    ax_joint.set_ylabel('Standard Deviation (rad)', fontsize=12, fontweight='bold')
    ax_joint.set_title('A. Joint States Standard Deviation', fontsize=14, fontweight='bold', loc='left')
    ax_joint.set_xticks(x_indices)
    ax_joint.set_xticklabels(joint_labels, rotation=45, ha='right', fontsize=9)
    ax_joint.legend(fontsize=11)
    ax_joint.grid(True, linestyle='--', alpha=0.5)

    # EEF plot in combined
    ax_eef.bar(x_eef_indices - bar_width/2, std_eef_obs, bar_width, label='Observation EEF State', color='#2ecc71', alpha=0.9)
    ax_eef.bar(x_eef_indices + bar_width/2, std_eef_actions, bar_width, label='Action EEF State', color='#9b59b6', alpha=0.9)
    ax_eef.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
    ax_eef.set_title('B. End-Effector Pose States Standard Deviation', fontsize=14, fontweight='bold', loc='left')
    ax_eef.set_xticks(x_eef_indices)
    ax_eef.set_xticklabels(eef_labels, rotation=30, ha='right', fontsize=10)
    ax_eef.legend(fontsize=11)
    ax_eef.grid(True, linestyle='--', alpha=0.5)

    fig.suptitle('Standard Deviation Statistics for Action and Observation States', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    combined_plot_path = Path(__file__).resolve().parent / "action_obs_std_comparison.png"
    plt.savefig(combined_plot_path, dpi=200)
    print(f"Saved Combined Comparison plot to {combined_plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
