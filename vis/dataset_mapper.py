import numpy as np

# A strict 1-to-1 correspondence mapping (dataset_index, genesis_dof_index, joint_name)
# Note: Genesis uses kinematic tree ordering, so dofs 0-5 are the floating base.
DATASET_TO_G1_MAPPING = [
    # Left Arm
    (0, 17, "left_shoulder_pitch_joint"),
    (1, 21, "left_shoulder_roll_joint"),
    (2, 25, "left_shoulder_yaw_joint"),
    (3, 27, "left_elbow_joint"),
    (4, 29, "left_wrist_roll_joint"),
    (5, 31, "left_wrist_pitch_joint"),
    (6, 33, "left_wrist_yaw_joint"),

    # Right Arm
    (7, 18, "right_shoulder_pitch_joint"),
    (8, 22, "right_shoulder_roll_joint"),
    (9, 26, "right_shoulder_yaw_joint"),
    (10, 28, "right_elbow_joint"),
    (11, 30, "right_wrist_roll_joint"),
    (12, 32, "right_wrist_pitch_joint"),
    (13, 34, "right_wrist_yaw_joint"),

    # Left Hand (Thumb, Middle, Index)
    (14, 35, "left_hand_thumb_0_joint"),
    (15, 41, "left_hand_thumb_1_joint"),
    (16, 47, "left_hand_thumb_2_joint"),
    (17, 36, "left_hand_middle_0_joint"),
    (18, 42, "left_hand_middle_1_joint"),
    (19, 37, "left_hand_index_0_joint"),
    (20, 43, "left_hand_index_1_joint"),

    # Right Hand (Thumb, Index, Middle)
    (21, 38, "right_hand_thumb_0_joint"),
    (22, 44, "right_hand_thumb_1_joint"),
    (23, 48, "right_hand_thumb_2_joint"),
    (24, 40, "right_hand_index_0_joint"),
    (25, 46, "right_hand_index_1_joint"),
    (26, 39, "right_hand_middle_0_joint"),
    (27, 45, "right_hand_middle_1_joint"),
]

def get_standing_pose():
    """
    Returns a default standing pose for all 49 DoFs.
    DoFs 0-5 are the floating base. Height (z) is at index 2.
    """
    pose = np.zeros(49, dtype=np.float32)
    pose[2] = 0.79 # Set Z height so it doesn't sink into floor
    return pose

def map_action_to_g1(dataset_action):
    """
    Maps a dataset action vector (shape [30]) to the G1 DoF space (shape [49]).
    DoFs not targeted by dataset are held in the default standing pose.
    Dimensions 28 and 29 of the dataset are ignored.
    """
    # Ensure input is array-like and we handle its components correctly
    g1_action = get_standing_pose()
    
    for src_idx, tgt_idx, name in DATASET_TO_G1_MAPPING:
        g1_action[tgt_idx] = dataset_action[src_idx]
        
    return g1_action

def map_horizon_to_g1(dataset_actions):
    """
    Vectorized or loop-based mapping for an entire horizon [H, 30].
    Returns [H, 49].
    """
    H = dataset_actions.shape[0]
    g1_actions = np.zeros((H, 49), dtype=np.float32)
    
    for i in range(H):
        g1_actions[i] = map_action_to_g1(dataset_actions[i])
        
    return g1_actions
