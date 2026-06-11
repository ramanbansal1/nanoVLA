import sys
from pathlib import Path
import numpy as np

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from vis.dataset_mapper import DATASET_TO_G1_MAPPING, get_standing_pose, map_action_to_g1

def test_mapping_logic():
    print("Running mapping sanity checks...")
    
    # 1. Check size of standing pose
    pose = get_standing_pose()
    assert pose.shape == (43,), f"Expected default pose size 43, got {pose.shape}"
    
    # 2. Check total mapped items
    assert len(DATASET_TO_G1_MAPPING) == 28, f"Expected 28 items in mapping, got {len(DATASET_TO_G1_MAPPING)}"
    
    # 3. Check for duplicates in source and target indices
    src_indices = [item[0] for item in DATASET_TO_G1_MAPPING]
    tgt_indices = [item[1] for item in DATASET_TO_G1_MAPPING]
    
    assert len(set(src_indices)) == 28, "Duplicate source dataset indices found!"
    assert len(set(tgt_indices)) == 28, "Duplicate target actuator indices found!"
    
    # 4. Check that unused target indices match expectations
    used_targets = set(tgt_indices)
    expected_unused_legs = set(range(0, 15))
    actual_unused = set(range(43)) - used_targets
    
    assert expected_unused_legs.issubset(actual_unused), "Legs/waist actuators (0-14) must not be mapped!"
    
    # 5. Check left hand target mapping order (thumb, middle, index)
    left_hand_targets = [item[1] for item in DATASET_TO_G1_MAPPING if 14 <= item[0] <= 20]
    expected_left_hand_targets = [22, 23, 24, 25, 26, 27, 28] # Thumb 0-2, Middle 0-1, Index 0-1
    assert left_hand_targets == expected_left_hand_targets, f"Left hand target mismatch: {left_hand_targets}"
    
    # 6. Check right hand target mapping order (thumb, index, middle)
    right_hand_targets = [item[1] for item in DATASET_TO_G1_MAPPING if 21 <= item[0] <= 27]
    expected_right_hand_targets = [36, 37, 38, 39, 40, 41, 42] # Thumb 0-2, Index 0-1, Middle 0-1
    assert right_hand_targets == expected_right_hand_targets, f"Right hand target mismatch: {right_hand_targets}"
    
    # 7. Check actual transformation
    dummy_action = np.arange(30, dtype=np.float32)
    mapped_action = map_action_to_g1(dummy_action)
    
    assert mapped_action.shape == (43,), "Mapped action shape must be 43"
    assert np.all(mapped_action[0:15] == 0.0), "First 15 actuators must be 0 (default pose)"
    
    # Specifically test left wrist pitch (src 5 -> tgt 20)
    assert mapped_action[20] == dummy_action[5], "Left wrist pitch mapping failed"
    
    print("All sanity checks passed successfully! ✅")

if __name__ == "__main__":
    test_mapping_logic()
