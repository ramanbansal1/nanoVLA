import threading
import queue
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import cv2
from tqdm.auto import tqdm
import torch 
from data.utils import build_episode_lookup, build_instruction



from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from transformers import AutoTokenizer



class VideoDataset(Dataset):
    """
    Multi-dataset + multi-camera RoboCOIN dataset.

    Expected folder structure:

    datasets/
    ├── G1edu-u3_pick_apple_a/
    │   └── frames/
    │       ├── cam_left_high/ 
    │       ├── cam_left_wrist/
    │       └── cam_right_wrist/
    ├── G1edu-u3_place_apple_c/
    │   └── frames/
    │       ├── cam_left_high/
    │       ├── cam_left_wrist/
    │       └── cam_right_wrist/
    """

    def __init__(
        self,
        dataset,
        datasets_root,
        action_horizon,
    ):
        self.dataset = dataset
        self.datasets_root = Path(datasets_root)
        self.action_horizon = action_horizon

        self.state_dim = len(
            dataset[0]["observation.state"]
        )

        self.action_dim = len(
            dataset[0]["action"]
        )

        self.episode_ranges = (
            build_episode_lookup(dataset)
        )

        self.max_episode_len = max(
            end - start
            for start, end
            in self.episode_ranges.values()
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/siglip2-base-patch16-naflex"
        )

    def __len__(self):
        return len(self.dataset)

    def _load_images(self, row):

        dataset_name = row["dataset_name"]

        frames_root = (
            self.datasets_root
            / dataset_name
            / "frames"
        )

        frame_name = (
            f"{row['frame_index']:06d}.jpg"
        )

        images = {}

        if not frames_root.exists():
            raise FileNotFoundError(
                f"Missing: {frames_root}"
            )

        has_cam_dirs = any(p.is_dir() for p in frames_root.iterdir())
        
        if not has_cam_dirs:
            image_path = frames_root / frame_name
            if image_path.exists():
                images['default_cam'] = Image.open(image_path).convert("RGB")
        else:
            for cam_dir in sorted(
                frames_root.iterdir()
            ):
    
                if not cam_dir.is_dir():
                    continue
    
                image_path = (
                    cam_dir / frame_name
                )
    
                if image_path.exists():
                    images[cam_dir.name] = (
                        Image.open(image_path)
                        .convert("RGB")
                    )

        if len(images) == 0:
            raise FileNotFoundError(
                f"No images found for "
                f"{frame_name} in {frames_root}"
            )

        return images

    def __getitem__(self, idx):
        row = self.dataset[idx]

        ep_id = row["episode_index"]

        ep_start, ep_end = (
            self.episode_ranges[ep_id]
        )

        actions = []
        for k in range(self.action_horizon):
            target_idx = min(
                idx + k,
                ep_end,
            )
            target_row = self.dataset[target_idx]
            actions.append(target_row["action"])

        instruction = build_instruction(row["subtask_annotation"])
        
        data = {
            "instruction": instruction,
            "input_ids": self.tokenizer(
                instruction,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="np",
            )["input_ids"][0].astype(np.int32),

            "observation_state":
                torch.tensor(
                    row["observation.state"],
                    dtype=torch.float32,
                ),

            "action":
                torch.tensor(
                    actions,
                    dtype=torch.float32,
                ),

            "dataset_name":
                row["dataset_name"],

            "episode_id":
                ep_id,

            "frame_index":
                row["frame_index"],
        }
        
        data["images"] = self._load_images(row)
            
        return data
        """

class VideoDataset(Dataset):
    def __init__(self, dataset, video_root, action_horizon):
        ""
        Store configuration and dataset metadata.
        ""
        self.dataset = dataset
        self.video_root = video_root
        #self.norm_stats = norm_stats
        #self.chunk_size = chunk_size
        #self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
    
        self.state_dim = len(dataset[0]["observation.state"])
        self.action_dim = len(dataset[0]["action"])
    
        self.episode_ranges = build_episode_lookup(dataset)
    
        self.max_episode_len = max(
            end - start
            for start, end in self.episode_ranges.values()
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
    
        ep_id = row["episode_index"]
        ep_start, ep_end = self.episode_ranges[ep_id]
    
        # future target index
        target_idx = idx + self.action_horizon
    
        # do not cross episode boundary
        if target_idx > ep_end:
            target_idx = ep_end
    
        target_row = self.dataset[target_idx]
    
        # image path (adjust extension if needed)
        image_path = (
            Path(self.video_root)
            / f"{row['frame_index']:06d}.jpg"
        )
    
        image = Image.open(image_path).convert("RGB")
        instruction = build_instruction(row["subtask_annotation"])
        
        return {
            "image": image,
            "instruction": instruction,
            "observation_state": torch.tensor(
                row["observation.state"],
                dtype=torch.float32,
            ),
            "eef_state": torch.tensor(
                row["eef_sim_pose_state"],
                dtype=torch.float32,
            ),
    
            # future targets
            "action": torch.tensor(
                target_row["action"],
                dtype=torch.float32,
            ),
            "eef_action": torch.tensor(
                target_row["eef_sim_pose_action"],
                dtype=torch.float32,
            ),
    
            "episode_id": ep_id,
            "frame_index": row["frame_index"],
        }


class EpisodeIterableDataset(VideoDataset, IterableDataset):

    def __init__(self, dataset, video_root, action_horizon, shuffle=False):
        self.dataset = dataset
        self.video_root = Path(video_root)
        self.action_horizon = action_horizon
        self.shuffle = shuffle

        self.episode_ranges = build_episode_lookup(dataset)

    def __iter__(self):
        worker_info = get_worker_info()
        episode_ids = list(self.episode_ranges.keys())

        if self.shuffle:
            import random
            random.shuffle(episode_ids)

        if worker_info is not None:
            episode_ids = episode_ids[
                worker_info.id::worker_info.num_workers
            ]

        for ep_id in episode_ids:

            ep_start, ep_end = self.episode_ranges[ep_id]

            for idx in range(ep_start, ep_end + 1):

                row = self.dataset[idx]

                image_path = (
                    self.video_root /
                    f"{row['frame_index']:06d}.jpg"
                )

                image = Image.open(image_path).convert("RGB")

                actions = []
                eef_actions = []

                # action chunk
                for k in range(self.action_horizon):

                    future_idx = min(idx + k, ep_end)
                    future_row = self.dataset[future_idx]

                    actions.append(
                        future_row["action"]
                    )

                    eef_actions.append(
                        future_row["eef_sim_pose_action"]
                    )

                yield {
                    "image": image,
                    "instruction": build_instruction(
                        row["subtask_annotation"]
                    ),

                    "observation_state": torch.tensor(
                        row["observation.state"],
                        dtype=torch.float32,
                    ),

                    "eef_state": torch.tensor(
                        row["eef_sim_pose_state"],
                        dtype=torch.float32,
                    ),

                    # diffusion targets
                    "actions": torch.tensor(
                        actions,
                        dtype=torch.float32,
                    ),  # [H, 30]

                    "eef_actions": torch.tensor(
                        eef_actions,
                        dtype=torch.float32,
                    ),  # [H, 12]

                    "episode_id": ep_id,
                    "frame_index": row["frame_index"],
                }"""