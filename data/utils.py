from pathlib import Path
import cv2
from tqdm.auto import tqdm
import numpy as np


def extract_frames(video_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_id = 0

    for video in tqdm(sorted(Path(video_dir).rglob("*.mp4"))):
        cap = cv2.VideoCapture(str(video))

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            cv2.imwrite(
                str(out_dir / f"{frame_id:06d}.jpg"),
                frame
            )

            frame_id += 1

        cap.release()

    print(f"Saved {frame_id} frames")

def build_episode_lookup(dataset):
    """
    Returns:
        {
            episode_id: (start_row, end_row)
        }
    """
    episodes = dataset["episode_index"]

    lookup = {}

    for row_idx, ep in enumerate(episodes):

        if ep not in lookup:
            lookup[ep] = [row_idx, row_idx]
        else:
            lookup[ep][1] = row_idx

    return {
        ep: (start, end)
        for ep, (start, end) in lookup.items()
    }

SUBTASK_MAP = {
    0: "End",
    1: "Grasp the lemon and lift it to the center of the view with left gripper",
    2: "Abnormal",
    3: "Grasp the lemon and lift it to the center of the view with right gripper",
    4: "Unknown",
}


def decode_subtask(subtask_annotation):
    """
    subtask_annotation: list[int]
    Example: [0, 1, 0, 0, 0]

    Returns:
        str
    """
    if subtask_annotation is None:
        return "Unknown"

    if isinstance(subtask_annotation, (list, tuple)):
        if len(subtask_annotation) == 0:
            return "Unknown"

        # one-hot encoding
        idx = int(np.argmax(subtask_annotation))

    else:
        idx = int(subtask_annotation)

    return SUBTASK_MAP.get(idx, "Unknown")


def build_instruction(subtask_annotation):
    subtask = decode_subtask(subtask_annotation)

    if subtask == "End":
        return "Episode completed."

    if subtask == "Abnormal":
        return "Recover from abnormal robot state."

    return subtask

if __name__ == "__main__":
    extract_frames("data/dataset/robocoin_lemon/videos", "data/dataset/robocoin_lemon/frames")

