import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from datasets import load_dataset
from tqdm.auto import tqdm
from PIL import Image
import concurrent.futures

from config import parse_args
from models.visual_encoder import SigLIP
from data.advanced_dataset import VideoDataset

def main():
    config = parse_args()
    
    precompute_path = Path(config.precompute_path)
    precompute_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading VLM Model...")
    # Initialize the SigLIP model; its __init__ automatically compiles the JIT functions.
    vlm = SigLIP(checkpoint_path=config.vlm_checkpoint_path, normalize=True)
    
    print("Loading datasets from datasets_root...")
    datasets_root_dir = Path(config.datasets_root)
    dataset_repos = sorted([p for p in datasets_root_dir.iterdir() if p.is_dir()])
    
    if not dataset_repos:
        print(f"No datasets found in {config.datasets_root}")
        return
        
    from datasets import concatenate_datasets
    all_datasets = []
    
    for repo_path in dataset_repos:
        repo_name = repo_path.name
        print(f"Loading dataset: {repo_name}")
        ds = load_dataset(str(repo_path))['train']
        ds = ds.add_column("dataset_name", [repo_name] * len(ds))
        all_datasets.append(ds)
        
    combined_hf_dataset = concatenate_datasets(all_datasets)
    print(f"\\nCombined dataset size: {len(combined_hf_dataset)} rows")
        
    video_dataset = VideoDataset(
        dataset=combined_hf_dataset,
        datasets_root=config.datasets_root,
        action_horizon=config.action_horizon,
        precompute_path=None
    )
    
    episodes = video_dataset.episode_ranges
    print(f"Precomputing for {len(episodes)} total episodes...")
    
    batch_size = config.batch_size
    
    for ep_id, (ep_start, ep_end) in tqdm(episodes.items(), desc="All Episodes"):
        repo_name = combined_hf_dataset[ep_start]["dataset_name"]
        
        save_dir = precompute_path / repo_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / f"ep_{ep_id}.npz"
        if save_path.exists():
            continue
            
        episode_vlm_outs = []
        batch_images = []
        batch_input_ids = []
        
        def process_batch(imgs, ids):
            # Extract the correct camera image (PIL Image)
            pil_images = []
            for img_dict in imgs:
                if 'default_cam' in img_dict:
                    pil_images.append(img_dict['default_cam'])
                else:
                    pil_images.append(list(img_dict.values())[0])
            
            # ids is already tokenized ints from the VideoDataset
            id_tensor = np.stack(ids)
            
            # Encode images and texts using the VLM
            img_hidden = vlm.encode_images(pil_images)
            txt_hidden = vlm.encode_texts(id_tensor)
            
            # Concat text and image hidden states along sequence dimension
            vlm_out = np.concatenate([txt_hidden, img_hidden], axis=1)
            return vlm_out
            
        for i in range(ep_start, ep_end + 1):
            data = video_dataset[i]
            batch_images.append(data["images"])
            batch_input_ids.append(data["input_ids"])
            
            if len(batch_images) == batch_size:
                vlm_out = process_batch(batch_images, batch_input_ids)
                episode_vlm_outs.append(vlm_out)
                batch_images = []
                batch_input_ids = []
                
        if len(batch_images) > 0:
            vlm_out = process_batch(batch_images, batch_input_ids)
            episode_vlm_outs.append(vlm_out)
            
        episode_vlm_outs = np.concatenate(episode_vlm_outs, axis=0)
        
        # Save compressed to save disk space
        np.savez_compressed(save_path, vlm_out=episode_vlm_outs)
        
if __name__ == "__main__":
    main()
