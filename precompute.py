import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time
from pathlib import Path
import jax
import jax.numpy as jnp
from config import TrainConfig
from train import setup_dataloader
import jax.sharding
import wandb
import concurrent.futures
from tqdm.auto import tqdm
from rich.console import Console

console = Console()

def save_batch(ds_names, ep_ids, frame_indices, img_hidden_np, txt_hidden_np, out_dir):
    save_t0 = time.time()
    for b in range(len(ds_names)):
        d_name = ds_names[b]
        e_id = ep_ids[b].item() if hasattr(ep_ids[b], "item") else ep_ids[b]
        f_idx = frame_indices[b].item() if hasattr(frame_indices[b], "item") else frame_indices[b]
        
        ep_dir = out_dir / d_name / f"ep_{e_id}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = ep_dir / f"{f_idx:06d}.npz"
        import numpy as np
        np.savez_compressed(save_path, img_hidden=img_hidden_np[b], txt_hidden=txt_hidden_np[b])
    return time.time() - save_t0

def main():
    wandb.init(project="nanovla-temp", name="siglip-eval-test")
    config = TrainConfig()
    print(f"JAX Default Backend: {jax.default_backend()}")
    print(f"JAX Devices available: {jax.devices()}")

    config.datasets_root = '../datasets'
    config.vlm_checkpoint_path = './siglip2_b16_naflex.npz'
    
    # Save the target path before we disable it for the dataloader
    target_precompute_path = config.precompute_path
    
    # Disable precomputed npz loading to force online image loading
    config.precompute_path = None
    
    console.print("[bold cyan]Setting up dataloader...[/bold cyan]")
    
    console.print("[bold cyan]Loading full dataset for precomputation...[/bold cyan]")
    from train import prepare_datasets, custom_collate_fn
    from data.advanced_dataset import VideoDataset
    from torch.utils.data import DataLoader
    from datasets import concatenate_datasets
    
    # Load all splits and concatenate them into one massive dataset
    train_ds, val_ds, test_ds = prepare_datasets(config.datasets_root)
    all_ds = [ds for ds in [train_ds, val_ds, test_ds] if ds is not None]
    full_hf_dataset = concatenate_datasets(all_ds)
    
    full_video_dataset = VideoDataset(
        dataset=full_hf_dataset,
        datasets_root=config.datasets_root,
        action_horizon=config.action_horizon,
        precompute_path=None,
    )
    
    seq_loader = DataLoader(
        full_video_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn,
        drop_last=False
    )

    console.print("[bold cyan]Loading SigLIP model...[/bold cyan]")
    from models.visual_encoder import SigLIP
    vlm = SigLIP(config.vlm_checkpoint_path)
    
    # [NEW] Create Mesh for explicit multi-GPU sharding
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    mesh = Mesh(jax.devices(), axis_names=('dp',))
    dp_sharding = NamedSharding(mesh, P('dp'))  # Shard across batch dimension
    
    if len(seq_loader) == 0:
        console.print("[bold red]No training data found.[/bold red]")
        return
        
    console.print(f"[bold green]Total batches:[/bold green] {len(seq_loader)}")
    
    # Target directory for saved representations
    out_dir = Path(target_precompute_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    t0 = time.time()
    pbar = tqdm(seq_loader, desc="Precomputing VLM", total=len(seq_loader))
    
    for i, batch in enumerate(pbar):
        # Retrieve the dataset name and episode ID
        ds_names = batch.get("dataset_name", ["Unknown"])
        ep_ids = batch.get("episode_id", [-1])
        frame_indices = batch.get("frame_index", [-1])
        
        # [NEW] Skip this batch if ALL frames in it are already precomputed
        all_exist = True
        for b in range(len(ds_names)):
            d_name = ds_names[b]
            e_id = ep_ids[b].item() if hasattr(ep_ids[b], "item") else ep_ids[b]
            f_idx = frame_indices[b].item() if hasattr(frame_indices[b], "item") else frame_indices[b]
            if not (out_dir / d_name / f"ep_{e_id}" / f"{f_idx:06d}.npz").exists():
                all_exist = False
                break
                
        if all_exist:
            pbar.update(1)
            continue
        
        # Take the first item in the batch for logging
        ds_name = ds_names[0]
        ep_id = ep_ids[0].item() if hasattr(ep_ids[0], "item") else ep_ids[0]
        
        elapsed = time.time() - t0
        
        pbar.set_postfix({"ds": ds_name, "ep": ep_id})
              
        import numpy as np
        
        # Online compute
        images = batch["image"].numpy()  # (B, H, W, C)
        input_ids = batch["input_ids"].numpy()
        
        # [NEW] Pad batch to be evenly divisible by number of GPUs
        n_devices = len(jax.devices())
        B_real = images.shape[0]
        pad_n = (n_devices - (B_real % n_devices)) % n_devices
        if pad_n > 0:
            images = np.concatenate([images, np.repeat(images[-1:], pad_n, axis=0)], axis=0)
            input_ids = np.concatenate([input_ids, np.repeat(input_ids[-1:], pad_n, axis=0)], axis=0)
        
        t_siglip = time.time()
        patches, ptype, yabs, xabs = vlm.images_to_naflex(images)
        
        # [NEW] Distribute (shard) inputs explicitly across the mesh
        patches_sharded = jax.device_put(jnp.asarray(patches), dp_sharding)
        ptype_sharded = jax.device_put(jnp.asarray(ptype), dp_sharding)
        yabs_sharded = jax.device_put(jnp.asarray(yabs), dp_sharding)
        xabs_sharded = jax.device_put(jnp.asarray(xabs), dp_sharding)
        
        input_ids_sharded = jax.device_put(jnp.asarray(input_ids), dp_sharding)
        
        # JAX's GSPMD automatically detects the sharded arrays and parallelizes the computation!
        
        with jax.sharding.set_mesh(mesh):
            img_hidden = vlm._encode_image_jit(
                patches_sharded, ptype_sharded, yabs_sharded, xabs_sharded
            )
            txt_hidden = vlm._encode_text_jit(input_ids_sharded)
        
        img_hidden.block_until_ready()
        txt_hidden.block_until_ready()
        
        siglip_time = time.time() - t_siglip
        
        # Transfer back to CPU as numpy arrays to save to disk
        # [NEW] Slice off the padding dummy elements using B_real
        img_hidden_np = np.array(img_hidden)[:B_real]
        txt_hidden_np = np.array(txt_hidden)[:B_real]
        
        # Prevent VRAM Leak: Manually free the JAX DeviceArrays immediately
        img_hidden.delete()
        txt_hidden.delete()
        
        # Fire and forget the save task to the background thread pool
        future = save_executor.submit(
            save_batch, ds_names, ep_ids, frame_indices, img_hidden_np, txt_hidden_np, out_dir
        )
        
        # Callback to log to WandB only when the background save actually finishes
        def _on_save_done(fut, b_idx, d_load, s_eval):
            try:
                s_time = fut.result()
                wandb.log({
                    "data_load_time": d_load,
                    "siglip_eval_time": s_eval,
                    "disk_save_time": s_time,
                    "total_batch_time": d_load + s_eval + s_time,
                    "batch_idx": b_idx
                })
            except Exception as e:
                print(f"Save failed for batch {b_idx}: {e}")

        future.add_done_callback(
            lambda f, idx=i, el=elapsed, st=siglip_time: _on_save_done(f, idx, el, st)
        )
              
        t0 = time.time()

    console.print("\n[bold yellow]Waiting for background saves to finish...[/bold yellow]")
    save_executor.shutdown(wait=True)
    console.print("[bold green]Precomputation Complete![/bold green]")
    wandb.finish()

if __name__ == "__main__":
    main()