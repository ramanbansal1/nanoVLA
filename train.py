import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import torch
import torch.utils.dlpack
from datasets import load_dataset
from torch.utils.data import DataLoader

from data.advanced_dataset import VideoDataset

def main():
    from config import parse_args, asdict
    config = parse_args()
    
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = config.jax_mem_fraction
    
    import wandb
    wandb.init(project=config.wandb_project, config=asdict(config))
    # 1. Load the underlying huggingface dataset
    
    
    from datasets import load_dataset, concatenate_datasets, Value

    all_datasets = []
    from pathlib import Path
    datasets_root_dir = Path('data/datasets')
    DATASETS = sorted([str(p) for p in datasets_root_dir.iterdir() if p.is_dir()])

    for repo in DATASETS:

        ds = load_dataset(repo)['train']

        if ds.features["timestamp"].dtype == "float32":
            ds = ds.cast_column(
                "timestamp",
                Value("float64")
            )

        ds = ds.add_column(
            "dataset_name",
            [repo.split("/")[-1]] * len(ds)
        )

        all_datasets.append(ds)

    combined_dataset = concatenate_datasets(
        all_datasets
    )

    # 3. Instantiate the VideoDataset
    train_dataset = VideoDataset(
        dataset=combined_dataset,
        datasets_root=config.datasets_root,
        action_horizon=config.action_horizon,
    )

    print(f"Loaded VideoDataset with {len(train_dataset)} items.")

    from torchvision.transforms import ToTensor
    from torch.utils.data._utils.collate import default_collate

    from torchvision.transforms import ToTensor
    from torch.utils.data._utils.collate import default_collate

    def custom_collate_fn(batch):
        to_tensor = ToTensor()
        collated = {}

        for key in batch[0].keys():
            if key == "images":
                imgs = []
                for item in batch:
                    first_cam = sorted(
                        item["images"].keys()
                    )[0]

                    imgs.append(
                        to_tensor(
                            item["images"][first_cam]
                        )
                    )
                collated["image"] = default_collate(imgs)
            elif key == "instruction":
                collated[key] = [
                    item[key]
                    for item in batch
                ]

            elif key != "images":
                collated[key] = default_collate(
                    [item[key] for item in batch]
                )

        return collated
    # 4. Create a DataLoader (example)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn
    )

    # 5. Instantiate VLA
    from models.vla import VLA
    from flax import nnx
    import jax.numpy as jnp
    import jax

    sample_item = train_dataset.dataset[0]
    obs_dim = len(sample_item["observation.state"]) + len(sample_item["eef_sim_pose_state"])
    rngs = nnx.Rngs(42)
    vla = VLA(
        hidden_size=config.hidden_size, 
        obs_dim=obs_dim, 
        rngs=rngs, 
        dummy=config.dummy_vlm,
        dit_num_blocks=config.dit_num_blocks,
        vla_k=config.vla_k
    )

    import optax
    
    # Initialize Optimizer
    optimizer = nnx.Optimizer(vla, optax.nadam(learning_rate=config.learning_rate), wrt=nnx.Param)
    
    # Define the loss function for a single batch
    def loss_fn(model, vlm_out, observation, token_ids, clean_mask, t, noise_key):
        clean_emb = model.action_tokenizer.action_emb(token_ids)
        
        noise = jax.random.normal(noise_key, clean_emb.shape)
        t_exp = t.reshape(-1, 1, 1) if clean_emb.ndim == 3 else t.reshape(-1, 1)
        x_t = (1 - t_exp) * noise + t_exp * clean_emb
        
        vlm_modulated, action_emb, action_mask, obs_emb, dit_out, latent, decoded_actions = model(
            images=None, 
            instruction=None, 
            observation=observation, 
            action=None,
            action_emb=x_t,
            action_mask=clean_mask,
            t=t,
            vlm_out=vlm_out
        )
        
        velocity_target = clean_emb - noise
        predicted_velocity = latent - x_t
            
        loss = (predicted_velocity - velocity_target) ** 2
        # Mask out padded tokens
        loss = loss * clean_mask[:, :, None]
        
        # Average loss only over valid (unmasked) tokens
        loss_val = jnp.sum(loss) / (jnp.sum(clean_mask) * loss.shape[-1])
        return loss_val, (latent, x_t, decoded_actions, clean_emb, clean_mask)

    @nnx.jit
    def train_step(model, optimizer, vlm_out, observation, token_ids, clean_mask, t, noise_key):
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss_val, aux), grads = grad_fn(model, vlm_out, observation, token_ids, clean_mask, t, noise_key)
        optimizer.update(model, grads)
        return loss_val, aux, grads

    import orbax.checkpoint as ocp
    
    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    checkpointers = {
        "model_state": ocp.PyTreeCheckpointer(),
        "optimizer_state": ocp.PyTreeCheckpointer()
    }
    checkpoint_manager = ocp.CheckpointManager(
        os.path.abspath(config.checkpoint_dir),
        checkpointers=checkpointers,
        options=options
    )

    from jax.sharding import Mesh, PartitionSpec, NamedSharding
    import numpy as np

    # 1. Define a device mesh across all GPUs
    devices = np.array(jax.devices())
    mesh = Mesh(devices, axis_names=("data",))

    # 2. Shard the batch dimension across GPUs
    batch_sharding = NamedSharding(mesh, PartitionSpec("data"))

    from tqdm.auto import tqdm

    global_step = 0
    for epoch in range(config.num_epochs):
        # Example iterating through batches
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for step, batch in enumerate(train_pbar):
        
            # Convert torch inputs to jnp.ndarray safely
            def torch_to_jax(t):
                try:
                    t_cuda = t.cuda(non_blocking=True).contiguous()
                    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(t_cuda))
                except Exception:
                    return jnp.array(t.contiguous().numpy())

            images_jnp = torch_to_jax(batch['image'])
            instruction = batch['instruction'][0]  # Take first instruction since batch_size=1
            
            observation_jnp = torch_to_jax(batch['observation_state'])
            eef_state_jnp = torch_to_jax(batch['eef_state'])
            observation_jnp = jnp.concatenate([observation_jnp, eef_state_jnp], axis=-1)
            
            action_jnp = torch_to_jax(batch['action'])
            eef_action_jnp = torch_to_jax(batch['eef_action'])
            action_jnp = jnp.concatenate([action_jnp, eef_action_jnp], axis=-1)
            
            # Generate timestep and noise key for Flow Matching
            key = jax.random.PRNGKey(global_step)
            key, noise_key, t_key = jax.random.split(key, 3)
            
            t = jax.random.uniform(t_key, shape=(action_jnp.shape[0],))
            
            # Precompute VLM output outside JIT
            vlm_out = vla.vlm(images_jnp, instruction)
            
            # Tokenize actions outside JIT to avoid Tracer issues with Scipy
            import numpy as np
            action_np = np.array(action_jnp)
            token_ids, clean_mask = vla.action_tokenizer.tokenize(action_np)
            
            # 3. In your train loop, shard each batch before the step
            with mesh:
                vlm_out = jax.device_put(vlm_out, batch_sharding)
                observation_jnp = jax.device_put(observation_jnp, batch_sharding)
                token_ids = jax.device_put(token_ids, batch_sharding)
                clean_mask = jax.device_put(clean_mask, batch_sharding)
                t = jax.device_put(t, batch_sharding)
                
                # Compute loss and gradients using JIT-compiled train_step
                loss_val, aux, grads = train_step(vla, optimizer, vlm_out, observation_jnp, token_ids, clean_mask, t, noise_key)
            latent, noisy_emb, pred_decoded, clean_emb, clean_mask = aux
            
            # Wait for async dispatch to finish before logging
            loss_val = jax.block_until_ready(loss_val)
            
            # Calculate aux metrics
            grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)]))
            latent_drift = jnp.mean(jnp.abs(latent - noisy_emb))
            if pred_decoded is not None:
                actual_decoded = vla.action_tokenizer.decode(clean_emb, mask=clean_mask)
            else:
                actual_decoded = None
            
            import numpy as np
            mses = []
            if pred_decoded is not None:
                for p, a in zip(pred_decoded, actual_decoded):
                    if p.shape == a.shape and p.size > 0:
                        mses.append(np.mean((p - a)**2))
            
            if len(mses) > 0:
                decoded_mse = float(np.mean(mses))
            else:
                decoded_mse = 0.0

            log_dict = {
                "train/loss": float(loss_val),
                "train/grad_norm": float(grad_norm),
                "train/latent_drift": float(latent_drift),
                "eval/decoded_mse": decoded_mse
            }
            
            log_dict["epoch"] = epoch + 1
            wandb.log(log_dict, step=global_step)
            train_pbar.set_postfix({
                "loss": f"{float(loss_val):.4f}",
                "grad": f"{float(grad_norm):.3f}",
                "drift": f"{float(latent_drift):.3f}",
                "mse": f"{decoded_mse:.4f}"
            })
            
            # Save checkpoint periodically
            if global_step % config.save_every == 0 and global_step > 0:
                _, model_state = nnx.split(vla)
                _, opt_state = nnx.split(optimizer)
                checkpoint_manager.save(
                    global_step, 
                    args=ocp.args.Composite(
                        model_state=ocp.args.PyTreeSave(model_state),
                        optimizer_state=ocp.args.PyTreeSave(opt_state)
                    )
                )
                checkpoint_manager.wait_until_finished()
                
            global_step += 1

if __name__ == "__main__":
    main()
