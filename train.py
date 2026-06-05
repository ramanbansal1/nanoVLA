import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from data.advanced_dataset import VideoDataset

def main():
    import wandb
    wandb.init(project="nanoVLA")
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

    # 2. Setup paths and parameters
    datasets_root = "data/datasets"
    action_horizon = 8

    # 3. Instantiate the VideoDataset
    
    train_dataset = VideoDataset(
        dataset=combined_dataset,
        datasets_root=datasets_root,
        action_horizon=action_horizon,
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
        batch_size=1, 
        shuffle=True, 
        pin_memory=True,
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
    vla = VLA(hidden_size=192, obs_dim=obs_dim, rngs=rngs, dummy=False)

    import optax
    
    # Initialize Optimizer
    optimizer = nnx.Optimizer(vla, optax.nadam(learning_rate=1e-4), wrt=nnx.Param)
    
    # Define the loss function for a single batch
    def loss_fn(model, images, instruction, observation, action, t, clean_action):
        vlm_modulated, action_emb, action_mask, obs_emb, dit_out, latent, decoded_actions = model(
            images=images, 
            instruction=instruction, 
            observation=observation, 
            action=action,
            t=t
        )
        
        clean_emb, clean_mask = model.action_tokenizer(clean_action)
        noisy_emb = action_emb
        
        target_len = noisy_emb.shape[1]
        current_len = clean_emb.shape[1]
        
        if current_len > target_len:
            clean_emb = clean_emb[:, :target_len, :]
            clean_mask = clean_mask[:, :target_len]
        elif current_len < target_len:
            pad_width = ((0, 0), (0, target_len - current_len), (0, 0))
            clean_emb = jnp.pad(clean_emb, pad_width)
            
            mask_pad_width = ((0, 0), (0, target_len - current_len))
            clean_mask = jnp.pad(clean_mask, mask_pad_width, constant_values=False)
            
        velocity_target = clean_emb - noisy_emb
        predicted_velocity = latent - noisy_emb
            
        loss = (predicted_velocity - velocity_target) ** 2
        # Mask out padded tokens
        loss = loss * action_mask[:, :, None]
        
        # Average loss only over valid (unmasked) tokens
        loss_val = jnp.sum(loss) / (jnp.sum(action_mask) * loss.shape[-1])
        return loss_val, (latent, noisy_emb, decoded_actions, clean_emb, clean_mask)

    from tqdm.auto import tqdm

    # Example iterating through batches
    train_pbar = tqdm(train_loader, desc="Training")
    for step, batch in enumerate(train_pbar):
        
        # Convert torch inputs to jnp.ndarray
        images_jnp = jnp.array(batch['image'].numpy())
        instruction = batch['instruction'][0]  # Take first instruction since batch_size=1
        
        observation_jnp = jnp.array(batch['observation_state'].numpy())
        eef_state_jnp = jnp.array(batch['eef_state'].numpy())
        observation_jnp = jnp.concatenate([observation_jnp, eef_state_jnp], axis=-1)
        
        action_jnp = jnp.array(batch['action'].numpy())
        eef_action_jnp = jnp.array(batch['eef_action'].numpy())
        action_jnp = jnp.concatenate([action_jnp, eef_action_jnp], axis=-1)
        
        # Generate noise and timestep for Flow Matching
        key = jax.random.PRNGKey(step)
        key, noise_key, t_key = jax.random.split(key, 3)
        
        noise = jax.random.normal(noise_key, action_jnp.shape)
        t = jax.random.uniform(t_key, shape=(action_jnp.shape[0],))
        t_exp = t.reshape(-1, 1, 1) if action_jnp.ndim == 3 else t.reshape(-1, 1)
        x_t = (1 - t_exp) * noise + t_exp * action_jnp
        
        # Compute loss and gradients
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss_val, aux), grads = grad_fn(vla, images_jnp, instruction, observation_jnp, x_t, t, action_jnp)
        latent, noisy_emb, pred_decoded, clean_emb, clean_mask = aux
        
        # Calculate aux metrics
        grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)]))
        latent_drift = jnp.mean(jnp.abs(latent - noisy_emb))
        actual_decoded = vla.action_tokenizer.decode(clean_emb, mask=clean_mask)
        
        # Update model weights
        optimizer.update(vla, grads)
        
        import numpy as np
        mses = []
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
        
        wandb.log(log_dict, step=step)
        train_pbar.set_postfix({
            "loss": f"{float(loss_val):.4f}",
            "grad": f"{float(grad_norm):.3f}",
            "drift": f"{float(latent_drift):.3f}",
            "mse": f"{decoded_mse:.4f}"
        })

if __name__ == "__main__":
    main()
