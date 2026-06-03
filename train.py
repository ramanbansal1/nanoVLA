import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from data.advanced_dataset import VideoDataset, EpisodeIterableDataset

def main():
    # 1. Load the underlying huggingface dataset
    hf_dataset = load_dataset('data/dataset/robocoin_lemon')['train']

    # 2. Setup paths and parameters
    video_root = "data/dataset/robocoin_lemon/frames"
    action_horizon = 8

    # 3. Instantiate the VideoDataset
    train_dataset = VideoDataset(
        dataset=hf_dataset,
        video_root=video_root,
        action_horizon=action_horizon
    )

    print(f"Loaded VideoDataset with {len(train_dataset)} items.")

    from torchvision.transforms import ToTensor
    from torch.utils.data._utils.collate import default_collate

    def custom_collate_fn(batch):
        to_tensor = ToTensor()
        collated = {}
        for key in batch[0].keys():
            if key == "image":
                collated[key] = default_collate([to_tensor(item[key]) for item in batch])
            elif key == "instruction":
                collated[key] = [item[key] for item in batch]
            else:
                collated[key] = default_collate([item[key] for item in batch])
        return collated

    # 4. Create a DataLoader (example)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    # 5. Instantiate VLA
    from models.vla import VLA
    from flax import nnx
    import jax.numpy as jnp
    import jax

    # Our observation dimension is 30, and desired hidden shape is 192
    rngs = nnx.Rngs(42)
    vla = VLA(hidden_size=192, obs_dim=30, rngs=rngs, dummy=True)

    # Example iterating through one batch
    for batch in train_loader:
        print("Batch loaded!")
        
        # Convert torch inputs to jnp.ndarray
        images_jnp = jnp.array(batch['image'].numpy())
        instruction = batch['instruction'][0]  # Take first instruction since batch_size=1
        
        observation_jnp = jnp.array(batch['observation_state'].numpy())
        action_jnp = jnp.array(batch['action'].numpy())
        

        
        # Generate noise and timestep for Flow Matching
        key = jax.random.PRNGKey(42)
        key, noise_key, t_key = jax.random.split(key, 3)
        
        noise = jax.random.normal(noise_key, action_jnp.shape)
        t = jax.random.uniform(t_key, shape=(action_jnp.shape[0],))
        t_exp = t.reshape(-1, 1, 1) if action_jnp.ndim == 3 else t.reshape(-1, 1)
        x_t = (1 - t_exp) * noise + t_exp * action_jnp
        
        # Run through VLA
        vlm_modulated, action_emb, obs_emb, dit_out, latent, decoded_actions = vla(
            images=images_jnp, 
            instruction=instruction, 
            observation=observation_jnp, 
            action=x_t,
            t=t
        )
        
        clean_emb = vla.action_tokenizer(action_jnp)
        noisy_emb = action_emb
        
        # Ensure clean_emb matches noisy_emb shape for valid MSE
        target_len = noisy_emb.shape[1]
        current_len = clean_emb.shape[1]
        
        if current_len > target_len:
            clean_emb = clean_emb[:, :target_len, :]
        elif current_len < target_len:
            pad_width = ((0, 0), (0, target_len - current_len), (0, 0))
            clean_emb = jnp.pad(clean_emb, pad_width)
            
        velocity_target = clean_emb - noisy_emb
        predicted_velocity = latent - noisy_emb
            
        loss = jnp.mean((predicted_velocity - velocity_target) ** 2)
        print(f"Flow Matching Loss: {loss}")

if __name__ == "__main__":
    main()
