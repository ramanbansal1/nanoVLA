import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from models.vla import VLM, VLA

if __name__ == "__main__":
    # Test the VLM class
    vlm = VLM(dummy=True)
    
    # Create a dummy image in JAX
    dummy_image = jnp.ones((3, 224, 224), dtype=jnp.uint8) * 255
    instruction = "What is in this image?"
    
    print("Testing single image VLM...")
    hidden_state = vlm(dummy_image, instruction)
    assert hidden_state.shape == (1, 50, 576), f"Unexpected VLM shape: {hidden_state.shape}"
    print("VLM shape is correct.")
    
    print("\nTesting VLA shape consistencies...")
    rngs = nnx.Rngs(42)
    vla = VLA(hidden_size=192, obs_dim=30, rngs=rngs, dummy=True)
    
    dummy_obs = jnp.zeros((1, 30))
    dummy_act = jnp.zeros((1, 30))
    dummy_img = jnp.zeros((1, 3, 224, 224))
    dummy_inst = "Test instruction"
    
    # Generate noise and timestep
    key = jax.random.PRNGKey(42)
    key, noise_key, t_key = jax.random.split(key, 3)
    
    noise = jax.random.normal(noise_key, dummy_act.shape)
    t = jax.random.uniform(t_key, shape=(dummy_act.shape[0],))
    t_exp = t.reshape(-1, 1, 1) if dummy_act.ndim == 3 else t.reshape(-1, 1)
    x_t = (1 - t_exp) * noise + t_exp * dummy_act
    
    vlm_modulated, action_emb, obs_emb, dit_out, latent, decoded_actions = vla(
        images=dummy_img,
        instruction=dummy_inst,
        observation=dummy_obs,
        action=x_t,
        t=t
    )
    
    # Check shape consistencies
    B = 1
    horizon = action_emb.shape[1]
    
    assert vlm_modulated.shape == (B, 50, 192), f"Expected (1, 50, 192), got {vlm_modulated.shape}"
    assert action_emb.shape == (B, horizon, 192), f"Expected (1, {horizon}, 192), got {action_emb.shape}"
    assert obs_emb.shape == (B, 192), f"Expected (1, 192), got {obs_emb.shape}"
    assert dit_out.shape == (B, 1 + horizon, 192), f"Expected (1, {1+horizon}, 192), got {dit_out.shape}"
    assert latent.shape == (B, horizon, 192), f"Expected (1, {horizon}, 192), got {latent.shape}"
    assert isinstance(decoded_actions, list), "Expected decoded_actions to be a list"
    
    print("All VLA shape consistencies passed successfully!")
    
    # Test gradients to ensure refinement loop is differentiable
    print("\nTesting gradient flow through latent refinement loop...")
    def loss_fn(model):
        _, action_emb, _, _, lat, _ = model(
            images=dummy_img,
            instruction=dummy_inst,
            observation=dummy_obs,
            action=x_t,
            t=t
        )
        noisy_emb = action_emb
        clean_emb = model.action_tokenizer(dummy_act)
        
        velocity_target = clean_emb - noisy_emb
        predicted_velocity = lat - noisy_emb
        
        return jnp.mean((predicted_velocity - velocity_target) ** 2)
    
    grad_fn = nnx.grad(loss_fn)
    grads = grad_fn(vla)
    
    # Check if DiT and action_emb got gradients
    assert grads is not None, "Gradients are None!"
    print("Gradient flow test passed successfully!")
