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
    
    vlm_modulated, action_emb, obs_emb, dit_out, decoded_actions = vla(
        images=dummy_img,
        instruction=dummy_inst,
        observation=dummy_obs,
        action=dummy_act
    )
    
    # Check shape consistencies
    B = 1
    horizon = action_emb.shape[1]
    
    assert vlm_modulated.shape == (B, 50, 192), f"Expected (1, 50, 192), got {vlm_modulated.shape}"
    assert action_emb.shape == (B, horizon, 192), f"Expected (1, {horizon}, 192), got {action_emb.shape}"
    assert obs_emb.shape == (B, 192), f"Expected (1, 192), got {obs_emb.shape}"
    assert dit_out.shape == (B, 1 + horizon, 192), f"Expected (1, {1+horizon}, 192), got {dit_out.shape}"
    assert len(decoded_actions) == B, f"Expected {B} decoded actions, got {len(decoded_actions)}"
    print(f"Type of decoded action: {type(decoded_actions[0])}")
    
    print("All VLA shape consistencies and decoded action tests passed successfully!")
