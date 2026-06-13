import jax
import jax.numpy as jnp
from flax import nnx

import math

class Modulator(nnx.Module):
    def __init__(self, in_dim: int, out_dim: int, rngs: nnx.Rngs):
        """
        Args:
            in_dim: The input dimension size (e.g., 768 from VLM).
            out_dim: The final target feature dimension per split token (e.g., hidden_size).
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Automatically determine the number of splits needed to comfortably project in_dim
        # such that each split has the target out_dim
        self.num_splits = math.ceil(in_dim / out_dim)
        self.proj_dim = self.num_splits * out_dim
        
        # Linear layer to project the input to the target dimension
        self.proj = nnx.Linear(in_dim, self.proj_dim, rngs=rngs)
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: input array of shape [B, S, in_dim]
            
        Returns:
            Projected, GELU-activated, and split array of shape [B, S * num_splits, out_dim]
        """
        # 1. Project: [B, S, in_dim] -> [B, S, proj_dim]
        x = self.proj(x)
        
        # 2. GELU activation
        x = nnx.gelu(x)
        
        # 3. Split into tokens: [B, S, proj_dim] -> [B, S * num_splits, out_dim]
        B, S, _ = x.shape
        x = x.reshape(B, S * self.num_splits, self.out_dim)
        
        return x

if __name__ == "__main__":
    # Simple test for the Modulator
    rngs = nnx.Rngs(0)
    
    b, s, in_dim = 2, 2, 512
    out_dim = 576
    num_splits = 6
    
    x = jax.random.normal(rngs.next(), (b, s, in_dim))
    
    modulator = Modulator(in_dim=in_dim, out_dim=out_dim, num_splits=num_splits, rngs=rngs)
    
    out = modulator(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    expected_shape = (b, s * num_splits, out_dim // num_splits)
    assert out.shape == expected_shape, f"Output shape {out.shape} != {expected_shape}"
    print("Modulator test passed!")
