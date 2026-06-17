import jax
import jax.numpy as jnp
from flax import nnx

import math

class Modulator(nnx.Module):
    def __init__(self, in_dim: int, out_dim: int, rngs: nnx.Rngs):
        """
        Args:
            in_dim: The input dimension size (e.g., 768 from VLM).
            out_dim: The target feature dimension (e.g., hidden_size).
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Linear layer to project the input to the target dimension
        self.proj = nnx.Linear(in_dim, out_dim, rngs=rngs)
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: input array of shape [B, N, in_dim]
            
        Returns:
            Projected and GELU-activated array of shape [B, N, out_dim]
        """
        # 1. Project: [B, N, in_dim] -> [B, N, out_dim]
        x = self.proj(x)
        
        # 2. GELU activation
        x = nnx.gelu(x)
        
        return x

if __name__ == "__main__":
    # Simple test for the Modulator
    rngs = nnx.Rngs(0)
    
    b, s, in_dim = 2, 8, 768
    out_dim = 576
    
    x = jax.random.normal(rngs.next(), (b, s, in_dim))
    
    modulator = Modulator(in_dim=in_dim, out_dim=out_dim, rngs=rngs)
    
    out = modulator(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    expected_shape = (b, s, out_dim)
    assert out.shape == expected_shape, f"Output shape {out.shape} != {expected_shape}"
    print("Modulator test passed!")
