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
        
        # User requested: add 2 layers, gradually decreasing.
        # "hidden size 3, hidden size 2, then hidden size"
        # "hidden size is 2 * out dim"
        h_dim = 2 * out_dim
        
        self.proj1 = nnx.Linear(in_dim, 3 * h_dim, rngs=rngs)
        self.norm1 = nnx.LayerNorm(3 * h_dim, rngs=rngs)
        
        self.proj2 = nnx.Linear(3 * h_dim, 2 * h_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(2 * h_dim, rngs=rngs)
        
        self.proj3 = nnx.Linear(2 * h_dim, out_dim, rngs=rngs)
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: input array of shape [B, N, in_dim]
            
        Returns:
            Projected array of shape [B, N, out_dim]
        """
        x = self.proj1(x)
        x = self.norm1(x)
        x = nnx.gelu(x)
        
        x = self.proj2(x)
        x = self.norm2(x)
        x = nnx.gelu(x)
        
        x = self.proj3(x)
        
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
