import jax
import jax.numpy as jnp
from flax import nnx

import math

class Modulator(nnx.Module):
    def __init__(self, in_dim: int, out_dim: int, rngs: nnx.Rngs):
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        h_dim = 2 * out_dim

        # Input projection (no norm before first layer)
        self.proj_in = nnx.Linear(in_dim, h_dim, rngs=rngs)

        # Block 1
        self.proj1 = nnx.Linear(h_dim, h_dim, rngs=rngs)
        self.norm1 = nnx.LayerNorm(h_dim, rngs=rngs)

        # Block 2  
        self.proj2 = nnx.Linear(h_dim, h_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(h_dim, rngs=rngs)

        # Output projection
        self.proj_out = nnx.Linear(h_dim, out_dim, rngs=rngs)


    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, N, in_dim)
        
        h = self.proj_in(x)          # (B, N, h_dim)
        
        # Block 1 — pre-norm residual
        r = h
        h = self.norm1(h)
        h = self.proj1(h)
        h = nnx.gelu(h)
        h = h + r
        
        r = h
        h = self.norm2(h)
        h = self.proj2(h)
        h = nnx.gelu(h)
        h = h + r

        out = self.proj_out(h)
    
        
        return out


        
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
