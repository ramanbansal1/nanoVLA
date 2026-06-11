import jax
import jax.numpy as jnp
from flax import nnx

class Modulator(nnx.Module):
    def __init__(self, in_dim: int, out_dim: int, num_splits: int, rngs: nnx.Rngs):
        """
        Args:
            in_dim: The input dimension size (e.g., 512 from VLM).
            out_dim: The projected dimension size (e.g., 576).
            num_splits: The number of tokens to split each projected token into.
            rngs: NNX Rngs object for initialization.
        """
        if out_dim % num_splits != 0:
            raise ValueError(f"Projected dimension must be divisible by {num_splits}, got {out_dim}")
            
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_splits = num_splits
        
        # Linear layer to project the input to the target dimension
        self.proj = nnx.Linear(in_dim, out_dim, rngs=rngs)
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: input array of shape [B, S, in_dim]
            
        Returns:
            Projected, GELU-activated, and split array of shape [B, S * num_splits, out_dim // num_splits]
        """
        # 1. Project: [B, S, in_dim] -> [B, S, out_dim]
        x = self.proj(x)
        
        # 2. GELU activation
        x = nnx.gelu(x)
        
        # 3. Split into tokens: [B, S, out_dim] -> [B, S * num_splits, out_dim // num_splits]
        B, S, _ = x.shape
        split_dim = self.out_dim // self.num_splits
        x = x.reshape(B, S * self.num_splits, split_dim)
        
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
