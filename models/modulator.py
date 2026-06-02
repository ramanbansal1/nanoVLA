import jax
import jax.numpy as jnp
from flax import nnx

class Modulator(nnx.Module):
    def __init__(self, dim: int, rngs: nnx.Rngs):
        """
        Args:
            dim: The input dimension size (d). Must be divisible by 3.
            rngs: NNX Rngs object for initialization.
        """
        if dim % 3 != 0:
            raise ValueError(f"Input dimension must be divisible by 3, got {dim}")
            
        self.dim = dim
        # Linear layer to produce 3 scores for the 3 splits
        self.score_proj = nnx.Linear(dim, 3, rngs=rngs)
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: input array of shape [b, s, d]
            
        Returns:
            Weighted sum of the 3 splits, shape [b, s, d // 3]
        """
        # 1. Compute scores: shape [b, s, 3]
        scores = self.score_proj(x)
        
        # 2. Normalize scores to weights using softmax
        weights = jax.nn.softmax(scores, axis=-1)  # shape [b, s, 3]
        
        # 3. Split input into 3 parts along the last dimension
        # Each split will have shape [b, s, d // 3]
        x1, x2, x3 = jnp.split(x, 3, axis=-1)
        
        # 4. Compute the weighted split
        # We slice weights to have shape [b, s, 1] to broadcast across the split feature dimension
        weighted_out = (weights[..., 0:1] * x1 + 
                        weights[..., 1:2] * x2 + 
                        weights[..., 2:3] * x3)
                        
        return weighted_out

if __name__ == "__main__":
    # Simple test for the Modulator
    rngs = nnx.Rngs(0)
    
    b, s, d = 2, 4, 12
    x = jax.random.normal(rngs.next(), (b, s, d))
    
    modulator = Modulator(dim=d, rngs=rngs)
    
    out = modulator(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (b, s, d // 3), "Output shape is incorrect!"
    print("Modulator test passed!")
