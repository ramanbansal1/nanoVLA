import jax
import jax.numpy as jnp
from flax import nnx

class Modulator(nnx.Module):
    def __init__(self, dim: int, num_splits: int, rngs: nnx.Rngs):
        """
        Args:
            dim: The input dimension size (d). Must be divisible by num_splits.
            num_splits: The number of parts to split the input into.
            rngs: NNX Rngs object for initialization.
        """
        if dim % num_splits != 0:
            raise ValueError(f"Input dimension must be divisible by {num_splits}, got {dim}")
            
        self.dim = dim
        self.num_splits = num_splits
        # Linear layer to produce scores for the splits
        self.score_proj = nnx.Linear(dim, num_splits, rngs=rngs)
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: input array of shape [b, s, d]
            
        Returns:
            Weighted sum of the splits, shape [b, s, d // num_splits]
        """
        # 1. Compute scores: shape [b, s, num_splits]
        scores = self.score_proj(x)
        
        # 2. Normalize scores to weights using softmax
        weights = jax.nn.softmax(scores, axis=-1)  # shape [b, s, num_splits]
        
        # 3. Split input into parts along the last dimension
        splits = jnp.split(x, self.num_splits, axis=-1)
        
        # 4. Compute the weighted split dynamically
        weighted_out = sum([weights[..., i:i+1] * splits[i] for i in range(self.num_splits)])
                        
        return weighted_out

if __name__ == "__main__":
    # Simple test for the Modulator
    rngs = nnx.Rngs(0)
    
    b, s, d = 2, 4, 12
    x = jax.random.normal(rngs.next(), (b, s, d))
    
    num_splits = 3
    modulator = Modulator(dim=d, num_splits=num_splits, rngs=rngs)
    
    out = modulator(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (b, s, d // num_splits), "Output shape is incorrect!"
    print("Modulator test passed!")
