import dataclasses
import math

import jax
import jax.numpy as jnp
from flax import nnx


# ==========================================
# Helper Functions
# ==========================================

def flatten_action_patches(x):
    """
    (B, A, N, D) -> (B, N, A, D) -> (B, N*A, D)
    """
    B, A, N, D = x.shape
    x = jnp.transpose(x, (0, 2, 1, 3))
    x = jnp.reshape(x, (B, N * A, D))
    return x


def unflatten_action_patches(x, A, N):
    """
    (B, N*A, D) -> (B, N, A, D) -> (B, A, N, D)
    """
    B, L, D = x.shape
    assert L == N * A, f"Sequence length {L} does not match N*A ({N}*{A})"
    x = jnp.reshape(x, (B, N, A, D))
    x = jnp.transpose(x, (0, 2, 1, 3))
    return x


def apply_rope(x, cos, sin):
    """
    Applies Rotary Position Embedding to the input tensor.
    x: [B, L, H, D]
    cos, sin: [1, L, 1, D] or broadcastable
    """
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
    rotated_x = jnp.stack([-x_reshaped[..., 1], x_reshaped[..., 0]], axis=-1)
    rotated_x = rotated_x.reshape(x.shape)
    return (x * cos) + (rotated_x * sin)


def modulate(x, shift, scale):
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


# ==========================================
# Configuration
# ==========================================

@dataclasses.dataclass
class DiTConfig:
    dim: int
    context_dim: int
    num_heads: int
    mlp_hidden_dim: int
    num_blocks: int
    frequency_embedding_size: int = 256


# ==========================================
# Core Modules
# ==========================================

class TimestepEmbedder(nnx.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(
        self,
        hidden_size: int,
        rngs: nnx.Rngs,
        frequency_embedding_size: int = 256,
    ):
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size

        self.fc1 = nnx.Linear(
            frequency_embedding_size,
            hidden_size,
            rngs=rngs,
            kernel_init=nnx.initializers.normal(0.02),
        )

        self.fc2 = nnx.Linear(
            hidden_size,
            hidden_size,
            rngs=rngs,
            kernel_init=nnx.initializers.normal(0.02),
        )

    def timestep_embedding(self, t, max_period=10000):
        """
        Args:
            t: shape [B], values typically in [0, 1]
        Returns:
            shape [B, frequency_embedding_size]
        """
        t = t * max_period

        dim = self.frequency_embedding_size
        half = dim // 2

        freqs = jnp.exp(
            -math.log(max_period)
            * jnp.arange(half, dtype=jnp.float32)
            / half
        )

        args = t[:, None] * freqs[None, :]

        emb = jnp.concatenate(
            [jnp.cos(args), jnp.sin(args)],
            axis=-1,
        )

        if dim % 2:
            emb = jnp.pad(emb, ((0, 0), (0, 1)))

        return emb

    def __call__(self, t):
        x = self.timestep_embedding(t)
        x = self.fc1(x)
        x = nnx.silu(x)
        x = self.fc2(x)
        return x


class SwiGLUMLP(nnx.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        rngs: nnx.Rngs,
    ):
        self.gate = nnx.Linear(dim, hidden_dim, rngs=rngs)
        self.up = nnx.Linear(dim, hidden_dim, rngs=rngs)
        self.down = nnx.Linear(hidden_dim, dim, rngs=rngs)

    def __call__(self, x):
        gate = nnx.silu(self.gate(x))
        up = self.up(x)
        x = gate * up
        x = self.down(x)
        return x


class SelfAttention(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nnx.Linear(dim, dim * 3, rngs=rngs)
        self.out = nnx.Linear(dim, dim, rngs=rngs)
        
    def __call__(self, x, cos=None, sin=None, mask=None):
        B, L, D = x.shape
        
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_heads, self.head_dim)
        v = v.reshape(B, L, self.num_heads, self.head_dim)
        
        if cos is not None and sin is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            
        attn_weights = jnp.einsum('blhd,bshd->bhls', q, k) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -1e9)
            
        attn = jax.nn.softmax(attn_weights, axis=-1)
        
        out = jnp.einsum('bhls,bshd->blhd', attn, v)
        out = out.reshape(B, L, D)
        
        return self.out(out)


class CrossAttention(nnx.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q = nnx.Linear(dim, dim, rngs=rngs)
        self.kv = nnx.Linear(context_dim, dim * 2, rngs=rngs)
        self.out = nnx.Linear(dim, dim, rngs=rngs)
        
    def __call__(self, x, context, mask=None):
        B, L, D = x.shape
        _, S, _ = context.shape
        
        q = self.q(x)
        kv = self.kv(context)
        k, v = jnp.split(kv, 2, axis=-1)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, S, self.num_heads, self.head_dim)
        v = v.reshape(B, S, self.num_heads, self.head_dim)
        
        attn_weights = jnp.einsum('blhd,bshd->bhls', q, k) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -1e9)
            
        attn = jax.nn.softmax(attn_weights, axis=-1)
        
        out = jnp.einsum('bhls,bshd->blhd', attn, v)
        out = out.reshape(B, L, D)
        
        return self.out(out)


class DiTBlock(nnx.Module):
    """
    A DiT block consisting of Cross-Attention, Self-Attention, and a SwiGLU MLP.
    Conditioning is applied via Adaptive Layer Normalization (adaLN).
    """
    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        rngs: nnx.Rngs,
    ):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.cross_attn = CrossAttention(dim, context_dim, num_heads, rngs=rngs)
        
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.self_attn = SelfAttention(dim, num_heads, rngs=rngs)
        
        self.norm3 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = SwiGLUMLP(dim, mlp_hidden_dim, rngs=rngs)
        
        # 9 parameters for shift, scale, and gate across 3 sub-blocks
        self.adaLN_modulation = nnx.Linear(
            dim, 
            9 * dim, 
            rngs=rngs,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
        )
        
    def __call__(self, x, context, c, cos=None, sin=None, mask=None, context_mask=None):
        # c is the conditioning signal (e.g., timestep embedding)
        modulation_params = self.adaLN_modulation(nnx.silu(c))
        shift_ca, scale_ca, gate_ca, shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            modulation_params, 9, axis=-1
        )
        
        # 1. Cross Attention
        x_ca = modulate(self.norm1(x), shift_ca, scale_ca)
        x = x + gate_ca[:, None, :] * self.cross_attn(x_ca, context, mask=context_mask)
        
        # 2. Self Attention
        x_sa = modulate(self.norm2(x), shift_sa, scale_sa)
        x = x + gate_sa[:, None, :] * self.self_attn(x_sa, cos=cos, sin=sin, mask=mask)
        
        # 3. MLP
        x_mlp = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None, :] * self.mlp(x_mlp)
        
        return x


class DiT(nnx.Module):
    """
    Diffusion Transformer (DiT) model.
    """
    def __init__(self, config: DiTConfig, rngs: nnx.Rngs):
        # Check config properly
        if config.dim <= 0:
            raise ValueError(f"dim must be positive, got {config.dim}")
        if config.context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {config.context_dim}")
        if config.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {config.num_heads}")
        if config.dim % config.num_heads != 0:
            raise ValueError(f"dim must be divisible by num_heads, got dim={config.dim}, num_heads={config.num_heads}")
        if config.mlp_hidden_dim <= 0:
            raise ValueError(f"mlp_hidden_dim must be positive, got {config.mlp_hidden_dim}")
        if config.num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {config.num_blocks}")
        
        self.config = config
        
        self.timestep_embedder = TimestepEmbedder(
            hidden_size=config.dim,
            rngs=rngs,
            frequency_embedding_size=config.frequency_embedding_size,
        )
        
        self.blocks = nnx.List([
            DiTBlock(
                dim=config.dim,
                context_dim=config.context_dim,
                num_heads=config.num_heads,
                mlp_hidden_dim=config.mlp_hidden_dim,
                rngs=rngs,
            ) for _ in range(config.num_blocks)
        ])
        
        self.final_norm = nnx.LayerNorm(config.dim, rngs=rngs)
        self.null_context = nnx.Param(
            jax.random.normal(rngs(), (1, 1, config.context_dim)) * 0.02
        )
        
    def _forward(self, x, obs_emb, context, t, cos=None, sin=None, mask=None, context_mask=None):
        B, A, N, D = x.shape
        x = flatten_action_patches(x)
        
        if obs_emb is not None:
            # Expand to (B, 1, D) and prepend
            obs_emb_exp = jnp.expand_dims(obs_emb, axis=1)
            x = jnp.concatenate([obs_emb_exp, x], axis=1)
        
        c = self.timestep_embedder(t)
        
        for block in self.blocks:
            x = nnx.remat(block)(
                x=x, 
                context=context, 
                c=c, 
                cos=cos, 
                sin=sin, 
                mask=mask, 
                context_mask=context_mask
            )
            
        x = self.final_norm(x)
        
        if obs_emb is not None:
            x = x[:, 1:, :]
            
        x = unflatten_action_patches(x, A, N)
        return x

    def predict_cond(self, x, obs_emb, context, t, cos=None, sin=None, mask=None, context_mask=None):
        """Forward pass with real context."""
        return self._forward(x, obs_emb, context, t, cos, sin, mask, context_mask)
        
    def predict_uncond(self, x, obs_emb, context_shape, t, cos=None, sin=None, mask=None, context_mask=None):
        """Forward pass with null context."""
        null_ctx = jnp.broadcast_to(self.null_context[...], context_shape)
        return self._forward(x, obs_emb, null_ctx, t, cos, sin, mask, context_mask)
        
    def cfg(self, x, obs_emb, context, t, cfg_scale=1.0, cos=None, sin=None, mask=None, context_mask=None):
        """Combines predictions using classifier-free guidance."""
        eps_cond = self.predict_cond(x, obs_emb, context, t, cos, sin, mask, context_mask)
        eps_uncond = self.predict_uncond(x, obs_emb, context.shape, t, cos, sin, mask, context_mask)
        return eps_uncond + cfg_scale * (eps_cond - eps_uncond)

    def __call__(self, x, obs_emb, context, t, cos=None, sin=None, mask=None, context_mask=None, 
                 cond_drop_prob=0.0, rngs: nnx.Rngs = None):
        """
        Training forward pass with optional condition dropout.
        Args:
            x: shape [B, A, N, dim]
            obs_emb: shape [B, dim] or None
            context: shape [B, S, context_dim]
            t: shape [B]
        """
        if cond_drop_prob > 0.0 and rngs is not None:
            # Training with cond dropout
            drop_mask = jax.random.bernoulli(rngs(), cond_drop_prob, (context.shape[0], 1, 1))
            null_ctx = jnp.broadcast_to(self.null_context[...], context.shape)
            context = jnp.where(drop_mask, null_ctx, context)
            
        return self._forward(x, obs_emb, context, t, cos, sin, mask, context_mask)


# ==========================================
# Testing block
# ==========================================

if __name__ == "__main__":
    # Test the DiT block
    rngs = nnx.Rngs(0)
    
    B, L, S = 2, 16, 8
    dim = 64
    context_dim = 128
    num_heads = 4
    mlp_hidden_dim = 256
    head_dim = dim // num_heads
    
    # Initialize the block
    block = DiTBlock(
        dim=dim,
        context_dim=context_dim,
        num_heads=num_heads,
        mlp_hidden_dim=mlp_hidden_dim,
        rngs=rngs
    )
    
    # Generate random inputs
    key = jax.random.PRNGKey(1)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    
    x = jax.random.normal(k1, (B, L, dim))
    context = jax.random.normal(k2, (B, S, context_dim))
    c = jax.random.normal(k3, (B, dim))
    
    # Generate dummy RoPE embeddings
    # Shape should be broadcastable to [B, L, num_heads, head_dim]
    cos = jax.random.normal(k4, (1, L, 1, head_dim))
    sin = jax.random.normal(k5, (1, L, 1, head_dim))
    
    # Forward pass
    out = block(x, context, c, cos=cos, sin=sin)
    
    print("Testing DiTBlock...")
    print(f"Input x shape:       {x.shape}")
    print(f"Input context shape: {context.shape}")
    print(f"Input c shape:       {c.shape}")
    print(f"Output shape:        {out.shape}")
    
    assert out.shape == x.shape, f"Output shape {out.shape} does not match input shape {x.shape}!"
    print("Test passed successfully!\n")
    
    # Test the DiT model
    print("Testing DiT...")
    config = DiTConfig(
        dim=dim,
        context_dim=context_dim,
        num_heads=num_heads,
        mlp_hidden_dim=mlp_hidden_dim,
        num_blocks=2,
    )
    
    dit_model = DiT(config, rngs=rngs)
    
    # We can reuse the keys from above, or just generate a new one for t
    k6, = jax.random.split(k5, 1)
    t = jax.random.uniform(k6, (B,))
    
    A, N = 4, 4
    x_dit = x.reshape(B, A, N, dim)
    
    print("Testing conditional predict...")
    out_cond = dit_model.predict_cond(x_dit, None, context, t, cos=cos, sin=sin)
    print(f"Output cond shape:   {out_cond.shape}")
    assert out_cond.shape == x_dit.shape
    
    print("Testing unconditional predict...")
    out_uncond = dit_model.predict_uncond(x_dit, None, context.shape, t, cos=cos, sin=sin)
    print(f"Output uncond shape: {out_uncond.shape}")
    assert out_uncond.shape == x_dit.shape
    
    print("Testing CFG inference...")
    out_cfg = dit_model.cfg(x_dit, None, context, t, cfg_scale=4.5, cos=cos, sin=sin)
    assert out_cfg.shape == x_dit.shape
    
    print("Testing training forward with dropout...")
    out_train = dit_model(x_dit, None, context, t, cos=cos, sin=sin, cond_drop_prob=0.5, rngs=rngs)
    assert out_train.shape == x_dit.shape
    
    print("DiT Test passed successfully!")