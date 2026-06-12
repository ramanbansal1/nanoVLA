import jax 
import jax.numpy as jnp
import numpy as np
from flax import nnx



class ObsProjector(nnx.Module):
    def __init__(self, obs_dim, hidden_size, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(obs_dim, hidden_size//2, rngs=rngs)
        self.layer_norm = nnx.LayerNorm(num_features=hidden_size//2, rngs=rngs)
        self.gelu = nnx.gelu
        self.linear2 = nnx.Linear(hidden_size // 2, hidden_size, rngs=rngs)

    def __call__(self, x):
        x = self.linear1(x)
        x = self.layer_norm(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class ActionProjector(nnx.Module):
    def __init__(self, action_dim: int, patch_size: int, hidden_size: int, compression: int, rngs: nnx.Rngs):
        self.patch_size = patch_size
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.compression = compression
        self.compressed_dim = action_dim // compression

        self.linear = nnx.Linear(patch_size, hidden_size, rngs=rngs)
        self.norm = nnx.LayerNorm(num_features=hidden_size, rngs=rngs)
        self.gelu = nnx.gelu
        self.conv = nnx.Conv(
            in_features=hidden_size,
            out_features=hidden_size,
            kernel_size=(compression,),
            strides=(compression,),
            rngs=rngs
        )

    def __call__(self, x):
        B, H, A = x.shape
        P = self.patch_size
        N = H // P
        assert H % P == 0

        # Step 1: (B, H, A) → (B, N, P, A)
        x = jnp.reshape(x, (B, N, P, A))

        # Step 2: → (B, N, A, P)  ← your desired shape
        x = jnp.transpose(x, (0, 1, 3, 2))

        # Step 3: Linear over P → (B, N, A, hidden_size)
        x = self.linear(x)
        x = self.norm(x)
        x = self.gelu(x)

        # Step 4: Conv over A → (B*N, A//c, hidden_size)
        x = jnp.reshape(x, (B * N, A, self.hidden_size))
        x = self.conv(x)                                  # (B*N, A//c, D)

        # Step 5: → (B, N * (A//c), hidden_size)
        out = jnp.reshape(x, (B, N * self.compressed_dim, self.hidden_size))
        return out


class ActionUnembed(nnx.Module):
    def __init__(self, action_dim: int, hidden_size: int, patch_size: int, compression: int, rngs: nnx.Rngs):
        self.patch_size = patch_size
        self.action_dim = action_dim
        self.compression = compression
        self.compressed_dim = action_dim // compression

        # Deconv to expand A//c → A
        self.deconv = nnx.ConvTranspose(
            in_features=hidden_size,
            out_features=hidden_size,
            kernel_size=(compression,),
            strides=(compression,),
            rngs=rngs
        )
        
        self.gelu = nnx.gelu

        # Linear over hidden_size → P
        self.linear = nnx.Linear(hidden_size, patch_size, rngs=rngs)

    def __call__(self, x, N: int):
        # x: (B, N * (A//c), D)
        B = x.shape[0]
        A_c = self.compressed_dim
        D = x.shape[-1]

        # Step 1: → (B*N, A//c, D)
        x = jnp.reshape(x, (B * N, A_c, D))

        # Step 2: Deconv → (B*N, A, D)
        x = self.deconv(x)
        x = self.gelu(x)

        # Step 3: Linear → (B*N, A, P)
        x = self.linear(x)                            # (B*N, A, P)

        # Step 4: → (B, N, A, P)
        x = jnp.reshape(x, (B, N, self.action_dim, self.patch_size))

        # Step 5: → (B, H, A)
        x = jnp.transpose(x, (0, 1, 3, 2))           # (B, N, P, A)
        out = jnp.reshape(x, (B, N * self.patch_size, self.action_dim))

        return out


if __name__ == "__main__":
    rngs = nnx.Rngs(0)

    hidden_size = 128

    # Random observation batch
    obs_dim = 64
    obs_data = jax.random.normal(
        jax.random.PRNGKey(42),
        (256, obs_dim),
    )

    obs_projector = ObsProjector(
        obs_dim=obs_dim,
        hidden_size=hidden_size,
        rngs=rngs,
    )

    projected_obs = obs_projector(obs_data)

    print("\n=== ObsProjector ===")
    print("Observation shape:", obs_data.shape)
    print("Projected shape:", projected_obs.shape)

    print("\n=== ActionProjector ===")
    B, H, A = 16, 60, 14
    P = 15
    D = 96
    
    assert H % P == 0
    
    ap_input = jax.random.normal(jax.random.PRNGKey(123), (B, H, A))
    
    action_projector = ActionProjector(
        action_dim=A,
        patch_size=P,
        hidden_size=D,
        rngs=rngs,
    )
    
    _ = action_projector(ap_input)

    print("\n=== ActionUnembed ===")
    N = H // P
    au_input = action_projector(ap_input)
    
    action_unembed = ActionUnembed(
        action_dim=A,
        hidden_size=D,
        patch_size=P,
        rngs=rngs,
    )
    
    unembed_out = action_unembed(au_input)
    assert unembed_out.shape == (B, H, A), f"Expected shape {(B, H, A)}, got {unembed_out.shape}"