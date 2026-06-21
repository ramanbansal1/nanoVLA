import jax 
import jax.numpy as jnp
import numpy as np
from flax import nnx



class ObsProjector(nnx.Module):
    def __init__(self, obs_dim, hidden_size, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(obs_dim, hidden_size * 4, rngs=rngs)
        self.layer_norm = nnx.LayerNorm(num_features=hidden_size * 4, rngs=rngs)
        self.gelu = nnx.gelu
        self.linear2 = nnx.Linear(hidden_size * 4, hidden_size * 2, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_size * 2, hidden_size, rngs=rngs)

    def __call__(self, x):
        x = self.linear1(x)
        x = self.layer_norm(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.gelu(x)
        x = self.linear3(x)
        return x


class ActionProjector(nnx.Module):
    def __init__(
        self,
        action_dim: int,
        patch_size: int,
        hidden_size: int,
        rngs: nnx.Rngs,
    ):
        self.patch_size = patch_size
        self.action_dim = action_dim

        patch_dim = patch_size * action_dim

        self.action_norm = nnx.LayerNorm(
            num_features=action_dim,
            rngs=rngs,
        )

        self.linear1 = nnx.Linear(
            patch_dim,
            hidden_size * 2,
            rngs=rngs,
        )

        self.norm = nnx.LayerNorm(
            num_features=hidden_size * 2,
            rngs=rngs,
        )

        self.linear2 = nnx.Linear(
            hidden_size * 2,
            hidden_size,
            rngs=rngs,
        )

        self.gelu = nnx.gelu

    def __call__(self, x):
        B, H, A = x.shape
        
        x = self.action_norm(x)

        P = self.patch_size
        N = H // P

        assert H % P == 0

        # (B,H,A) -> (B,N,P,A)
        x = x.reshape(B, N, P, A)

        # (B,N,P,A) -> (B,N,P*A)
        x = x.reshape(B, N, P * A)

        # Patch encoder
        x = self.linear1(x)
        x = self.norm(x)
        x = self.gelu(x)
        x = self.linear2(x)

        # (B,N,D)
        return x


class ActionUnembed(nnx.Module):
    def __init__(
        self,
        action_dim: int,
        patch_size: int,
        hidden_size: int,
        rngs: nnx.Rngs,
    ):
        self.patch_size = patch_size
        self.action_dim = action_dim

        patch_dim = patch_size * action_dim

        self.linear1 = nnx.Linear(
            hidden_size,
            hidden_size * 4,
            rngs=rngs,
        )

        self.norm = nnx.LayerNorm(
            num_features=hidden_size * 4,
            rngs=rngs,
        )

        self.linear2 = nnx.Linear(
            hidden_size * 4,
            patch_dim,
            rngs=rngs,
        )

        self.gelu = nnx.gelu
        self.out_scale = nnx.Param(jnp.ones((action_dim,)))

    def __call__(self, x):
        B, N, D = x.shape

        x = self.linear1(x)
        x = self.norm(x)
        x = self.gelu(x)
        x = self.linear2(x)

        # (B,N,P*A)
        x = x.reshape(
            B,
            N,
            self.patch_size,
            self.action_dim,
        )

        # (B,H,A)
        x = x.reshape(
            B,
            N * self.patch_size,
            self.action_dim,
        )

        x = x * self.out_scale

        return x


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