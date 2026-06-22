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
        hidden_size: int,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim

        self.action_norm = nnx.LayerNorm(
            num_features=action_dim,
            rngs=rngs,
            use_scale=True,
            use_bias=True,
        )

        # magnitude gets its own small embedding path
        # log-magnitude per action dim preserves scale info
        self.mag_linear = nnx.Linear(
            action_dim,
            hidden_size,
            rngs=rngs,
        )

        self.linear1 = nnx.Linear(
            action_dim,
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

        # FiLM: magnitude modulates the normalized direction
        self.film_linear = nnx.Linear(
            hidden_size,
            hidden_size * 2,  # gamma + beta
            rngs=rngs,
        )

        self.gelu = nnx.gelu

    def __call__(self, x):
        # x: (B, H, A)

        # --- magnitude path ---
        # log1p so small values don't blow up, sign preserved separately
        mag = jnp.log1p(jnp.abs(x)) * jnp.sign(x)  # (B, H, A)
        mag_emb = self.mag_linear(mag)                # (B, H, D)
        mag_emb = self.gelu(mag_emb)

        # FiLM params from magnitude
        film = self.film_linear(mag_emb)              # (B, H, D*2)
        gamma, beta = jnp.split(film, 2, axis=-1)     # each (B, H, D)
        # init-safe: gamma near 1, beta near 0 handled by linear init

        # --- direction path ---
        x_norm = self.action_norm(x)                  # normalized direction
        h = self.linear1(x_norm)
        h = self.norm(h)
        h = self.gelu(h)
        h = self.linear2(h)                           # (B, H, D)

        # --- FiLM fusion ---
        # magnitude tells direction-path how to rescale itself
        out = gamma * h + beta                        # (B, H, D)

        return out


class ActionUnembed(nnx.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_size: int,
        context_dim: int,      # e.g. obs embedding dim or DiT hidden size
        num_mixtures: int = 4,
        rngs: nnx.Rngs = None,
    ):
        self.action_dim = action_dim
        self.num_mixtures = num_mixtures

        # Main projection
        self.linear1 = nnx.Linear(hidden_size, hidden_size * 4, rngs=rngs)
        self.norm = nnx.LayerNorm(num_features=hidden_size * 4, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_size * 4, action_dim * num_mixtures, rngs=rngs)

        # FiLM generator from context (obs embedding or DiT output)
        # Produces (gamma, beta) per action dim per mixture
        self.film_linear = nnx.Linear(
            context_dim,
            action_dim * num_mixtures * 2,   # *2 for scale + shift
            rngs=rngs,
        )

        # Mixture logits from context
        self.mixture_logits = nnx.Linear(
            context_dim,
            num_mixtures,
            rngs=rngs,
        )

        self.gelu = nnx.gelu

    def __call__(self, x, context):
        # x: (B, H, D)  — DiT output tokens
        # context: (B, context_dim)  — e.g. pooled obs embedding

        B, H, D = x.shape

        # --- main branch ---
        h = self.linear1(x)
        h = self.norm(h)
        h = self.gelu(h)
        h = self.linear2(h)
        # h: (B, H, A*M)
        h = h.reshape(B, H, self.action_dim, self.num_mixtures)

        # --- FiLM conditioning from context ---
        film = self.film_linear(context)          # (B, A*M*2)
        film = film.reshape(B, 1, self.action_dim, self.num_mixtures, 2)
        gamma = film[..., 0]   # (B, 1, A, M)
        beta  = film[..., 1]   # (B, 1, A, M)

        # scale + shift — this is where context controls per-dim range
        h = gamma * h + beta   # broadcast over H

        # --- mixture weights from context ---
        logits = self.mixture_logits(context)     # (B, M)
        weights = jax.nn.softmax(logits, axis=-1) # (B, M)
        weights = weights[:, None, None, :]        # (B, 1, 1, M)

        # weighted sum over mixtures
        out = (h * weights).sum(axis=-1)           # (B, H, A)

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
    D = 96
    
    ap_input = jax.random.normal(jax.random.PRNGKey(123), (B, H, A))
    
    action_projector = ActionProjector(
        action_dim=A,
        hidden_size=D,
        rngs=rngs,
    )
    
    _ = action_projector(ap_input)

    print("\n=== ActionUnembed ===")
    au_input = action_projector(ap_input)
    
    action_unembed = ActionUnembed(
        action_dim=A,
        hidden_size=D,
        context_dim=D,
        rngs=rngs,
    )
    
    dummy_context = jax.random.normal(jax.random.PRNGKey(456), (B, D))
    unembed_out = action_unembed(au_input, dummy_context)
    assert unembed_out.shape == (B, H, A), f"Expected shape {(B, H, A)}, got {unembed_out.shape}"