import jax 
import jax.numpy as jnp
import numpy as np
from transformers import AutoProcessor
from flax import nnx

class ActionTokenizer(nnx.Module):
    def __init__(self, hidden_size, rngs: nnx.Rngs):

        self.tokenizer = AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )

        self.action_emb = nnx.Embed(self.tokenizer.vocab_size + 1, hidden_size, rngs=rngs)

    def tokenize(self, action):
        # Convert to numpy to avoid tracer issues in scipy
        action_np = np.array(action)
        token_ids = self.tokenizer(action_np)
        
        # Pad sequences to max length in the batch
        max_len = max(len(seq) for seq in token_ids)
        pad_token = self.tokenizer.vocab_size
        padded_token_ids = [seq + [pad_token] * (max_len - len(seq)) for seq in token_ids]

        token_ids = jnp.asarray(padded_token_ids)
        mask = (token_ids != pad_token)
        return token_ids, mask

    def __call__(self, action):
        token_ids, mask = self.tokenize(action)
        action_embed = self.action_emb(token_ids)
        return action_embed, mask

    def decode(self, action_embed, mask=None):
        emb_matrix = self.action_emb.embedding.get_value()
        # Compute L2 distance squared: (x-y)^2 = x^2 + y^2 - 2xy
        x_sq = jnp.sum(action_embed ** 2, axis=-1, keepdims=True)
        y_sq = jnp.sum(emb_matrix ** 2, axis=-1)
        xy = action_embed @ emb_matrix.T
        distances = x_sq + y_sq - 2 * xy
        tokens = jnp.argmin(distances, axis=-1)
        # Convert to numpy for iteration and masking
        tokens = np.array(tokens)
        if mask is not None:
            mask = np.array(mask)
            
        decoded_batch = []
        pad_token = self.tokenizer.vocab_size
        for i in range(tokens.shape[0]):
            valid_tokens = tokens[i][mask[i]] if mask is not None else tokens[i]
            
            # During inference (mask=None), the model might predict pad_tokens at the end
            # We must filter them out so the HF tokenizer doesn't crash/return zeros.
            valid_tokens = valid_tokens[valid_tokens != pad_token]
            
            # Wrap the 1D list in another list to indicate a batch of size 1
            decoded = self.tokenizer.decode([valid_tokens.tolist()])
            decoded_batch.append(decoded[0])
        return decoded_batch
        


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
    def __init__(self, patch_size: int, hidden_size: int, rngs: nnx.Rngs):
        self.patch_size = patch_size
        self.linear = nnx.Linear(patch_size, hidden_size, rngs=rngs)

    def __call__(self, x):
        B, H, A = x.shape
        x_transposed = jnp.transpose(x, (0, 2, 1))
        
        P = self.patch_size
        assert H % P == 0, f"Horizon H ({H}) must be divisible by patch size P ({P})"
        N = H // P
        
        x_patched = jnp.reshape(x_transposed, (B, A, N, P))
        out = self.linear(x_patched)
        
        return out


class ActionUnembed(nnx.Module):
    def __init__(self, hidden_size: int, patch_size: int, rngs: nnx.Rngs):
        self.patch_size = patch_size
        self.linear = nnx.Linear(hidden_size, patch_size, rngs=rngs)

    def __call__(self, x):
        B, A, N, D = x.shape
        
        x_projected = self.linear(x)
        
        P = self.patch_size
        H = N * P
        out = jnp.reshape(x_projected, (B, A, H))
        out = jnp.transpose(out, (0, 2, 1))
        
        return out


if __name__ == "__main__":
    rngs = nnx.Rngs(0)

    hidden_size = 128

    # Random action chunk: [batch, horizon, action_dim]
    action_data = np.random.rand(256, 50, 14).astype(np.float32)

    action_tokenizer = ActionTokenizer(
        hidden_size=hidden_size,
        rngs=rngs,
    )

    action_embeddings, mask = action_tokenizer(action_data)

    print("=== ActionTokenizer ===")
    print("Action input shape:", action_data.shape)
    print("Action embedding shape:", action_embeddings.shape)

    decoded_actions = action_tokenizer.decode(action_embeddings[:4], mask=mask[:4])
    print("Decoded samples:")
    for i, action in enumerate(decoded_actions):
        print(f"{i}: {action}")

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
        patch_size=P,
        hidden_size=D,
        rngs=rngs,
    )
    
    _ = action_projector(ap_input)

    print("\n=== ActionUnembed ===")
    N = H // P
    au_input = action_projector(ap_input)
    
    action_unembed = ActionUnembed(
        hidden_size=D,
        patch_size=P,
        rngs=rngs,
    )
    
    unembed_out = action_unembed(au_input)
    assert unembed_out.shape == (B, H, A), f"Expected shape {(B, H, A)}, got {unembed_out.shape}"