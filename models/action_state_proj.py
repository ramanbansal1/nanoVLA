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

        self.action_emb = nnx.Embed(self.tokenizer.vocab_size, hidden_size, rngs=rngs)

    def __call__(self, action):
        token_ids = self.tokenizer(action)
        
        # Pad sequences to max length in the batch
        max_len = max(len(seq) for seq in token_ids)
        padded_token_ids = [seq + [0] * (max_len - len(seq)) for seq in token_ids]

        token_ids = jnp.asarray(padded_token_ids)
        mask = (token_ids != 0)
        
        action_embed = self.action_emb(token_ids)
        return action_embed, mask

    def decode(self, action_embed, mask=None):
        tokens = jnp.argmax(action_embed @ self.action_emb.embedding.get_value().T, axis=-1)
        # Convert to numpy for iteration and masking
        tokens = np.array(tokens)
        if mask is not None:
            mask = np.array(mask)
            
        decoded_batch = []
        for i in range(tokens.shape[0]):
            valid_tokens = tokens[i][mask[i]] if mask is not None else tokens[i]
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



if __name__ == "__main__":
    rngs = nnx.Rngs(0)

    hidden_size = 128

    # Random action chunk: [batch, horizon, action_dim]
    action_data = np.random.rand(256, 50, 14).astype(np.float32)

    action_tokenizer = ActionTokenizer(
        hidden_size=hidden_size,
        rngs=rngs,
    )

    action_embeddings = action_tokenizer(action_data)

    print("=== ActionTokenizer ===")
    print("Action input shape:", action_data.shape)
    print("Action embedding shape:", action_embeddings.shape)

    decoded_actions = action_tokenizer.decode(action_embeddings[:4])
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