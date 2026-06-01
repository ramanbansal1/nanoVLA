import jax 
import jax.numpy as jnp
import numpy as np
from transformers import AutoProcessor
from flax import nnx

class ActionTokenizer(nnx.Module):
    def __init__(self, hidden_size, rngs: nnx.Rngs):

        self.tokenizer = AutoProcessor.from_pretrained(
            "physical-intelligence/fast"
        )

        self.action_emb = nnx.Embed(self.tokenizer.vocab_size, hidden_size, rngs=rngs)

    def __call__(self, action):
        tokenized_actions = self.tokenizer(action, return_tensors="jax")
        token_ids = tokenized_actions["input_ids"]
        action_embed = self.action_emb(token_ids)
        return action_embed

    def decode(self, action_embed):
        tokens = jnp.argmax(action_embed @ self.action_emb.embedding.get_value().T, axis=-1)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in tokens]
        


class ObsProjector(nnx.Module):
    def __init__(self, obs_dim, hidden_size, rngs: nnx.Rngs):
        self.linear1 = nnx.Dense(hidden_size//2, rngs=rngs)
        self.layer_norm = nnx.LayerNorm(num_features=hidden_size//2, rngs=rngs)
        self.gelu = nnx.gelu
        self.layer2 = nnx.Dense(hidden_size // 2, hidden_size, rngs=rngs)

    def __call__(self, x):
        x = self.linear1(x)
        x = self.layer_norm(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x

