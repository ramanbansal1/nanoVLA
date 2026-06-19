import os
import sys

# Allow running this file directly by appending project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os

import sys

# Allow running this file directly by appending project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import torch
import torch.utils.dlpack
import numpy as np
from flax import nnx
from models.action_state_proj import ActionProjector, ActionUnembed, ObsProjector
from models.modulator import Modulator
from models.DiT import DiT, DiTConfig
from models.visual_encoder import SigLIP

class VLA(nnx.Module):
    def __init__(self, hidden_size: int, obs_dim: int, rngs: nnx.Rngs, dit_num_blocks: int = 4, vla_k: int = 4, patch_size: int = 5, action_dim: int = 43, horizon: int = 120, vlm_checkpoint_path: str = "checkpoints/siglip2_naflex.npz", action_compression: int = 5):
        self.hidden_size = hidden_size
        self.vla_k = vla_k
        self.patch_size = patch_size
        self.action_dim = action_dim
        self.horizon = horizon
        self.action_compression = action_compression
        
        self.vlm = SigLIP(checkpoint_path=vlm_checkpoint_path, normalize=True)

        self.modulator = Modulator(in_dim=768, out_dim=hidden_size * 4, rngs=rngs)
        
        self.action_projector = ActionProjector(action_dim=action_dim, patch_size=patch_size, hidden_size=hidden_size, rngs=rngs)
        self.action_unembed = ActionUnembed(action_dim=action_dim, hidden_size=hidden_size, patch_size=patch_size, rngs=rngs)
        self.obs_projector = ObsProjector(obs_dim=obs_dim, hidden_size=hidden_size, rngs=rngs)
        
        dit_config = DiTConfig(
            dim=hidden_size,
            context_dim=hidden_size * 4,
            num_heads=6,
            mlp_hidden_dim=hidden_size * 4,
            num_blocks=dit_num_blocks
        )
        self.dit = DiT(config=dit_config, rngs=rngs)

    def _get_rope(self, seq_len):
        num_heads = self.dit.config.num_heads
        head_dim = self.hidden_size // num_heads
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        
        t_rope = np.arange(seq_len - 1, dtype=np.float32)
        freqs = np.outer(t_rope, inv_freq)
        emb = np.repeat(freqs, 2, axis=-1)
        action_cos = jnp.array(np.cos(emb)[None, :, None, :])
        action_sin = jnp.array(np.sin(emb)[None, :, None, :])
        
        obs_cos = jnp.ones((1, 1, 1, head_dim))
        obs_sin = jnp.zeros((1, 1, 1, head_dim))
        
        cos = jnp.concatenate([obs_cos, action_cos], axis=1)
        sin = jnp.concatenate([obs_sin, action_sin], axis=1)
        return cos, sin

    def __call__(self, images, input_ids, observation, action=None, t=None, vlm_out=None, key=None, cond_drop_prob=0.0, rngs=None):
        """
        Executes the Vision-Language-Action forward pass, or K-step flow matching if action is None.
        
        Args:
            images: (B, H, W, C) array of images or PIL images.
            input_ids: (B, 64) array of text token indices.
            observation: (B, obs_dim) array of robot state.
            action: (B, horizon, action_dim) array of continuous actions (can be noisy x_t).
            t: (B,) array of timesteps (for training).
            vlm_out: Optional pre-computed VLM features.
            key: PRNG key for generation sampling.
            cond_drop_prob: Dropout probability for CFG training.
            rngs: NNX PRNG state for training dropout.
            
        Returns:
            pred_v_raw for training, or x_t for inference.
        """
        if vlm_out is None:
            img_hidden = jnp.asarray(self.vlm.encode_images(images))
            txt_hidden = jnp.asarray(self.vlm.encode_texts(input_ids))
            vlm_out = jnp.concatenate([txt_hidden, img_hidden], axis=1)
        
        vlm_modulated = self.modulator(vlm_out)
        obs_emb = self.obs_projector(observation)
        
        # Training Mode
        if action is not None:
            B = obs_emb.shape[0]
            
            action_proj = self.action_projector(action)
                
            _, N, _ = action_proj.shape
            
            seq_len = N + 1
            cos, sin = self._get_rope(seq_len)
            
            if t is None:
                t = jnp.ones((B,))

            dit_out = self.dit(
                x=action_proj, 
                obs_emb=obs_emb, 
                context=vlm_modulated, 
                t=t, 
                cos=cos, 
                sin=sin, 
                mask=None,
                cond_drop_prob=cond_drop_prob,
                rngs=rngs
            )
            
            pred_v_raw = self.action_unembed(dit_out)
            return pred_v_raw
            
        # Inference / Generation Mode (Continuous Flow Matching Euler steps)
        else:
            if key is None:
                raise ValueError("A PRNG key must be provided for generation.")
            
            B = obs_emb.shape[0]
            N = self.horizon // self.patch_size
            
            # Start with random noise in unembedded space
            key, subkey = jax.random.split(key)
            x_t = jax.random.normal(subkey, (B, self.horizon, self.action_dim))
            
            seq_len = N + 1
            cos, sin = self._get_rope(seq_len)
            
            dt = 1.0 / self.vla_k
            v_pred_raw = None
            
            for k in range(self.vla_k):
                t_val = jnp.full((B,), k / self.vla_k)
                
                action_proj = self.action_projector(x_t)
                
                v_pred_proj = self.dit.cfg(
                    x=action_proj, 
                    obs_emb=obs_emb, 
                    context=vlm_modulated, 
                    t=t_val, 
                    cfg_scale=1.5, 
                    cos=cos, 
                    sin=sin, 
                    mask=None
                )
                
                v_pred_raw = self.action_unembed(v_pred_proj)
                
                x_t = x_t + v_pred_raw * dt
                
            return x_t
