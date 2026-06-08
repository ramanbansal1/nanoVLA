import jax
import jax.numpy as jnp
import torch
import torch.utils.dlpack
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq
from flax import nnx
from models.action_state_proj import ActionProjector, ActionUnembed, ObsProjector
from models.modulator import Modulator
from models.DiT import DiT, DiTConfig

class VLM:
    def __init__(self, model_id="HuggingFaceTB/SmolVLM-256M-Instruct", device=None, dummy: bool = False):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.dummy = dummy
        self.model_id = model_id
        
        if not self.dummy:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
            ).to(self.device)
            self.model.eval()
        else:
            print(f"Initializing Dummy VLM for fast testing (mocking {model_id})...")

    def __call__(self, images: jnp.ndarray, instruction: str) -> jnp.ndarray:
        """
        Passes an image and instruction through the VLM to get multimodal hidden states.
        
        Args:
            images: (B, H, W, C) array of images.
            instruction: string or list of strings containing instructions.
            
        Returns:
            last_hidden_jnp: (B, seq_len, hidden_dim) array of hidden states.
        """
        if self.dummy:
            images_np = np.array(images)
            is_batch = images_np.ndim == 4
            batch_size = images_np.shape[0] if is_batch else 1
            key = jax.random.PRNGKey(np.random.randint(0, 10000))
            return jax.random.normal(key, (batch_size, 50, 576))

        images_np = np.array(images)
        is_batch = True
        if images_np.ndim == 3:
            images_list = [images_np]
            is_batch = False
        elif images_np.ndim == 4:
            images_list = [img for img in images_np]
        else:
            raise ValueError(f"Expected images to be 3D or 4D, got {images_np.ndim}D")

        messages = []
        for i in range(len(images_list)):
            inst = instruction[i] if isinstance(instruction, list) else instruction
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": inst},
                    ],
                }
            ])

        prompts = [self.processor.apply_chat_template(msg, add_generation_prompt=True) for msg in messages]
        inputs = self.processor(
            text=prompts if is_batch else prompts[0],
            images=images_list if is_batch else images_list[0],
            return_tensors="pt",
            padding=True,
            do_rescale=False,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

        last_hidden = outputs.hidden_states[-1].contiguous().detach()
        try:
            last_hidden_jnp = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(last_hidden))
        except Exception:
            last_hidden_jnp = jnp.array(last_hidden.cpu().to(torch.float32).numpy())
            
        return last_hidden_jnp


class VLA(nnx.Module):
    def __init__(self, hidden_size: int, obs_dim: int, rngs: nnx.Rngs, vlm_dim: int = 576, dummy: bool = False, dit_num_blocks: int = 4, vla_k: int = 4, patch_size: int = 5, action_dim: int = 43, horizon: int = 120):
        self.hidden_size = hidden_size
        self.vla_k = vla_k
        self.patch_size = patch_size
        self.action_dim = action_dim
        self.horizon = horizon
        self.vlm = VLM(dummy=dummy)
        
        self.num_splits = 6
        self.vlm_proj = nnx.Linear(vlm_dim, hidden_size * self.num_splits, rngs=rngs)
        self.modulator = Modulator(dim=hidden_size * self.num_splits, num_splits=self.num_splits, rngs=rngs)
        
        self.action_projector = ActionProjector(patch_size=patch_size, hidden_size=hidden_size, rngs=rngs)
        self.action_unembed = ActionUnembed(hidden_size=hidden_size, patch_size=patch_size, rngs=rngs)
        self.obs_projector = ObsProjector(obs_dim=obs_dim, hidden_size=hidden_size, rngs=rngs)
        
        dit_config = DiTConfig(
            dim=hidden_size,
            context_dim=hidden_size,
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

    def __call__(self, images, instruction, observation, action=None, t=None, vlm_out=None, key=None, cond_drop_prob=0.0, rngs=None):
        """
        Executes the Vision-Language-Action forward pass, or K-step flow matching if action is None.
        
        Args:
            images: (B, H, W, C) array of images.
            instruction: string or list of instructions.
            observation: (B, obs_dim) array of robot state.
            action: (B, horizon, action_dim) array of continuous actions (can be noisy x_t).
            t: (B,) array of timesteps (for training).
            vlm_out: Optional pre-computed VLM features.
            key: PRNG key for generation sampling.
            cond_drop_prob: Dropout probability for CFG training.
            rngs: NNX PRNG state for training dropout.
            
        Returns:
            Tuple containing modulated VLM features, projected actions, None, obs_emb, dit_out, un-embedded actions, decoded actions.
        """
        if vlm_out is None:
            vlm_out = self.vlm(images, instruction)
        
        vlm_proj_out = self.vlm_proj(vlm_out)
        vlm_modulated = self.modulator(vlm_proj_out)
        obs_emb = self.obs_projector(observation)
        
        # Training Mode
        if action is not None:
            B = obs_emb.shape[0]
            
            action_proj = self.action_projector(action)
                
            _, A, N, _ = action_proj.shape
            
            seq_len = A * N + 1
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
            return vlm_modulated, action_proj, None, obs_emb, dit_out, pred_v_raw, None
            
        # Inference / Generation Mode (Continuous Flow Matching Euler steps)
        else:
            if key is None:
                raise ValueError("A PRNG key must be provided for generation.")
            
            B = obs_emb.shape[0]
            N = self.horizon // self.patch_size
            
            # Start with random noise in unembedded space
            key, subkey = jax.random.split(key)
            x_t = jax.random.normal(subkey, (B, self.horizon, self.action_dim))
            
            seq_len = self.action_dim * N + 1
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
                
            return vlm_modulated, x_t, None, obs_emb, v_pred_raw, None, x_t
