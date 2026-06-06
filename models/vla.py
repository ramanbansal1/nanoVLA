import jax
import jax.numpy as jnp
import torch
import torch.utils.dlpack
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq

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
        Passes an image (or batch of images) and a text instruction through SmolVLM,
        returning the final multimodal hidden state.

        Args:
            images: jnp.ndarray of shape (H, W, C) for a single image, 
                    or (B, H, W, C) / (B, C, H, W) for a batch.
            instruction: string containing the instruction.

        Returns:
            final_hidden_state: jnp.ndarray of shape (B, seq_len, hidden_dim)
        """
        if self.dummy:
            images_np = np.array(images)
            is_batch = images_np.ndim == 4
            batch_size = images_np.shape[0] if is_batch else 1
            seq_len = 50  # Mock sequence length
            hidden_dim = 576  # SmolVLM text hidden dim
            key = jax.random.PRNGKey(np.random.randint(0, 10000))
            return jax.random.normal(key, (batch_size, seq_len, hidden_dim))

        # Convert JAX array to numpy for the Hugging Face processor
        images_np = np.array(images)
        
        # Determine if batch or single
        is_batch = True
        if images_np.ndim == 3:
            images_list = [images_np]
            is_batch = False
        elif images_np.ndim == 4:
            # The processor expects a list of images for a batch
            images_list = [img for img in images_np]
        else:
            raise ValueError(f"Expected images to be 3D or 4D, got {images_np.ndim}D")

        # Build messages format for the processor
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

        # Apply chat template
        prompts = [
            self.processor.apply_chat_template(msg, add_generation_prompt=True)
            for msg in messages
        ]

        # Prepare inputs for the model
        inputs = self.processor(
            text=prompts if is_batch else prompts[0],
            images=images_list if is_batch else images_list[0],
            return_tensors="pt",
            padding=True,
            do_rescale=False,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # Extract the final hidden state
        # Shape: (B, SequenceLength, HiddenDimension)
        last_hidden = outputs.hidden_states[-1]
        
        # Ensure contiguous memory before converting
        last_hidden = last_hidden.contiguous().detach()
        
        try:
            # PyTorch tensors natively support the DLPack protocol in recent versions
            last_hidden_jnp = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(last_hidden))
        except Exception:
            # Fallback to NumPy conversion if DLPack fails or device mismatch
            last_hidden_jnp = jnp.array(last_hidden.cpu().to(torch.float32).numpy())
            
        return last_hidden_jnp




from flax import nnx
from models.action_state_proj import ActionTokenizer, ObsProjector
from models.modulator import Modulator
from models.DiT import DiT, DiTConfig

class VLA(nnx.Module):
    def __init__(self, hidden_size: int, obs_dim: int, rngs: nnx.Rngs, vlm_dim: int = 576, dummy: bool = False, dit_num_blocks: int = 4, vla_k: int = 4):
        self.hidden_size = hidden_size
        self.vla_k = vla_k
        self.vlm = VLM(dummy=dummy)
        
        # Project VLM output to 3 * hidden_size for the Modulator
        self.vlm_proj = nnx.Linear(vlm_dim, hidden_size * 3, rngs=rngs)
        self.modulator = Modulator(dim=hidden_size * 3, rngs=rngs)
        
        self.action_tokenizer = ActionTokenizer(hidden_size=hidden_size, rngs=rngs)
        self.obs_projector = ObsProjector(obs_dim=obs_dim, hidden_size=hidden_size, rngs=rngs)
        
        dit_config = DiTConfig(
            dim=hidden_size,
            context_dim=hidden_size,
            num_heads=6,
            mlp_hidden_dim=hidden_size * 4,
            num_blocks=dit_num_blocks
        )
        self.dit = DiT(config=dit_config, rngs=rngs)

    def __call__(self, images, instruction, observation, action=None, action_emb=None, action_mask=None, t=None, decode_action=False, vlm_out=None):
        """
        Returns:
            vlm_modulated: shape [B, S, hidden_size]
            action_emb: shape [B, horizon, hidden_size]
            action_mask: shape [B, horizon]
            obs_emb: shape [B, hidden_size]
            dit_out: shape [B, 1 + horizon, hidden_size]
            latent: shape [B, horizon, hidden_size]
            decoded_actions: list of decoded action sequences
        """
        # 1. Process images and instructions via VLM
        if vlm_out is None:
            vlm_out = self.vlm(images, instruction)
        
        # 2. Project and Modulate VLM features
        vlm_proj_out = self.vlm_proj(vlm_out)
        vlm_modulated = self.modulator(vlm_proj_out)
        
        # 3. Action Tokenizer (using whatever is passed as action, e.g., x_t)
        if action is not None:
            action_emb, action_mask = self.action_tokenizer(action)
        elif action_emb is None or action_mask is None:
            raise ValueError("Either action or (action_emb, action_mask) must be provided")
        
        # 4. Observation Projector
        obs_emb = self.obs_projector(observation)
        
        # Compute RoPE once for all iterations
        action_len = action_emb.shape[1]
        num_heads = self.dit.config.num_heads
        head_dim = self.hidden_size // num_heads
        
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        t_rope = np.arange(action_len, dtype=np.float32)
        freqs = np.outer(t_rope, inv_freq)
        emb = np.repeat(freqs, 2, axis=-1)
        action_cos = jnp.array(np.cos(emb)[None, :, None, :])
        action_sin = jnp.array(np.sin(emb)[None, :, None, :])
        
        # Identity RoPE (no rotation) for observation token
        obs_cos = jnp.ones((1, 1, 1, head_dim))
        obs_sin = jnp.zeros((1, 1, 1, head_dim))
        
        cos = jnp.concatenate([obs_cos, action_cos], axis=1)
        sin = jnp.concatenate([obs_sin, action_sin], axis=1)
        
        # 5. DiT Integration (Predict and Refine)
        B = obs_emb.shape[0]
        obs_emb_seq = obs_emb[:, None, :] # [B, 1, hidden_size]
        
        if t is None:
            current_t = jnp.zeros((B,))
        else:
            current_t = t
            
        latent = action_emb
        K = self.vla_k
        
        for k_iter in range(K):
            x = jnp.concatenate([obs_emb_seq, latent], axis=1) # [B, 1 + horizon, hidden_size]
            
            # Create full mask (True for obs token, action_mask for action tokens)
            obs_mask = jnp.ones((B, 1), dtype=jnp.bool_)
            full_mask = jnp.concatenate([obs_mask, action_mask], axis=1)
            # Reshape for broadcasting in self-attention: [B, 1, 1, 1 + action_len]
            full_mask = full_mask[:, None, None, :]
            
            dit_out = self.dit(x=x, context=vlm_modulated, t=current_t, cos=cos, sin=sin, mask=full_mask)
            
            dit_action_emb = dit_out[:, 1:, :]
            latent = latent + (dit_action_emb / K)
            
        # 6. Final Decode (Only once at the end)
        if decode_action:
            decoded_actions = self.action_tokenizer.decode(latent, mask=action_mask)
        else:
            decoded_actions = None
            
        return vlm_modulated, action_emb, action_mask, obs_emb, dit_out, latent, decoded_actions
