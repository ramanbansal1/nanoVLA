import jax
import jax.numpy as jnp
import torch
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
        # If batch, we duplicate the instruction for each image in the batch
        messages = []
        for _ in range(len(images_list)):
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": instruction},
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
            last_hidden_jnp = jax.dlpack.from_dlpack(last_hidden)
        except (TypeError, ValueError):
            # Fallback to NumPy conversion if DLPack fails or device mismatch
            last_hidden_jnp = jnp.array(last_hidden.cpu().to(torch.float32).numpy())
            
        return last_hidden_jnp




from flax import nnx
from models.action_state_proj import ActionTokenizer, ObsProjector
from models.modulator import Modulator
from models.DiT import DiT, DiTConfig

class VLA(nnx.Module):
    def __init__(self, hidden_size: int, obs_dim: int, rngs: nnx.Rngs, vlm_dim: int = 576, dummy: bool = False):
        self.hidden_size = hidden_size
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
            num_blocks=4
        )
        self.dit = DiT(config=dit_config, rngs=rngs)

    def __call__(self, images, instruction, observation, action, t=None):
        """
        Returns:
            vlm_modulated: shape [B, S, hidden_size]
            action_emb: shape [B, horizon, hidden_size]
            obs_emb: shape [B, hidden_size]
            dit_out: shape [B, 1 + horizon, hidden_size]
        """
        # 1. Process images and instructions via VLM
        vlm_out = self.vlm(images, instruction)
        
        # 2. Project and Modulate VLM features
        vlm_proj_out = self.vlm_proj(vlm_out)
        vlm_modulated = self.modulator(vlm_proj_out)
        
        # 3. Action Tokenizer
        action_emb = self.action_tokenizer(action)
        
        # 4. Observation Projector
        obs_emb = self.obs_projector(observation)
        
        # 5. DiT Integration (K=4 iterations)
        B = obs_emb.shape[0]
        obs_emb_seq = obs_emb[:, None, :] # [B, 1, hidden_size]
        
        current_action = action
        K = 4
        
        print("\n===== Iterative Refinement Debug =====")
        
        for k_iter in range(K):
            print(f"\n--- Iteration {k_iter} ---")

            print("current_action")
            print("  type :", type(current_action))

            if hasattr(current_action, "shape"):
                print("  shape:", current_action.shape)

            # Action Tokenizer
            action_emb = self.action_tokenizer(current_action)
            
            print("action_emb")
            print("  shape:", action_emb.shape)
            
            x = jnp.concatenate([obs_emb_seq, action_emb], axis=1) # [B, 1 + horizon, hidden_size]
            
            # Compute RoPE only for action tokens
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
            
            # Generate dummy t for forward pass if not provided
            if t is None:
                current_t = jnp.zeros((B,))
            else:
                current_t = t
                
            dit_out = self.dit(x=x, context=vlm_modulated, t=current_t, cos=cos, sin=sin)
            
            print("dit_out")
            print("  shape:", dit_out.shape)
            
            # Decode Action Tokens
            dit_action_emb = dit_out[:, 1:, :]
            
            print("dit_action_emb")
            print("  shape:", dit_action_emb.shape)
            
            decoded_actions = self.action_tokenizer.decode(dit_action_emb)
            
            print("decoded_actions")
            print("  type:", type(decoded_actions))
            print("  batch size:", len(decoded_actions))
        
            print("decoded_actions[0]")
            print("  type:", type(decoded_actions[0]))
        
            if hasattr(decoded_actions[0], "shape"):
                print("  shape:", decoded_actions[0].shape)
        
            print("  preview:", repr(decoded_actions[0])[:200])
            
            # Update action for next iteration
            # decoded_actions is a list of numpy arrays. The tokenizer decode might return (1, horizon, dim).
            # We need to stack them into a JAX array of shape (batch, horizon, dim)
            stacked_actions = []
            for a in decoded_actions:
                arr = jnp.array(a)
                # If the tokenizer wrapped it in an extra batch dimension, remove it
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr[0]
                stacked_actions.append(arr)
            current_action = jnp.stack(stacked_actions)
            
        return vlm_modulated, action_emb, obs_emb, dit_out, decoded_actions

if __name__ == "__main__":
    # Test the VLM class
    vlm = VLM(dummy=True)
    
    # Create a dummy image in JAX
    dummy_image = jnp.ones((3, 224, 224), dtype=jnp.uint8) * 255
    instruction = "What is in this image?"
    
    print("Testing single image VLM...")
    hidden_state = vlm(dummy_image, instruction)
    assert hidden_state.shape == (1, 50, 576), f"Unexpected VLM shape: {hidden_state.shape}"
    print("VLM shape is correct.")
    
    print("\nTesting VLA shape consistencies...")
    rngs = nnx.Rngs(42)
    vla = VLA(hidden_size=192, obs_dim=30, rngs=rngs, dummy=True)
    
    dummy_obs = jnp.zeros((1, 30))
    dummy_act = jnp.zeros((1, 30))
    dummy_img = jnp.zeros((1, 3, 224, 224))
    dummy_inst = "Test instruction"
    
    vlm_modulated, action_emb, obs_emb, dit_out, decoded_actions = vla(
        images=dummy_img,
        instruction=dummy_inst,
        observation=dummy_obs,
        action=dummy_act
    )
    
    # Check shape consistencies
    B = 1
    horizon = action_emb.shape[1]
    
    assert vlm_modulated.shape == (B, 50, 192), f"Expected (1, 50, 192), got {vlm_modulated.shape}"
    assert action_emb.shape == (B, horizon, 192), f"Expected (1, {horizon}, 192), got {action_emb.shape}"
    assert obs_emb.shape == (B, 192), f"Expected (1, 192), got {obs_emb.shape}"
    assert dit_out.shape == (B, 1 + horizon, 192), f"Expected (1, {1+horizon}, 192), got {dit_out.shape}"
    assert len(decoded_actions) == B, f"Expected {B} decoded actions, got {len(decoded_actions)}"
    print(f"Type of decoded action: {type(decoded_actions[0])}")
    
    print("All VLA shape consistencies and decoded action tests passed successfully!")
