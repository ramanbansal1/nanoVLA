import jax
import jax.numpy as jnp
import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq

class VLM:
    def __init__(self, model_id="HuggingFaceTB/SmolVLM-256M-Instruct", device=None):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()

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

if __name__ == "__main__":
    # Test the VLM class
    vlm = VLM()
    
    # Create a dummy image in JAX
    # Let's say shape is (3, 224, 224) matching (C, H, W)
    dummy_image = jnp.ones((3, 224, 224), dtype=jnp.uint8) * 255
    instruction = "What is in this image?"
    
    print("Testing single image...")
    hidden_state = vlm(dummy_image, instruction)
    print(f"Output hidden state shape: {hidden_state.shape}")
    print(f"Output hidden state dtype: {hidden_state.dtype}")
    
    print("\nTesting batched images...")
    dummy_batch = jnp.ones((2, 3, 224, 224), dtype=jnp.uint8) * 128
    hidden_state_batch = vlm(dummy_batch, instruction)
    print(f"Batch output hidden state shape: {hidden_state_batch.shape}")
    print(f"Batch output hidden state dtype: {hidden_state_batch.dtype}")
