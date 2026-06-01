import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

image = load_image(
    "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
)

processor = AutoProcessor.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct"
)

model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    torch_dtype=torch.bfloat16,
).to(DEVICE)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Can you describe this image?"},
        ],
    }
]

prompt = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
)

inputs = processor(
    text=prompt,
    images=[image],
    return_tensors="pt",
).to(DEVICE)

with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    )

# =====================================================
# LAST MULTIMODAL HIDDEN STATE
# =====================================================

last_hidden = outputs.hidden_states[-1]

print("Last hidden:", last_hidden.shape)

# =====================================================
# IMAGE TOKEN MASK
# =====================================================

image_token_id = model.config.image_token_id

image_mask = inputs["input_ids"] == image_token_id
text_mask = ~image_mask

print("Image positions:", image_mask.sum().item())
print("Text positions :", text_mask.sum().item())

# =====================================================
# EXTRACT TOKENS
# =====================================================

image_tokens = last_hidden[image_mask]
text_tokens = last_hidden[text_mask]

print("Image tokens shape:", image_tokens.shape)
print("Text tokens shape :", text_tokens.shape)