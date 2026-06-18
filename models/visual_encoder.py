import numpy as np
import jax
import jax.numpy as jnp

from PIL import Image
from transformers import AutoTokenizer

from big_vision import utils
from big_vision.models.proj.image_text.naflex_vit import (
    _Model as ImageModel,
)
from big_vision.models.proj.image_text.text_transformer import (
    _Model as TextModel,
)


class SigLIP:
    """
    SigLIP2 Base (Big Vision checkpoint)
    """

    def __init__(
        self,
        checkpoint_path: str,
        normalize: bool = True,
        max_text_length: int = 64,
    ):
        self.normalize = normalize
        self.max_text_length = max_text_length

        ckpt = utils.load_params(checkpoint_path)
        self.img_params = ckpt["img"]
        self.txt_params = ckpt["txt"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/siglip2-base-patch16-naflex"
        )
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            "google/siglip2-base-patch16-naflex"
        )

        # ================= IMAGE MODEL =================
        self.image_model = ImageModel(
            width=768,
            depth=12,
            mlp_dim=3072,
            num_heads=12,
            pool_type="map",
            nposemb=16,
            scan=True,
        )

        image_dummy = (
            jnp.zeros((1, 256, 768), dtype=jnp.float32),
            jnp.ones((1, 256), dtype=jnp.int32),
            jnp.zeros((1, 256), dtype=jnp.int32),
            jnp.zeros((1, 256), dtype=jnp.int32),
        )

        self.image_model.init(jax.random.PRNGKey(0), image_dummy, train=False)

        # ================= TEXT MODEL =================
        self.text_model = TextModel(
            num_classes=768,
            width=768,
            depth=12,
            mlp_dim=3072,
            num_heads=12,
            vocab_size=256000,
            pool_type="last",
            scan=True,
        )

        text_dummy = jnp.zeros((1, max_text_length), dtype=jnp.int32)
        self.text_model.init(jax.random.PRNGKey(0), text_dummy, train=False)

        # ================= JIT =================
        self._encode_image_jit = jax.jit(self._encode_image_impl)
        self._encode_text_jit = jax.jit(self._encode_text_impl)

        img_out = self._encode_image_jit(*image_dummy)
        img_out.block_until_ready()   # hidden
        
        txt_out = self._encode_text_jit(text_dummy)
        txt_out.block_until_ready()   # hidden

    # ======================================================
    # IMAGE ENCODER (emb + hidden)
    # ======================================================
    def _encode_image_impl(self, patches, ptype, yabs, xabs):
        emb, out = self.image_model.apply(
            {"params": self.img_params},
            (patches, ptype, yabs, xabs),
            train=False,
        )

        hidden = out["encoded"]   # (B, Npatches, 768)

        return hidden

    # ======================================================
    # TEXT ENCODER (emb + hidden)
    # ======================================================
    def _encode_text_impl(self, input_ids):
        emb, out = self.text_model.apply(
            {"params": self.txt_params},
            input_ids,
            train=False,
        )

        hidden = out['transformed']

        return hidden

    # ======================================================
    # IMAGE PREPROCESSING
    # ======================================================
    def images_to_naflex(self, images):
        if not isinstance(images, list) and not isinstance(images, tuple) and not (
            hasattr(images, "ndim") and images.ndim == 4
        ):
            images = [images]

        out = self.processor(images=images, return_tensors="pt", padding=True)
        pixel_values = out["pixel_values"].numpy()
        ptypes = out["pixel_attention_mask"].numpy().astype(np.int32)
        spatial_shapes = out["spatial_shapes"].numpy()

        batch_size, max_patches, _ = pixel_values.shape
        yabs = np.zeros((batch_size, max_patches), dtype=np.int32)
        xabs = np.zeros((batch_size, max_patches), dtype=np.int32)

        for i in range(batch_size):
            h, w = spatial_shapes[i]
            yabs[i, :h * w] = np.repeat(np.arange(h), w)
            xabs[i, :h * w] = np.tile(np.arange(w), h)

        return (
            pixel_values,
            ptypes,
            yabs,
            xabs,
        )

    # ======================================================
    # IMAGE API
    # ======================================================
    def encode_images(self, images):
        patches, ptype, yabs, xabs = self.images_to_naflex(images)

        hidden = self._encode_image_jit(
            jnp.asarray(patches), 
            jnp.asarray(ptype), 
            jnp.asarray(yabs), 
            jnp.asarray(xabs)
        )

        return np.asarray(hidden)

    def encode_image(self, image):
        return self.encode_images([image])[0]

    # ======================================================
    # TEXT API
    # ======================================================
    def encode_texts(self, texts):
        if isinstance(texts, str) or (
            isinstance(texts, list) and isinstance(texts[0], str)
        ):
            batch = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="np",
            )
            input_ids = batch["input_ids"]
        else:
            input_ids = texts

        hidden = self._encode_text_jit(jnp.asarray(input_ids))
        return np.asarray(hidden)

    def encode_text(self, text):
        return self.encode_texts([text])[0]