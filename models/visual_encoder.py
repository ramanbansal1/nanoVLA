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

    Image embedding:
        PIL.Image -> (768,)

    Text embedding:
        str -> (768,)

    Similarity:
        cosine similarity via dot product
        (embeddings are normalized)
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

        # ==================================================
        # IMAGE MODEL
        # ==================================================

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

        self.image_model.init(
            jax.random.PRNGKey(0),
            image_dummy,
            train=False,
        )

        # ==================================================
        # TEXT MODEL
        # ==================================================

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

        text_dummy = jnp.zeros(
            (1, max_text_length),
            dtype=jnp.int32,
        )

        self.text_model.init(
            jax.random.PRNGKey(0),
            text_dummy,
            train=False,
        )

        # ==================================================
        # JIT COMPILE
        # ==================================================

        self._encode_image_jit = jax.jit(
            self._encode_image_impl
        )

        self._encode_text_jit = jax.jit(
            self._encode_text_impl
        )

        self._encode_image_jit(
            *image_dummy
        ).block_until_ready()

        self._encode_text_jit(
            text_dummy
        ).block_until_ready()

    # ======================================================
    # INTERNAL IMAGE ENCODER
    # ======================================================

    def _encode_image_impl(
        self,
        patches,
        ptype,
        yabs,
        xabs,
    ):
        emb, _ = self.image_model.apply(
            {"params": self.img_params},
            (patches, ptype, yabs, xabs),
            train=False,
        )

        if self.normalize:
            emb = emb / jnp.linalg.norm(
                emb,
                axis=-1,
                keepdims=True,
            )

        return emb

    # ======================================================
    # INTERNAL TEXT ENCODER
    # ======================================================

    def _encode_text_impl(self, input_ids):
        emb, _ = self.text_model.apply(
            {"params": self.txt_params},
            input_ids,
            train=False,
        )

        if self.normalize:
            emb = emb / jnp.linalg.norm(
                emb,
                axis=-1,
                keepdims=True,
            )

        return emb

    # ======================================================
    # IMAGE PREPROCESSING
    # ======================================================

    @staticmethod
    def image_to_naflex(image):
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            image = image.resize(
                (256, 256),
                Image.BICUBIC,
            )
            image = np.asarray(
                image,
                dtype=np.float32,
            )
        else:
            image = np.asarray(
                image,
                dtype=np.float32,
            )

            if image.shape[:2] != (256, 256):
                image = np.asarray(
                    Image.fromarray(
                        image.astype(np.uint8)
                    ).resize(
                        (256, 256),
                        Image.BICUBIC,
                    ),
                    dtype=np.float32,
                )

        image /= 255.0

        patches = (
            image.reshape(
                16, 16,
                16, 16,
                3,
            )
            .transpose(
                0, 2, 1, 3, 4
            )
            .reshape(
                256,
                768,
            )
        )

        yabs = np.repeat(
            np.arange(16),
            16,
        )

        xabs = np.tile(
            np.arange(16),
            16,
        )

        ptype = np.ones(
            256,
            dtype=np.int32,
        )

        return (
            patches.astype(np.float32),
            ptype,
            yabs.astype(np.int32),
            xabs.astype(np.int32),
        )

    def images_to_naflex(self, images):
        patches = []
        ptypes = []
        yabs = []
        xabs = []

        for image in images:
            p, pt, y, x = self.image_to_naflex(image)

            patches.append(p)
            ptypes.append(pt)
            yabs.append(y)
            xabs.append(x)

        return (
            jnp.asarray(patches),
            jnp.asarray(ptypes),
            jnp.asarray(yabs),
            jnp.asarray(xabs),
        )

    # ======================================================
    # IMAGE API
    # ======================================================

    def encode_image(self, image):
        return self.encode_images([image])[0]

    def encode_images(self, images):
        patches, ptype, yabs, xabs = (
            self.images_to_naflex(images)
        )

        emb = self._encode_image_jit(
            patches,
            ptype,
            yabs,
            xabs,
        )

        return np.asarray(emb)

    # ======================================================
    # TEXT API
    # ======================================================

    def encode_text(self, text):
        return self.encode_texts([text])[0]

    def encode_texts(self, texts):
        if isinstance(texts, str) or (isinstance(texts, list) and isinstance(texts[0], str)):
            batch = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="np",
            )
            input_ids = batch["input_ids"]
        else:
            # Assume it's already an array of input_ids
            input_ids = texts

        emb = self._encode_text_jit(
            jnp.asarray(input_ids)
        )

        return np.asarray(emb)

    # ======================================================
    # SIMILARITY
    # ======================================================

    @staticmethod
    def similarity(
        image_embeddings,
        text_embeddings,
    ):
        return image_embeddings @ text_embeddings.T