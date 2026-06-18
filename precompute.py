"""
precompute_fast.py
------------------
Blazingly fast multi-GPU VLM precomputation.

Key optimisations
─────────────────
1. jax.pmap   – shards every batch across ALL visible GPUs in one call.
2. CPU prefetch pipeline – a background ThreadPoolExecutor builds the next
   batch while the GPUs are busy with the current one (double-buffering).
3. Async npz saves – a separate executor writes compressed files so disk I/O
   never stalls the GPU pipeline.
4. One-shot text tokenisation – the instruction string is the same for every
   step in an episode; we tokenise once and broadcast.
5. Pad-to-multiple batching – guarantees every pmap batch is a multiple of
   n_devices so we never waste a partial replicated batch.
"""

import os
import time
import queue
import threading
import concurrent.futures
from pathlib import Path

import wandb
import numpy as np
import jax
import jax.numpy as jnp
from datasets import load_dataset, concatenate_datasets
from tqdm.auto import tqdm

from config import parse_args
from models.visual_encoder import SigLIP
from data.advanced_dataset import VideoDataset


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def pad_to_multiple(arrays: list[np.ndarray], n: int) -> tuple[np.ndarray, int]:
    """Pad a list of arrays along axis-0 to the next multiple of *n*.

    Returns (padded_array, original_length).
    """
    real_len = len(arrays)
    stacked   = np.stack(arrays, axis=0)          # (B, ...)
    remainder = real_len % n
    if remainder != 0:
        pad_n    = n - remainder
        pad_tile = np.stack([arrays[-1]] * pad_n)  # repeat last row
        stacked  = np.concatenate([stacked, pad_tile], axis=0)
    return stacked, real_len


def shard(x: np.ndarray, n_devices: int) -> np.ndarray:
    """Reshape (B, ...) → (n_devices, B//n_devices, ...) for pmap."""
    return x.reshape((n_devices, x.shape[0] // n_devices) + x.shape[1:])


def unshard(x: np.ndarray) -> np.ndarray:
    """Collapse (n_devices, local_B, ...) → (B, ...)."""
    return x.reshape((-1,) + x.shape[2:])


# ──────────────────────────────────────────────────────────────────────────────
# pmap-compatible encode functions
# (these are module-level so jax.pmap can find them cleanly)
# ──────────────────────────────────────────────────────────────────────────────

_IMG_MODEL   = None   # set in main() after SigLIP is built
_TXT_MODEL   = None
_IMG_PARAMS  = None
_TXT_PARAMS  = None


def _pmap_encode_image(patches, ptype, yabs, xabs):
    """Each device receives its local shard; returns (local_B, Npatches, D)."""
    _, out = _IMG_MODEL.apply(
        {"params": _IMG_PARAMS},
        (patches, ptype, yabs, xabs),
        train=False,
    )
    return out["encoded"]


def _pmap_encode_text(input_ids):
    """Each device receives its local shard; returns (local_B, T, D)."""
    _, out = _TXT_MODEL.apply(
        {"params": _TXT_PARAMS},
        input_ids,
        train=False,
    )
    return out["transformed"]


# ──────────────────────────────────────────────────────────────────────────────
# Batch processor  (runs on CPU thread, submits to GPU)
# ──────────────────────────────────────────────────────────────────────────────

class MultiGPUEncoder:
    """Wraps pmap-compiled image+text encoders."""

    def __init__(self, vlm: SigLIP, per_gpu_bs: int):
        self.vlm       = vlm
        self.n_devices = jax.device_count()
        self.per_gpu_bs = per_gpu_bs
        self.global_bs = self.n_devices * per_gpu_bs
        print(f"[MultiGPUEncoder] Detected {self.n_devices} device(s): "
              f"{[str(d) for d in jax.devices()]}")

        # Set module-level references so pmap lambdas can reach them
        global _IMG_MODEL, _TXT_MODEL, _IMG_PARAMS, _TXT_PARAMS
        _IMG_MODEL  = vlm.image_model
        _TXT_MODEL  = vlm.text_model
        _IMG_PARAMS = vlm.img_params
        _TXT_PARAMS = vlm.txt_params

        # Replicate params across devices once (handled implicitly by jax.pmap capturing globals)

        # Compile pmap kernels
        self._pmap_img = jax.pmap(_pmap_encode_image, axis_name="batch")
        self._pmap_txt = jax.pmap(_pmap_encode_text,  axis_name="batch")

        # Warm up with dummy data so first real call has no compile latency
        self._warmup(vlm, per_gpu_bs)

    # ------------------------------------------------------------------
    def _warmup(self, vlm: SigLIP, per_gpu_bs: int):
        print(f"[MultiGPUEncoder] Starting JIT compilation (pmap warm-up) for local batch {per_gpu_bs}...")
        t0 = time.perf_counter()
        n = self.n_devices
        dummy_patches = jnp.zeros((n, per_gpu_bs, 256, 768), dtype=jnp.float32)
        dummy_ptype   = jnp.ones ((n, per_gpu_bs, 256),      dtype=jnp.int32)
        dummy_yabs    = jnp.zeros((n, per_gpu_bs, 256),      dtype=jnp.int32)
        dummy_xabs    = jnp.zeros((n, per_gpu_bs, 256),      dtype=jnp.int32)
        dummy_ids     = jnp.zeros((n, per_gpu_bs, vlm.max_text_length), dtype=jnp.int32)

        r = self._pmap_img(dummy_patches, dummy_ptype, dummy_yabs, dummy_xabs)
        r.block_until_ready()
        r = self._pmap_txt(dummy_ids)
        r.block_until_ready()
        print(f"[MultiGPUEncoder] pmap warm-up done in {time.perf_counter() - t0:.2f}s.")

    # ------------------------------------------------------------------
    def encode_batch(
        self,
        patches: np.ndarray,
        ptype: np.ndarray,
        yabs: np.ndarray,
        xabs: np.ndarray,
        input_ids: np.ndarray,
    ) -> np.ndarray:              # (B, T+Npatches, D)
        """
        Encode a batch of images + texts using ALL GPUs via pmap.

        The batch is padded to exactly global_bs, sharded, encoded,
        unsharded, and trimmed back to the original batch size.
        """
        n = self.n_devices
        B_orig = patches.shape[0]

        def pad_shard(arr_np):
            if B_orig < self.global_bs:
                pad_n  = self.global_bs - B_orig
                pad    = np.repeat(arr_np[-1:], pad_n, axis=0)
                arr_np = np.concatenate([arr_np, pad], axis=0)
            elif B_orig > self.global_bs:
                raise ValueError(f"Batch {B_orig} > global {self.global_bs}")
            return shard(arr_np, n)

        s_patches = jnp.asarray(pad_shard(patches))
        s_ptype   = jnp.asarray(pad_shard(ptype))
        s_yabs    = jnp.asarray(pad_shard(yabs))
        s_xabs    = jnp.asarray(pad_shard(xabs))

        # input_ids: pad text batch same way
        ids_np    = np.asarray(input_ids)
        remainder = B_orig % n
        if remainder:
            pad_n  = n - remainder
            ids_np = np.concatenate([ids_np, np.repeat(ids_np[-1:], pad_n, axis=0)], axis=0)
        s_ids = jnp.asarray(shard(ids_np, n))

        # ── pmap encode ───────────────────────────────────────────────
        img_hidden = self._pmap_img(s_patches, s_ptype, s_yabs, s_xabs)
        txt_hidden = self._pmap_txt(s_ids)

        img_hidden.block_until_ready()
        txt_hidden.block_until_ready()

        # Unshard: (n_devices, per_gpu_bs, ...) -> (global_bs, ...)
        img_np = np.asarray(img_hidden).reshape(-1, img_hidden.shape[-2], img_hidden.shape[-1])
        txt_np = np.asarray(txt_hidden).reshape(-1, txt_hidden.shape[-2], txt_hidden.shape[-1])

        # Free JAX device buffers immediately to prevent memory ballooning
        img_hidden.delete()
        txt_hidden.delete()
        s_patches.delete()
        s_ptype.delete()
        s_yabs.delete()
        s_xabs.delete()
        s_ids.delete()

        # Concatenate on the text dim (B, max_text_length + max_patches, D)
        out = np.concatenate([txt_np, img_np], axis=1)

        # Trim back to B_orig
        return out[:B_orig]


# ──────────────────────────────────────────────────────────────────────────────
# Async save helper
# ──────────────────────────────────────────────────────────────────────────────

class AsyncSaver:
    """Saves npz files in a background thread so the GPU never waits on disk."""

    def __init__(self, max_workers: int = 2):
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._futures: list[concurrent.futures.Future] = []

    def save(self, path: Path, vlm_out: np.ndarray):
        fut = self._pool.submit(np.savez_compressed, path, vlm_out=vlm_out)
        self._futures.append(fut)
        # Opportunistically collect completed futures
        self._futures = [f for f in self._futures if not f.done()]

    def flush(self):
        """Wait for all pending saves to complete."""
        concurrent.futures.wait(self._futures)
        self._futures.clear()

    def __enter__(self):  return self
    def __exit__(self, *_): self.flush(); self._pool.shutdown(wait=True)


from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor

# ──────────────────────────────────────────────────────────────────────────────
# Prefetch pipeline via PyTorch DataLoader (Multi-Process)
# ──────────────────────────────────────────────────────────────────────────────

class EpisodePrecomputeDataset(Dataset):
    def __init__(self, video_dataset, ep_items, global_bs):
        self.video_dataset = video_dataset
        self.ep_items = ep_items
        self.global_bs = global_bs
        self.processor = None

    def __len__(self):
        return len(self.ep_items)

    def _images_to_naflex(self, images):
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-naflex")
            
        out = self.processor(images=images, return_tensors="pt", padding=True)
        pixel_values = out["pixel_values"].numpy()
        ptypes = out["pixel_attention_mask"].numpy().astype(np.int32)
        spatial_shapes = out["spatial_shapes"].numpy()

        batch_size, max_patches, patch_dim = pixel_values.shape
        MAX_PATCHES = 256

        if max_patches > MAX_PATCHES:
            pixel_values = pixel_values[:, :MAX_PATCHES, :]
            ptypes = ptypes[:, :MAX_PATCHES]
            max_patches = MAX_PATCHES
        elif max_patches < MAX_PATCHES:
            pad_len = MAX_PATCHES - max_patches
            pixel_values = np.pad(pixel_values, ((0,0), (0, pad_len), (0,0)), mode='constant')
            ptypes = np.pad(ptypes, ((0,0), (0, pad_len)), mode='constant')
            max_patches = MAX_PATCHES

        yabs = np.zeros((batch_size, max_patches), dtype=np.int32)
        xabs = np.zeros((batch_size, max_patches), dtype=np.int32)

        for i in range(batch_size):
            h, w = spatial_shapes[i]
            valid_len = min(h * w, max_patches)
            yabs[i, :valid_len] = np.repeat(np.arange(h), w)[:valid_len]
            xabs[i, :valid_len] = np.tile(np.arange(w), h)[:valid_len]

        return pixel_values, ptypes, yabs, xabs

    def __getitem__(self, idx):
        ep_key, (ep_start, ep_end) = self.ep_items[idx]
        repo_name = self.video_dataset.dataset[ep_start]["dataset_name"]
        ep_id = ep_key[1] if isinstance(ep_key, tuple) else ep_key

        images_acc = []
        ids_acc = []
        batches = []

        def flush():
            if images_acc:
                patches, ptype, yabs, xabs = self._images_to_naflex(images_acc)
                batches.append((
                    patches, ptype, yabs, xabs,
                    np.stack(ids_acc, axis=0)
                ))
                images_acc.clear()
                ids_acc.clear()

        for i in range(ep_start, ep_end + 1):
            data = self.video_dataset[i]
            img_dict = data["images"]
            pil = img_dict.get("default_cam", next(iter(img_dict.values())))
            images_acc.append(pil)
            ids_acc.append(np.asarray(data["input_ids"]))

            if len(images_acc) == self.global_bs:
                flush()

        flush()
        return ep_id, repo_name, batches


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    config = parse_args()
    wandb.init(project="nanovla-precompute", config=config)

    precompute_path = Path(config.precompute_path)
    precompute_path.mkdir(parents=True, exist_ok=True)

    # ── Load SigLIP ────────────────────────────────────────────────────────
    print("Loading VLM Model …")
    vlm = SigLIP(checkpoint_path=config.vlm_checkpoint_path, normalize=True)

    # Batch size per pmap call – scale with number of GPUs
    # Use config.batch_size as the *per-GPU* batch size
    per_gpu_bs   = config.batch_size

    # ── Wrap with multi-GPU encoder ────────────────────────────────────────
    encoder = MultiGPUEncoder(vlm, per_gpu_bs)
    n_dev   = encoder.n_devices
    global_bs    = per_gpu_bs * n_dev

    # ── Load datasets ──────────────────────────────────────────────────────
    print("Loading datasets …")
    datasets_root_dir = Path(config.datasets_root)
    dataset_repos     = sorted([p for p in datasets_root_dir.iterdir() if p.is_dir()])

    if not dataset_repos:
        print(f"No datasets found in {config.datasets_root}")
        return

    all_datasets = []
    for repo_path in dataset_repos:
        repo_name = repo_path.name
        print(f"  Loading: {repo_name}")
        ds = load_dataset(str(repo_path))["train"]
        ds = ds.add_column("dataset_name", [repo_name] * len(ds))
        all_datasets.append(ds)

    combined_hf_dataset = concatenate_datasets(all_datasets)
    print(f"Combined dataset: {len(combined_hf_dataset):,} rows")

    video_dataset = VideoDataset(
        dataset=combined_hf_dataset,
        datasets_root=config.datasets_root,
        action_horizon=config.action_horizon,
        precompute_path=None,
    )

    episodes = video_dataset.episode_ranges
    print(f"Episodes to precompute: {len(episodes):,}")

    # Filter out already-done episodes
    todo_episodes = {}
    for ep_key, rng in episodes.items():
        repo_name = combined_hf_dataset[rng[0]]["dataset_name"]
        ep_id = ep_key[1] if isinstance(ep_key, tuple) else ep_key
        
        if not (precompute_path / repo_name / f"ep_{ep_id}.npz").exists():
            todo_episodes[ep_key] = rng

    print(f"Episodes remaining: {len(todo_episodes):,}  "
          f"(skipping {len(episodes) - len(todo_episodes):,} already done)")

    if not todo_episodes:
        print("All episodes already precomputed. Nothing to do.")
        return

    ep_items   = list(todo_episodes.items())
    total_eps  = len(ep_items)

    ep_dataset = EpisodePrecomputeDataset(video_dataset, ep_items, global_bs)
    dataloader = DataLoader(
        ep_dataset,
        batch_size=1,
        num_workers=8,
        prefetch_factor=2,
        shuffle=False,
        collate_fn=lambda x: x[0],
    )

    t0 = time.perf_counter()
    steps_done = 0

    total_frames = sum(rng[1] - rng[0] + 1 for rng in todo_episodes.values())
    
    with AsyncSaver(max_workers=2) as saver, \
         tqdm(total=total_eps, desc="Episodes", unit="ep", position=0) as pbar_ep, \
         tqdm(total=total_frames, desc="Frames", unit="fr", position=1) as pbar_fr:

        for ep_id, repo_name, batches in dataloader:

            save_dir  = precompute_path / repo_name
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"ep_{ep_id}.npz"

            ep_vlm_outs = []
            for patches, ptype, yabs, xabs, input_ids in batches:
                vlm_out = encoder.encode_batch(patches, ptype, yabs, xabs, input_ids)
                ep_vlm_outs.append(vlm_out)
                num_frames = patches.shape[0]
                steps_done += num_frames
                pbar_fr.update(num_frames)

            episode_vlm_out = np.concatenate(ep_vlm_outs, axis=0)
            saver.save(save_path, episode_vlm_out)

            elapsed = time.perf_counter() - t0
            pbar_ep.set_postfix(
                steps_s=f"{steps_done / elapsed:.1f}",
                repo=repo_name[:20],
            )
            pbar_ep.update(1)
            
            wandb.log({
                "steps": steps_done,
                "steps_per_sec": steps_done / elapsed,
                "episodes_done": pbar_ep.n,
            })

    wandb.finish()
    print(f"\nDone — {steps_done:,} steps in {time.perf_counter() - t0:.1f}s "
          f"({steps_done / (time.perf_counter() - t0):.1f} steps/s)")


if __name__ == "__main__":
    main()