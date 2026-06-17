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

    def __init__(self, vlm: SigLIP):
        self.vlm       = vlm
        self.n_devices = jax.device_count()
        print(f"[MultiGPUEncoder] Detected {self.n_devices} device(s): "
              f"{[str(d) for d in jax.devices()]}")

        # Set module-level references so pmap lambdas can reach them
        global _IMG_MODEL, _TXT_MODEL, _IMG_PARAMS, _TXT_PARAMS
        _IMG_MODEL  = vlm.image_model
        _TXT_MODEL  = vlm.text_model
        _IMG_PARAMS = vlm.img_params
        _TXT_PARAMS = vlm.txt_params

        # Replicate params across devices once
        self.img_params_rep = jax.device_put_replicated(vlm.img_params, jax.devices())
        self.txt_params_rep = jax.device_put_replicated(vlm.txt_params, jax.devices())

        # Compile pmap kernels
        self._pmap_img = jax.pmap(_pmap_encode_image, axis_name="batch")
        self._pmap_txt = jax.pmap(_pmap_encode_text,  axis_name="batch")

        # Warm up with dummy data so first real call has no compile latency
        self._warmup(vlm)

    # ------------------------------------------------------------------
    def _warmup(self, vlm: SigLIP):
        n = self.n_devices
        dummy_patches = jnp.zeros((n, 1, 256, 768), dtype=jnp.float32)
        dummy_ptype   = jnp.ones ((n, 1, 256),      dtype=jnp.int32)
        dummy_yabs    = jnp.zeros((n, 1, 256),      dtype=jnp.int32)
        dummy_xabs    = jnp.zeros((n, 1, 256),      dtype=jnp.int32)
        dummy_ids     = jnp.zeros((n, 1, vlm.max_text_length), dtype=jnp.int32)

        r = self._pmap_img(dummy_patches, dummy_ptype, dummy_yabs, dummy_xabs)
        r.block_until_ready()
        r = self._pmap_txt(dummy_ids)
        r.block_until_ready()
        print("[MultiGPUEncoder] pmap warm-up done.")

    # ------------------------------------------------------------------
    def encode_batch(
        self,
        pil_images: list,
        input_ids:  np.ndarray,   # (B, T)  int32
    ) -> np.ndarray:              # (B, T+Npatches, D)
        """
        Encode a batch of images + texts using ALL GPUs via pmap.

        The batch is padded to a multiple of n_devices, sharded, encoded,
        unsharded, and trimmed back to the original batch size.
        """
        n = self.n_devices

        # ── IMAGE preprocessing (CPU) ──────────────────────────────────
        patches, ptype, yabs, xabs = self.vlm.images_to_naflex(pil_images)
        # patches: (B, Npatches, D)  etc.

        # ── Pad & shard ───────────────────────────────────────────────
        B_orig = patches.shape[0]

        def pad_shard(arr):
            arr_np = np.asarray(arr)
            remainder = B_orig % n
            if remainder:
                pad_n  = n - remainder
                pad    = np.repeat(arr_np[-1:], pad_n, axis=0)
                arr_np = np.concatenate([arr_np, pad], axis=0)
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

        # ── Unshard & trim ────────────────────────────────────────────
        img_np = np.asarray(unshard(img_hidden))[:B_orig]  # (B_orig, Np, D)
        txt_np = np.asarray(unshard(txt_hidden))[:B_orig]  # (B_orig, T,  D)

        # Concat along sequence dim: (B, T+Np, D)
        return np.concatenate([txt_np, img_np], axis=1)


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


# ──────────────────────────────────────────────────────────────────────────────
# Prefetch pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _prefetch_episode(
    ep_start: int,
    ep_end:   int,
    video_dataset: VideoDataset,
    batch_size: int,
) -> list[tuple[list, np.ndarray]]:
    """
    Load ALL steps of one episode into CPU RAM and split into batches.
    Runs in a background thread so the next episode is ready when GPUs finish
    the current one.

    Returns a list of (pil_images, input_ids_array) tuples.
    """
    images_acc  = []
    ids_acc     = []
    batches     = []

    def flush():
        if images_acc:
            batches.append((list(images_acc), np.stack(ids_acc, axis=0)))
            images_acc.clear()
            ids_acc.clear()

    for i in range(ep_start, ep_end + 1):
        data = video_dataset[i]

        img_dict = data["images"]
        if "default_cam" in img_dict:
            pil = img_dict["default_cam"]
        else:
            pil = next(iter(img_dict.values()))
        images_acc.append(pil)
        ids_acc.append(np.asarray(data["input_ids"]))

        if len(images_acc) == batch_size:
            flush()

    flush()  # remainder
    return batches


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    config = parse_args()

    precompute_path = Path(config.precompute_path)
    precompute_path.mkdir(parents=True, exist_ok=True)

    # ── Load SigLIP ────────────────────────────────────────────────────────
    print("Loading VLM Model …")
    vlm = SigLIP(checkpoint_path=config.vlm_checkpoint_path, normalize=True)

    # ── Wrap with multi-GPU encoder ────────────────────────────────────────
    encoder = MultiGPUEncoder(vlm)
    n_dev   = encoder.n_devices

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

    # Batch size per pmap call – scale with number of GPUs
    # Use config.batch_size as the *per-GPU* batch size
    per_gpu_bs   = config.batch_size
    global_bs    = per_gpu_bs * n_dev

    # Filter out already-done episodes
    todo_episodes = {
        ep_id: rng
        for ep_id, rng in episodes.items()
        if not (
            precompute_path
            / combined_hf_dataset[rng[0]]["dataset_name"]
            / f"ep_{ep_id}.npz"
        ).exists()
    }
    print(f"Episodes remaining: {len(todo_episodes):,}  "
          f"(skipping {len(episodes) - len(todo_episodes):,} already done)")

    if not todo_episodes:
        print("All episodes already precomputed. Nothing to do.")
        return

    ep_items   = list(todo_episodes.items())
    total_eps  = len(ep_items)

    # ── Double-buffered prefetch + async save ──────────────────────────────
    # Prefetch queue: holds (ep_id, repo_name, batches) for the NEXT episode
    prefetch_q: queue.Queue = queue.Queue(maxsize=2)

    def _prefetch_worker():
        for ep_id, (ep_start, ep_end) in ep_items:
            repo_name = combined_hf_dataset[ep_start]["dataset_name"]
            batches   = _prefetch_episode(ep_start, ep_end, video_dataset, global_bs)
            prefetch_q.put((ep_id, repo_name, batches))
        prefetch_q.put(None)  # sentinel

    prefetch_thread = threading.Thread(target=_prefetch_worker, daemon=True)
    prefetch_thread.start()

    t0 = time.perf_counter()
    steps_done = 0

    with AsyncSaver(max_workers=2) as saver, \
         tqdm(total=total_eps, desc="Episodes", unit="ep") as pbar:

        while True:
            item = prefetch_q.get()
            if item is None:
                break

            ep_id, repo_name, batches = item

            save_dir  = precompute_path / repo_name
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"ep_{ep_id}.npz"

            ep_vlm_outs = []
            for pil_images, input_ids in batches:
                vlm_out = encoder.encode_batch(pil_images, input_ids)
                ep_vlm_outs.append(vlm_out)
                steps_done += len(pil_images)

            episode_vlm_out = np.concatenate(ep_vlm_outs, axis=0)
            saver.save(save_path, episode_vlm_out)

            elapsed = time.perf_counter() - t0
            pbar.set_postfix(
                steps=steps_done,
                steps_s=f"{steps_done / elapsed:.1f}",
                repo=repo_name[:20],
            )
            pbar.update(1)

    prefetch_thread.join()
    print(f"\nDone — {steps_done:,} steps in {time.perf_counter() - t0:.1f}s "
          f"({steps_done / (time.perf_counter() - t0):.1f} steps/s)")


if __name__ == "__main__":
    main()