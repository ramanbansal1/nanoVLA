# nanoVLA

nanoVLA is a fast, scalable Vision-Language-Action (VLA) model implementation optimized for training on GPU architectures. It leverages JAX and Flax NNX for the DiT architecture and combines it with PyTorch-based VLMs for robust feature extraction.

## Features
- **Flow Matching Latent Diffusion**: Uses DiT architecture to denoise and predict actions.
- **Flax NNX**: Built with the latest Flax neural network API for intuitive module state management.
- **JIT-Compiled Training Step**: The entire core DiT and linear projection training step is `jax.jit` compiled for maximum GPU utilization.
- **Gradient Checkpointing**: Uses `nnx.remat` on DiT blocks to scale to large sequence lengths efficiently.
- **Hybrid JAX/PyTorch Support**: Carefully manages GPU memory via `XLA_PYTHON_CLIENT_MEM_FRACTION` to allow Hugging Face Transformers and JAX to coexist.

## Setup Instructions

### Prerequisites
Make sure you have a CUDA 12 compatible environment. The project uses `uv` for lightning-fast dependency management.

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd nanoVLA
   ```

2. **Install dependencies:**
   Using `uv` is recommended to automatically sync the exact dependencies from `uv.lock`.
   ```bash
   uv sync
   ```
   *Alternatively, using pip:*
   ```bash
   pip install -e .
   ```

3. **Verify JAX GPU Installation:**
   Check that JAX can see your GPU devices:
   ```bash
   python -c "import jax; print(jax.devices())"
   ```

### Data Preparation

Data is handled natively by huggingface datasets. The expected structure is a directory containing huggingface datasets, for example:
```
data/
└── datasets/
    ├── dataset_1/
    └── dataset_2/
```

### Running Training

To launch training, simply run:

```bash
python train.py --batch_size 128 --hidden_size 192 --wandb_project nanoVLA
```

You can customize the memory split between JAX and PyTorch in `config.py` using `--jax_mem_fraction`. By default, it allocates 70% of the GPU memory to JAX (`.70`).

```bash
python train.py --jax_mem_fraction .75
```

### Key Optimizations 🚀

- **Precomputed VLM Features:** We bypass tracer errors and keep PyTorch execution out of the critical path by precomputing VLM embeddings right before the JAX JIT compiled step.
- **Asynchronous Execution:** Output metrics are synchronized using `jax.block_until_ready()`, preventing async dispatches from queueing indefinitely and masking OOM errors.
- **Gradient Checkpointing (`nnx.remat`):** Retains peak memory capacity during training, allowing larger batch sizes.
