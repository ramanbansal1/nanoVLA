import argparse
from dataclasses import dataclass, field, asdict

@dataclass
class TrainConfig:
    datasets_root: str = "data/datasets"
    action_horizon: int = 8
    batch_size: int = 100
    num_workers: int = 4
    hidden_size: int = 192
    learning_rate: float = 1e-4
    dummy_vlm: bool = False
    wandb_project: str = "nanoVLA"
    jax_mem_fraction: str = ".70"
    num_epochs: int = 30
    save_every: int = 100
    checkpoint_dir: str = "checkpoints/nanoVLA"
    dit_num_blocks: int = 4
    vla_k: int = 4
    vlm_context_dir: str = "data/vlm_context"

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="nanoVLA Training Configuration")
    
    # We use the defaults from the dataclass
    default_config = TrainConfig()
    
    parser.add_argument("--datasets_root", type=str, default=default_config.datasets_root,
                        help=f"Path to datasets root directory (default: {default_config.datasets_root})")
    parser.add_argument("--action_horizon", type=int, default=default_config.action_horizon,
                        help=f"Action horizon (default: {default_config.action_horizon})")
    parser.add_argument("--batch_size", type=int, default=default_config.batch_size,
                        help=f"Batch size (default: {default_config.batch_size})")
    parser.add_argument("--num_workers", type=int, default=default_config.num_workers,
                        help=f"Number of dataloader workers (default: {default_config.num_workers})")
    parser.add_argument("--hidden_size", type=int, default=default_config.hidden_size,
                        help=f"Hidden size for models (default: {default_config.hidden_size})")
    parser.add_argument("--learning_rate", type=float, default=default_config.learning_rate,
                        help=f"Learning rate (default: {default_config.learning_rate})")
    
    # Booleans are slightly tricky with argparse, but we can use simple string matching or store_true/store_false
    parser.add_argument("--dummy_vlm", type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=default_config.dummy_vlm,
                        help=f"Use dummy VLM for fast testing (default: {default_config.dummy_vlm})")
    
    parser.add_argument("--wandb_project", type=str, default=default_config.wandb_project,
                        help=f"W&B project name (default: {default_config.wandb_project})")
    
    parser.add_argument("--jax_mem_fraction", type=str, default=default_config.jax_mem_fraction,
                        help=f"XLA_PYTHON_CLIENT_MEM_FRACTION (default: {default_config.jax_mem_fraction})")
    
    parser.add_argument("--num_epochs", type=int, default=default_config.num_epochs,
                        help=f"Number of training epochs (default: {default_config.num_epochs})")
    parser.add_argument("--save_every", type=int, default=default_config.save_every,
                        help=f"Save checkpoint every N steps (default: {default_config.save_every})")
    parser.add_argument("--checkpoint_dir", type=str, default=default_config.checkpoint_dir,
                        help=f"Directory to save checkpoints (default: {default_config.checkpoint_dir})")
    parser.add_argument("--dit_num_blocks", type=int, default=default_config.dit_num_blocks,
                        help=f"Number of DiT blocks (default: {default_config.dit_num_blocks})")
    parser.add_argument("--vla_k", type=int, default=default_config.vla_k,
                        help=f"Number of flow matching iterations K (default: {default_config.vla_k})")
    parser.add_argument("--vlm_context_dir", type=str, default=default_config.vlm_context_dir,
                        help=f"Directory containing precomputed VLM embeddings (default: {default_config.vlm_context_dir})")
    
    args = parser.parse_args()
    
    return TrainConfig(
        datasets_root=args.datasets_root,
        action_horizon=args.action_horizon,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        dummy_vlm=args.dummy_vlm,
        wandb_project=args.wandb_project,
        jax_mem_fraction=args.jax_mem_fraction,
        num_epochs=args.num_epochs,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        dit_num_blocks=args.dit_num_blocks,
        vla_k=args.vla_k,
        vlm_context_dir=args.vlm_context_dir
    )
