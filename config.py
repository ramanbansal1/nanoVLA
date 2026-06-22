import argparse
from dataclasses import dataclass, field, asdict

@dataclass
class TrainConfig:
    datasets_root: str = "data/datasets"
    action_horizon: int = 25
    batch_size: int = 100
    num_workers: int = 4
    hidden_size: int = 96
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    lambda_recon: float = 0.05
    wandb_project: str = "nanoVLA"
    num_epochs: int = 30
    save_every: int = 100
    checkpoint_dir: str = "checkpoints/nanoVLA"
    dit_num_blocks: int = 4
    vla_k: int = 4
    vlm_checkpoint_path: str = "checkpoints/siglip2_naflex.npz"
    precompute_path: str = "data/precomputed_vlm"

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
    parser.add_argument("--weight_decay", type=float, default=default_config.weight_decay,
                        help=f"Weight decay (default: {default_config.weight_decay})")
    parser.add_argument("--lambda_recon", type=float, default=default_config.lambda_recon,
                        help=f"Reconstruction loss weight (default: {default_config.lambda_recon})")
    
    parser.add_argument("--wandb_project", type=str, default=default_config.wandb_project,
                        help=f"W&B project name (default: {default_config.wandb_project})")
    
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
    parser.add_argument("--vlm_checkpoint_path", type=str, default=default_config.vlm_checkpoint_path,
                        help=f"Path to VLM checkpoint (default: {default_config.vlm_checkpoint_path})")
    parser.add_argument("--precompute_path", type=str, default=default_config.precompute_path,
                        help=f"Path to save precomputed VLM embeddings (default: {default_config.precompute_path})")
    
    args = parser.parse_args()
    
    return TrainConfig(
        datasets_root=args.datasets_root,
        action_horizon=args.action_horizon,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lambda_recon=args.lambda_recon,
        wandb_project=args.wandb_project,
        num_epochs=args.num_epochs,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        dit_num_blocks=args.dit_num_blocks,
        vla_k=args.vla_k,
        vlm_checkpoint_path=args.vlm_checkpoint_path,
        precompute_path=args.precompute_path
    )
