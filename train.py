#!.venv/bin/python
import os
import time
from rich.console import Console
console = Console()

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import torch
import torch.utils.dlpack
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from datasets import load_dataset, concatenate_datasets
from pathlib import Path
import wandb
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flax import nnx
import optax
import orbax.checkpoint as ocp
from jax.sharding import Mesh, PartitionSpec, NamedSharding

from config import parse_args, asdict
from data.advanced_dataset import VideoDataset
from models.vla import VLA


def custom_collate_fn(batch):
    collated = {}

    keys = set()
    for item in batch:
        keys.update(item.keys())

    for key in keys:
        if key == "images":
            imgs = []
            for item in batch:
                if "images" in item and len(item["images"]) > 0:
                    first_cam = sorted(item["images"].keys())[0]
                    imgs.append(np.array(item["images"][first_cam], copy=True))
                else:
                    imgs.append(np.zeros((224, 224, 3), dtype=np.uint8))
            collated["image"] = default_collate(imgs)
        elif key == "input_ids":
            vals = []
            for item in batch:
                if "input_ids" in item:
                    vals.append(np.array(item["input_ids"], copy=True))
                else:
                    vals.append(np.zeros((64,), dtype=np.int32))
            collated[key] = default_collate(vals)
        elif key == "vlm_out":
            vals = []
            for item in batch:
                if "vlm_out" in item:
                    vals.append(np.array(item["vlm_out"], copy=True))
                else:
                    vals.append(np.zeros((256 + 64, 768), dtype=np.float32))
            collated[key] = default_collate(vals)
        else:
            vals = []
            for item in batch:
                if key in item:
                    vals.append(item[key])
                else:
                    # Provide a dummy value of the same type as the first available
                    for ref_item in batch:
                        if key in ref_item:
                            vals.append(ref_item[key])
                            break
            collated[key] = default_collate(vals)

    return collated


def torch_to_jax(t):
    if t.dtype == torch.int64:
        t = t.to(torch.int32)
    elif t.dtype == torch.float64:
        t = t.to(torch.float32)
    return jax.dlpack.from_dlpack(t.contiguous())


def prepare_datasets(datasets_root_dir):
    datasets_root_dir = Path(datasets_root_dir)
    train_datasets = []
    val_datasets = []
    test_datasets = []
    dataset_repos = sorted([str(p) for p in datasets_root_dir.iterdir() if p.is_dir()])

    for repo in dataset_repos:
        ds = load_dataset(repo)['train']
        if "timestamp" in ds.features and ds.features["timestamp"].dtype == "float32":
            from datasets import Value
            ds = ds.cast_column("timestamp", Value("float64"))
        ds = ds.add_column("dataset_name", [repo.split("/")[-1]] * len(ds))
        
        episodes = list(set(ds['episode_index']))
        num_eps = len(episodes)
        
        train_end = int(0.9 * num_eps)
        val_end = int(0.95 * num_eps)
        
        train_eps = set(episodes[:train_end])
        val_eps = set(episodes[train_end:val_end])
        test_eps = set(episodes[val_end:])
        
        train_datasets.append(ds.filter(lambda x: x["episode_index"] in train_eps))
        if val_eps:
            val_datasets.append(ds.filter(lambda x: x["episode_index"] in val_eps))
        if test_eps:
            test_datasets.append(ds.filter(lambda x: x["episode_index"] in test_eps))

    if not train_datasets:
        raise ValueError(f"No datasets found in {datasets_root_dir}")

    train_ds = concatenate_datasets(train_datasets)
    val_ds = concatenate_datasets(val_datasets) if val_datasets else None
    test_ds = concatenate_datasets(test_datasets) if test_datasets else None

    return train_ds, val_ds, test_ds

def setup_dataloader(config):
    train_ds, val_ds, test_ds = prepare_datasets(config.datasets_root)
    
    def make_loader(ds, is_train):
        if ds is None or len(ds) == 0:
            return None
        dataset = VideoDataset(
            dataset=ds,
            datasets_root=config.datasets_root,
            action_horizon=config.action_horizon,
            precompute_path=config.precompute_path,
        )
        return DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            shuffle=is_train, 
            pin_memory=False,
            num_workers=config.num_workers if is_train else 0,
            collate_fn=custom_collate_fn,
            drop_last=True
        )
        
    train_loader = make_loader(train_ds, True)
    val_loader = make_loader(val_ds, False)
    test_loader = make_loader(test_ds, False)
    
    print(f"Using {len(train_loader.dataset)} items for train, "
          f"{len(val_loader.dataset) if val_loader else 0} items for val, "
          f"{len(test_loader.dataset) if test_loader else 0} items for test.")
    
    sample_item = train_loader.dataset.dataset[0]
    obs_dim = len(sample_item["observation.state"])
    action_dim = len(sample_item["action"])
    
    return train_loader, val_loader, test_loader, obs_dim, action_dim


def loss_fn(model, vlm_out, observation, action, t, noise_key, lambda_recon, lambda_temporal):
    # 1. Action is already in raw space: (B, H, A)
    x_1 = action
    
    # 2. Generate random noise (x_0)
    x_0 = jax.random.normal(noise_key, x_1.shape)
    
    # 3. Flow Matching in raw space: x_t = (1-t) * x_0 + t * x_1
    t_exp = t.reshape(-1, 1, 1)
    x_t = (1.0 - t_exp) * x_0 + t_exp * x_1
    
    # 4. Target Velocity in raw space
    target_v = x_1 - x_0
    
    outputs = model(
        images=None, 
        input_ids=None, 
        observation=observation, 
        action=x_t,
        t=t,
        vlm_out=vlm_out,
        return_attn=True
    )
    pred_v_raw = outputs["pred_v_raw"]
    all_attns = outputs["all_attns"]
    vlm_modulated = outputs["vlm_modulated"]
    obs_emb = outputs["obs_emb"]
    action_proj = outputs["action_proj"]
    
    # 5. MSE Loss on raw velocity
    fm_loss = jnp.mean((pred_v_raw - target_v) ** 2)
    
    # 6. Action Reconstruction Huber Loss
    action_proj = model.action_projector(action)
    action_recon = model.action_unembed(action_proj, obs_emb)
    recon_loss = jnp.mean(optax.huber_loss(action_recon, action, delta=1.0))
    
    # 7. Temporal Consistency Loss
    # Reconstruct predicted action x_1
    pred_x_1 = x_t + (1.0 - t_exp) * pred_v_raw
    diff_pred = pred_x_1[:, 1:, :] - pred_x_1[:, :-1, :]
    diff_gt = action[:, 1:, :] - action[:, :-1, :]
    temporal_loss = jnp.mean((diff_pred - diff_gt) ** 2)
    
    loss_val = fm_loss + lambda_recon * recon_loss + lambda_temporal * temporal_loss
    return loss_val, (pred_v_raw, target_v, x_t, fm_loss, recon_loss, temporal_loss, vlm_modulated, obs_emb, action_proj, all_attns)


@nnx.jit
def train_step(model, optimizer, vlm_out, observation, action, t, noise_key, lambda_recon, lambda_temporal):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss_val, aux), grads = grad_fn(model, vlm_out, observation, action, t, noise_key, lambda_recon, lambda_temporal)
    optimizer.update(model, grads)
    
    grad_norm = optax.global_norm(grads)
    
    comp_grad_norms = {
        "action_projector": optax.global_norm(grads.action_projector) if hasattr(grads, "action_projector") else 0.0,
        "action_unembed": optax.global_norm(grads.action_unembed) if hasattr(grads, "action_unembed") else 0.0,
        "dit": optax.global_norm(grads.dit) if hasattr(grads, "dit") else 0.0,
        "modulator": optax.global_norm(grads.modulator) if hasattr(grads, "modulator") else 0.0,
        "obs_projector": optax.global_norm(grads.obs_projector) if hasattr(grads, "obs_projector") else 0.0,
    }
    
    return loss_val, aux, grad_norm, comp_grad_norms


@nnx.jit
def eval_step(model, vlm_out, observation, action, t, noise_key, lambda_recon, lambda_temporal):
    loss_val, aux = loss_fn(model, vlm_out, observation, action, t, noise_key, lambda_recon, lambda_temporal)
    return loss_val, aux

@nnx.jit
def val_generate_step(model, vlm_out, observation, noise_key):
    outputs = model(
        images=None, 
        input_ids=None, 
        observation=observation, 
        action=None, 
        vlm_out=vlm_out,
        key=noise_key
    )
    return outputs["pred_action"]


def log_train_metrics(
    console, pred_v_raw, target_v_raw, fm_loss, recon_loss, temporal_loss,
    vlm_modulated, obs_emb, action_proj, all_attns,
    loss_val, grad_norm, comp_grad_norms,
    epoch, global_step, step, lr
):
    # Shape: (action_dim,) — tells you WHICH joints are mispredicted
    per_dim_bias = jnp.mean(pred_v_raw - target_v_raw, axis=(0, 1))  # (A,)
    per_dim_mae  = jnp.mean(jnp.abs(pred_v_raw - target_v_raw), axis=(0, 1))
    per_dim_target_abs_mean = jnp.mean(jnp.abs(target_v_raw), axis=(0, 1)) + 1e-8

    max_relative_mae = jnp.max(per_dim_mae / per_dim_target_abs_mean)
    max_relative_bias = jnp.max(jnp.abs(per_dim_bias) / per_dim_target_abs_mean)
    
    pred_mean = jnp.mean(pred_v_raw)
    target_mean = jnp.mean(target_v_raw)
    cov = jnp.mean((pred_v_raw - pred_mean) * (target_v_raw - target_mean))
    correlation = cov / (jnp.std(pred_v_raw) * jnp.std(target_v_raw) + 1e-8)

    log_dict = {
        "train/loss": float(loss_val),
        "train/recon_loss": float(recon_loss),
        "train/temporal_loss": float(temporal_loss),
        "train/grad_norm": float(grad_norm),
        "train/max_relative_bias": float(max_relative_bias),
        "train/correlation": float(correlation),
        "train/max_relative_mae": float(max_relative_mae),
        "epoch": epoch + 1,
        "train/lr": float(lr)
    }
    
    # Add modulator output mean and std
    mod_mean = jnp.mean(vlm_modulated)
    mod_std = jnp.std(vlm_modulated)
    log_dict["ob_projector/mod_out_mean"] = float(mod_mean)
    log_dict["ob_projector/mod_out_std"] = float(mod_std)
    
    # Add observation embedding mean and std
    obs_mean = jnp.mean(obs_emb)
    obs_std = jnp.std(obs_emb)
    log_dict["ob_projector/obs_emb_mean"] = float(obs_mean)
    log_dict["ob_projector/obs_emb_std"] = float(obs_std)
    
    # Add action embedding mean and std
    act_mean = jnp.mean(action_proj)
    act_std = jnp.std(action_proj)
    log_dict["ob_projector/action_emb_mean"] = float(act_mean)
    log_dict["ob_projector/action_emb_std"] = float(act_std)
    
    if step % 50 == 0:
        console.print(f"[dim]Modulator Output | Mean: {mod_mean:.4f} | Std: {mod_std:.4f}[/dim]")
        
        # Generate and log attention heatmap for all blocks
        try:
            num_blocks = len(all_attns)
            fig, axes = plt.subplots(2, num_blocks, figsize=(6 * num_blocks, 12), squeeze=False)
                
            for i, (sa_attn, ca_attn) in enumerate(all_attns):
                # Self attention
                sa_matrix = sa_attn[-1]  # Last element in the batch
                sa_matrix_mean = jnp.mean(sa_matrix, axis=0)  # Average across heads
                sa_matrix_np = np.array(sa_matrix_mean)
                
                sns.heatmap(sa_matrix_np, ax=axes[0, i], cmap="viridis")
                axes[0, i].set_title(f"Self-Attn Block {i}")
                
                # Cross attention
                ca_matrix = ca_attn[-1]
                ca_matrix_mean = jnp.mean(ca_matrix, axis=0)
                ca_matrix_np = np.array(ca_matrix_mean)
                
                sns.heatmap(ca_matrix_np, ax=axes[1, i], cmap="viridis")
                axes[1, i].set_title(f"Cross-Attn Block {i}")
                
            fig.suptitle(f"Step {global_step} Attention Heatmaps (Batch Element -1)", fontsize=16)
            plt.tight_layout()
            log_dict["ob_projector/attention_heatmap"] = wandb.Image(fig)
            plt.close(fig)
        except Exception as e:
            console.print(f"[red]Failed to generate heatmap: {e}[/red]")
    
    for comp_name, norm_val in comp_grad_norms.items():
        log_dict[f"grad/{comp_name}"] = float(norm_val)
    
    wandb.log(log_dict, step=global_step)
    
    if step % 100 == 0:
        console.print(f"[dim]Step {global_step} | Loss: {float(loss_val):.4f} (FM: {float(fm_loss):.4f}, Recon: {float(recon_loss):.4f}) | Grad: {float(grad_norm):.3f}[/dim]")


def log_val_metrics(
    console, val_losses, val_recon_losses, val_temporal_losses,
    val_correlations, val_max_relative_biases, val_max_relative_maes,
    val_action_mses, global_step, epoch
):
    if not val_losses:
        return
        
    mean_val_loss = float(np.mean(val_losses))
    mean_val_recon_loss = float(np.mean(val_recon_losses))
    mean_val_temporal_loss = float(np.mean(val_temporal_losses))
    mean_val_correlation = float(np.mean(val_correlations))
    mean_val_max_relative_bias = float(np.mean(val_max_relative_biases))
    mean_val_max_relative_mae = float(np.mean(val_max_relative_maes))
    mean_val_action_mse = float(np.mean(val_action_mses))
    
    console.print(f"[bold magenta]Step {global_step} - Val Loss: {mean_val_loss:.4f} | Action MSE: {mean_val_action_mse:.4f} | Recon: {mean_val_recon_loss:.4f}[/bold magenta]\n")
    wandb.log({
        "val/loss": mean_val_loss,
        "val/recon_loss": mean_val_recon_loss,
        "val/temporal_loss": mean_val_temporal_loss,
        "val/action_mse": mean_val_action_mse,
        "val/correlation": mean_val_correlation,
        "val/max_relative_bias": mean_val_max_relative_bias,
        "val/max_relative_mae": mean_val_max_relative_mae,
        "epoch": epoch + 1
    }, step=global_step)


def main():
    config = parse_args()
    
    wandb.init(project=config.wandb_project, config=asdict(config))
    
    train_loader, val_loader, test_loader, obs_dim, action_dim = setup_dataloader(config)
    
    rngs = nnx.Rngs(42)
    vla = VLA(
        hidden_size=config.hidden_size, 
        obs_dim=obs_dim, 
        rngs=rngs, 
        dit_num_blocks=config.dit_num_blocks,
        vla_k=config.vla_k,
        horizon=config.action_horizon,
        action_dim=action_dim,
        vlm_checkpoint_path=config.vlm_checkpoint_path
    )
    
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(vla, nnx.Param)))
    print(f"Model initialized with {num_params:,} trainable parameters.")
    wandb.config.update({"num_trainable_params": num_params})

    num_train_steps = config.num_epochs * len(train_loader)
    warmup_steps = int(0.03 * num_train_steps)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=num_train_steps,
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.nadamw(learning_rate=schedule, weight_decay=config.weight_decay)
    )
    optimizer = nnx.Optimizer(vla, tx, wrt=nnx.Param)

    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True, enable_async_checkpointing=False)
    checkpoint_manager = ocp.CheckpointManager(
        os.path.abspath(config.checkpoint_dir),
        item_names=("model_state", "optimizer_state"),
        options=options
    )

    devices = np.array(jax.devices())
    mesh = Mesh(devices, axis_names=("data",))
    batch_sharding = NamedSharding(mesh, PartitionSpec("data"))

    global_step = 0
    
    with jax.set_mesh(mesh):
        for epoch in range(config.num_epochs):
            console.print(f"\n[bold cyan]=== Epoch {epoch+1}/{config.num_epochs} ===[/bold cyan]")
            
            for step, batch in enumerate(train_loader):
                t0 = time.time()
            
                vlm_out_jnp = torch_to_jax(batch['vlm_out'])
                observation_jnp = torch_to_jax(batch['observation_state'])
                action_jnp = torch_to_jax(batch['action'])
                
                key = jax.random.PRNGKey(epoch * len(train_loader) + step)
                key, noise_key, t_key = jax.random.split(key, 3)
                u = jax.random.beta(t_key, 1.5, 1.0, shape=(action_jnp.shape[0],))
                t = 0.999 * (1.0 - u)
                
                t1 = time.time()
                t2 = time.time()
                
                vlm_out_jnp = jax.device_put(vlm_out_jnp, batch_sharding)
                observation_jnp = jax.device_put(observation_jnp, batch_sharding)
                action_jnp = jax.device_put(action_jnp, batch_sharding)
                t = jax.device_put(t, batch_sharding)
                t3 = time.time()
                
                loss_val, aux, grad_norm, comp_grad_norms = train_step(vla, optimizer, vlm_out_jnp, observation_jnp, action_jnp, t, noise_key, config.lambda_recon, config.lambda_temporal)
                
                t4 = time.time()
                pred_v_raw, target_v_raw, x_t, fm_loss, recon_loss, temporal_loss, vlm_modulated, obs_emb, action_proj, all_attns = aux
                
                loss_val = jax.block_until_ready(loss_val)
                t5 = time.time()
                
                if step < 5 or step % 50 == 0:
                    console.print(f"[bold cyan]Step {step}[/bold cyan] [dim]Profiling | Prep: {t1-t0:.4f}s | VLM: {t2-t1:.4f}s | DevicePut: {t3-t2:.4f}s | JIT Execute: {t5-t3:.4f}s[/dim]")
                
                log_train_metrics(
                    console=console,
                    pred_v_raw=pred_v_raw, target_v_raw=target_v_raw,
                    fm_loss=fm_loss, recon_loss=recon_loss, temporal_loss=temporal_loss,
                    vlm_modulated=vlm_modulated, obs_emb=obs_emb, action_proj=action_proj, all_attns=all_attns,
                    loss_val=loss_val, grad_norm=grad_norm, comp_grad_norms=comp_grad_norms,
                    epoch=epoch, global_step=global_step, step=step, lr=schedule(global_step)
                )
                
                if global_step % config.save_every == 0 and global_step > 0:
                    console.print(f"\n[bold yellow]Step {global_step}: Saving Checkpoint...[/bold yellow]")
                    model_state = nnx.state(vla)
                    opt_state = nnx.state(optimizer)
                    checkpoint_manager.save(
                        global_step, 
                        args=ocp.args.Composite(
                            model_state=ocp.args.StandardSave(model_state),
                            optimizer_state=ocp.args.StandardSave(opt_state)
                        )
                    )
                    checkpoint_manager.wait_until_finished()
                    console.print(f"[bold green]Step {global_step}: Checkpoint Saved![/bold green]")
                    
                if global_step % 100 == 0 and global_step > 0 and val_loader is not None:
                    console.print(f"\n[bold yellow]Step {global_step}: Starting Validation...[/bold yellow]")
                    val_losses = []
                    val_fm_losses = []
                    val_recon_losses = []
                    val_temporal_losses = []
                    val_correlations = []
                    val_max_relative_biases = []
                    val_max_relative_maes = []
                    val_action_mses = []
                    
                    MAX_VAL_BATCHES = 50
                    for val_idx, val_batch in enumerate(val_loader):
                        if val_idx >= MAX_VAL_BATCHES:
                            break
                            
                        val_vlm_out_jnp = torch_to_jax(val_batch['vlm_out'])
                        val_observation_jnp = torch_to_jax(val_batch['observation_state'])
                        val_action_jnp = torch_to_jax(val_batch['action'])
                        
                        val_key = jax.random.PRNGKey(epoch * len(val_loader) + val_idx)
                        val_key, val_noise_key, val_t_key = jax.random.split(val_key, 3)
                        val_u = jax.random.beta(val_t_key, 1.5, 1.0, shape=(val_action_jnp.shape[0],))
                        val_t = 0.999 * (1.0 - val_u)
                        
                        val_vlm_out_jnp = jax.device_put(val_vlm_out_jnp, batch_sharding)
                        val_observation_jnp = jax.device_put(val_observation_jnp, batch_sharding)
                        val_action_jnp = jax.device_put(val_action_jnp, batch_sharding)
                        val_t = jax.device_put(val_t, batch_sharding)
                        
                        if global_step == 100 and val_idx == 0:
                            console.print("[bold cyan]Note: JAX is compiling the validation step for the first time. This may take 3-5 minutes...[/bold cyan]")
                            
                        val_loss_val, val_aux = eval_step(vla, val_vlm_out_jnp, val_observation_jnp, val_action_jnp, val_t, val_noise_key, config.lambda_recon, config.lambda_temporal)
                        val_pred_action = val_generate_step(vla, val_vlm_out_jnp, val_observation_jnp, val_noise_key)
                        
                        val_loss_val = jax.block_until_ready(val_loss_val)
                        val_pred_action = jax.block_until_ready(val_pred_action)
                        
                        val_losses.append(float(val_loss_val))
                        val_pred_v_raw, val_target_v_raw, _, val_fm_loss, val_recon_loss, val_temporal_loss, val_vlm_modulated, val_obs_emb, val_action_proj, val_all_attns = val_aux
                        val_recon_losses.append(float(val_recon_loss))
                        val_temporal_losses.append(float(val_temporal_loss))
                        
                        # Multi-step integrated metrics
                        val_action_mse = jnp.mean((val_pred_action - val_action_jnp) ** 2)
                        val_action_mses.append(float(val_action_mse))
                        
                        val_per_dim_bias = jnp.mean(val_pred_action - val_action_jnp, axis=(0, 1))
                        val_per_dim_mae = jnp.mean(jnp.abs(val_pred_action - val_action_jnp), axis=(0, 1))
                        
                        # Use a floor of 1e-2 to prevent division by zero/near-zero for inactive dimensions
                        val_per_dim_target_abs_mean = jnp.maximum(jnp.mean(jnp.abs(val_action_jnp), axis=(0, 1)), 1e-2)
                        
                        val_max_relative_mae = jnp.max(val_per_dim_mae / val_per_dim_target_abs_mean)
                        val_max_relative_bias = jnp.max(jnp.abs(val_per_dim_bias) / val_per_dim_target_abs_mean)
                        
                        val_pred_mean = jnp.mean(val_pred_action)
                        val_target_mean = jnp.mean(val_action_jnp)
                        val_cov = jnp.mean((val_pred_action - val_pred_mean) * (val_action_jnp - val_target_mean))
                        val_correlation = val_cov / (jnp.std(val_pred_action) * jnp.std(val_action_jnp) + 1e-8)
                        
                        val_correlations.append(float(val_correlation))
                        val_max_relative_biases.append(float(val_max_relative_bias))
                        val_max_relative_maes.append(float(val_max_relative_mae))
                    
                    log_val_metrics(
                        console=console,
                        val_losses=val_losses,
                        val_recon_losses=val_recon_losses,
                        val_temporal_losses=val_temporal_losses,
                        val_correlations=val_correlations,
                        val_max_relative_biases=val_max_relative_biases,
                        val_max_relative_maes=val_max_relative_maes,
                        val_action_mses=val_action_mses,
                        global_step=global_step,
                        epoch=epoch
                    )
                    
                global_step += 1

if __name__ == "__main__":
    main()
