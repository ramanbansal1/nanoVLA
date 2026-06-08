#!.venv/bin/python
import os
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import torch
import torch.utils.dlpack
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import ToTensor
from datasets import load_dataset, concatenate_datasets, Value
from pathlib import Path
import wandb
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax
import orbax.checkpoint as ocp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from tqdm.auto import tqdm

from config import parse_args, asdict
from data.advanced_dataset import VideoDataset
from models.vla import VLA


def custom_collate_fn(batch):
    to_tensor = ToTensor()
    collated = {}

    for key in batch[0].keys():
        if key == "images":
            imgs = []
            for item in batch:
                first_cam = sorted(item["images"].keys())[0]
                imgs.append(to_tensor(item["images"][first_cam]))
            collated["image"] = default_collate(imgs)
        elif key == "instruction":
            collated[key] = [item[key] for item in batch]
        elif key != "images":
            collated[key] = default_collate([item[key] for item in batch])

    return collated


def torch_to_jax(t):
    try:
        t_cuda = t.cuda(non_blocking=True).contiguous()
        return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(t_cuda))
    except Exception:
        return jnp.array(t.contiguous().numpy())


def prepare_datasets(datasets_root_dir):
    datasets_root_dir = Path(datasets_root_dir)
    train_datasets = []
    val_datasets = []
    test_datasets = []
    dataset_repos = sorted([str(p) for p in datasets_root_dir.iterdir() if p.is_dir()])

    for repo in dataset_repos:
        ds = load_dataset(repo)['train']
        if ds.features["timestamp"].dtype == "float32":
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
    # Check VLM Context Path and fallback to dummy if missing
    if config.vlm_context_dir and not os.path.exists(config.vlm_context_dir):
        print(f"WARNING: VLM context directory '{config.vlm_context_dir}' not found. Falling back to dummy VLM.")
        config.vlm_context_dir = None
        config.dummy_vlm = True
        
    train_ds, val_ds, test_ds = prepare_datasets(config.datasets_root)
    
    def make_loader(ds, is_train):
        if ds is None or len(ds) == 0:
            return None
        dataset = VideoDataset(
            dataset=ds,
            datasets_root=config.datasets_root,
            action_horizon=config.action_horizon,
            vlm_context_root=config.vlm_context_dir,
        )
        return DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            shuffle=is_train, 
            pin_memory=True,
            num_workers=config.num_workers,
            collate_fn=custom_collate_fn,
            drop_last=is_train
        )
        
    train_loader = make_loader(train_ds, True)
    val_loader = make_loader(val_ds, False)
    test_loader = make_loader(test_ds, False)
    
    print(f"Using {len(train_loader.dataset)} items for train, "
          f"{len(val_loader.dataset) if val_loader else 0} items for val, "
          f"{len(test_loader.dataset) if test_loader else 0} items for test.")
    
    sample_item = train_loader.dataset.dataset[0]
    obs_dim = len(sample_item["observation.state"]) + len(sample_item["eef_sim_pose_state"])
    
    return train_loader, val_loader, test_loader, obs_dim


def loss_fn(model, vlm_out, observation, action, t, noise_key):
    # 1. Action is already in raw space: (B, H, A)
    x_1 = action
    
    # 2. Generate random noise (x_0)
    x_0 = jax.random.normal(noise_key, x_1.shape)
    
    # 3. Flow Matching in raw space: x_t = (1-t) * x_0 + t * x_1
    t_exp = t.reshape(-1, 1, 1)
    x_t = (1.0 - t_exp) * x_0 + t_exp * x_1
    
    # 4. Target Velocity in raw space
    target_v = x_1 - x_0
    
    vlm_modulated, action_proj, _, obs_emb, dit_out, pred_v_raw, decoded_actions = model(
        images=None, 
        instruction=None, 
        observation=observation, 
        action=x_t,
        t=t,
        vlm_out=vlm_out
    )
    
    # 5. MSE Loss on raw velocity
    loss_val = jnp.mean((pred_v_raw - target_v) ** 2)
    return loss_val, (pred_v_raw, target_v, x_t, decoded_actions)


@nnx.jit
def train_step(model, optimizer, vlm_out, observation, action, t, noise_key):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss_val, aux), grads = grad_fn(model, vlm_out, observation, action, t, noise_key)
    optimizer.update(model, grads)
    return loss_val, aux, grads


@nnx.jit
def eval_step(model, vlm_out, observation, action, t, noise_key):
    loss_val, aux = loss_fn(model, vlm_out, observation, action, t, noise_key)
    return loss_val, aux


def main():
    config = parse_args()
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = config.jax_mem_fraction
    
    wandb.init(project=config.wandb_project, config=asdict(config))
    
    train_loader, val_loader, test_loader, obs_dim = setup_dataloader(config)
    
    rngs = nnx.Rngs(42)
    vla = VLA(
        hidden_size=config.hidden_size, 
        obs_dim=obs_dim, 
        rngs=rngs, 
        dummy=config.dummy_vlm,
        dit_num_blocks=config.dit_num_blocks,
        vla_k=config.vla_k,
        patch_size=config.patch_size,
        horizon=config.action_horizon
    )
    
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(vla, nnx.Param)))
    print(f"Model initialized with {num_params:,} trainable parameters.")
    wandb.config.update({"num_trainable_params": num_params})

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.nadamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    )
    optimizer = nnx.Optimizer(vla, tx, wrt=nnx.Param)

    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    checkpoint_manager = ocp.CheckpointManager(
        os.path.abspath(config.checkpoint_dir),
        item_names=("model_state", "optimizer_state"),
        options=options
    )

    devices = np.array(jax.devices())
    mesh = Mesh(devices, axis_names=("data",))
    batch_sharding = NamedSharding(mesh, PartitionSpec("data"))

    global_step = 0
    for epoch in range(config.num_epochs):
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for step, batch in enumerate(train_pbar):
        
            if 'vlm_context' in batch:
                vlm_out = torch_to_jax(batch['vlm_context'])
                images_jnp = None
            else:
                images_jnp = torch_to_jax(batch['image'])
                vlm_out = None
                
            instruction = batch['instruction'][0]  # Take first instruction since batch_size=1
            
            observation_jnp = torch_to_jax(batch['observation_state'])
            eef_state_jnp = torch_to_jax(batch['eef_state'])
            observation_jnp = jnp.concatenate([observation_jnp, eef_state_jnp], axis=-1)
            
            action_jnp = torch_to_jax(batch['action'])
            eef_action_jnp = torch_to_jax(batch['eef_action'])
            action_jnp = jnp.concatenate([action_jnp, eef_action_jnp], axis=-1)
            
            key = jax.random.PRNGKey(global_step)
            key, noise_key, t_key = jax.random.split(key, 3)
            t = jax.random.uniform(t_key, shape=(action_jnp.shape[0],))
            
            if vlm_out is None:
                vlm_out = vla.vlm(images_jnp, instruction)
            
            with jax.set_mesh(mesh):
                vlm_out = jax.device_put(vlm_out, batch_sharding)
                observation_jnp = jax.device_put(observation_jnp, batch_sharding)
                action_jnp = jax.device_put(action_jnp, batch_sharding)
                t = jax.device_put(t, batch_sharding)
                
                loss_val, aux, grads = train_step(vla, optimizer, vlm_out, observation_jnp, action_jnp, t, noise_key)
            
            pred_v_raw, target_v_raw, x_t, decoded_actions = aux
            
            loss_val = jax.block_until_ready(loss_val)
            
            grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)]))
            mean_pred_v = jnp.mean(pred_v_raw)
            mean_target_v = jnp.mean(target_v_raw)

            log_dict = {
                "train/loss": float(loss_val),
                "train/grad_norm": float(grad_norm),
                "train/mean_pred_v": float(mean_pred_v),
                "train/mean_target_v": float(mean_target_v),
                "epoch": epoch + 1
            }
            
            wandb.log(log_dict, step=global_step)
            train_pbar.set_postfix({
                "loss": f"{float(loss_val):.4f}",
                "grad": f"{float(grad_norm):.3f}",
            })
            
            if global_step % config.save_every == 0 and global_step > 0:
                _, model_state = nnx.split(vla)
                _, opt_state = nnx.split(optimizer)
                checkpoint_manager.save(
                    global_step, 
                    args=ocp.args.Composite(
                        model_state=ocp.args.StandardSave(model_state),
                        optimizer_state=ocp.args.StandardSave(opt_state)
                    )
                )
                checkpoint_manager.wait_until_finished()
                
            if global_step % 100 == 0 and global_step > 0 and val_loader is not None:
                val_losses = []
                for val_batch in val_loader:
                    if 'vlm_context' in val_batch:
                        val_vlm_out = torch_to_jax(val_batch['vlm_context'])
                        val_images_jnp = None
                    else:
                        val_images_jnp = torch_to_jax(val_batch['image'])
                        val_vlm_out = None
                        
                    val_instruction = val_batch['instruction'][0]
                    
                    val_observation_jnp = torch_to_jax(val_batch['observation_state'])
                    val_eef_state_jnp = torch_to_jax(val_batch['eef_state'])
                    val_observation_jnp = jnp.concatenate([val_observation_jnp, val_eef_state_jnp], axis=-1)
                    
                    val_action_jnp = torch_to_jax(val_batch['action'])
                    val_eef_action_jnp = torch_to_jax(val_batch['eef_action'])
                    val_action_jnp = jnp.concatenate([val_action_jnp, val_eef_action_jnp], axis=-1)
                    
                    val_key = jax.random.PRNGKey(global_step)
                    val_key, val_noise_key, val_t_key = jax.random.split(val_key, 3)
                    val_t = jax.random.uniform(val_t_key, shape=(val_action_jnp.shape[0],))
                    
                    if val_vlm_out is None:
                        val_vlm_out = vla.vlm(val_images_jnp, val_instruction)
                    
                    with jax.set_mesh(mesh):
                        val_vlm_out = jax.device_put(val_vlm_out, batch_sharding)
                        val_observation_jnp = jax.device_put(val_observation_jnp, batch_sharding)
                        val_action_jnp = jax.device_put(val_action_jnp, batch_sharding)
                        val_t = jax.device_put(val_t, batch_sharding)
                        
                        val_loss_val, _ = eval_step(vla, val_vlm_out, val_observation_jnp, val_action_jnp, val_t, val_noise_key)
                    val_losses.append(float(val_loss_val))
                
                if val_losses:
                    mean_val_loss = float(np.mean(val_losses))
                    print(f"\nStep {global_step} - Validation Loss (MSE): {mean_val_loss:.4f}")
                    wandb.log({"val/loss": mean_val_loss}, step=global_step)
                
            global_step += 1

if __name__ == "__main__":
    main()
