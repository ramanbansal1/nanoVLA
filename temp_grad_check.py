import jax
import jax.numpy as jnp
from flax import nnx

from models.vla import VLA

def check_gradients():
    rngs = nnx.Rngs(42)
    vla = VLA(hidden_size=192, obs_dim=30, rngs=rngs, dummy=True)
    
    dummy_obs = jnp.zeros((1, 30))
    dummy_act = jnp.zeros((1, 30))
    dummy_img = jnp.zeros((1, 3, 224, 224))
    dummy_inst = "Test instruction"
    
    key = jax.random.PRNGKey(42)
    key, noise_key, t_key = jax.random.split(key, 3)
    
    noise = jax.random.normal(noise_key, dummy_act.shape)
    t = jax.random.uniform(t_key, shape=(dummy_act.shape[0],))
    t_exp = t.reshape(-1, 1, 1) if dummy_act.ndim == 3 else t.reshape(-1, 1)
    x_t = (1 - t_exp) * noise + t_exp * dummy_act
    
    def loss_fn(model):
        _, action_emb, _, _, lat, _ = model(
            images=dummy_img,
            instruction=dummy_inst,
            observation=dummy_obs,
            action=x_t,
            t=t
        )
        noisy_emb = action_emb
        clean_emb = model.action_tokenizer(dummy_act)
        
        velocity_target = clean_emb - noisy_emb
        predicted_velocity = lat - noisy_emb
        
        return jnp.mean((predicted_velocity - velocity_target) ** 2)
    
    print("Computing gradients...")
    grad_fn = nnx.grad(loss_fn)
    grads = grad_fn(vla)
    
    print("\n=== Gradient Check ===")
    
    # Flatten the state dict to easily print keys and gradient norms
    flat_grads = nnx.state(grads).flat_state()
    
    active_modules = set()
    for path, grad_val in flat_grads.items():
        # path is a tuple of keys, e.g., ('dit', 'blocks', 0, 'attn', 'wq', 'value')
        module_name = path[0] 
        grad_norm = jnp.linalg.norm(grad_val)
        
        if grad_norm > 0:
            active_modules.add(module_name)
            
    print(f"Modules that successfully received gradients: {list(active_modules)}")
    
    print("\nDetailed layer gradient norms:")
    for path, grad_val in flat_grads.items():
        # Just show the top-level keys and their aggregated norms to keep it clean
        if len(path) > 1:
            layer = ".".join([str(p) for p in path[:2]])
            if layer not in active_modules:
                pass
        
        grad_norm = jnp.linalg.norm(grad_val)
        if grad_norm > 0:
            # Print specifically to see DiT, Tokenizer, etc.
            print(f"Layer: {'.'.join([str(p) for p in path])} | Norm: {grad_norm:.6f}")
            
if __name__ == "__main__":
    check_gradients()
