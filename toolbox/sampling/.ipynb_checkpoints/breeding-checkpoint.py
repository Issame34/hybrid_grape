import jax
import jax.numpy as jnp


def sample_with_breeding(key, size, rngs, ri, pt_losses, sgms):

    no_conc = (pt_losses is None) or (ri == 0)
    
    n_expl = size if no_conc else (size - jnp.floor(size * ri).astype(int))
    
    key, subkey = jax.random.split(key)
    expl_pts = jax.random.uniform(subkey, shape=(n_expl, len(rngs)))
    
    for i in range(len(rngs)):
        expl_pts = (expl_pts
                    .at[:, i].mul(rngs[i][1] - rngs[i][0])
                    .at[:, i].add(rngs[i][0])
                    )
    
    if no_conc:
        pts = expl_pts
    
    else:
        n_conc = size - n_expl

        key, subkey = jax.random.split(key)
        conc_pts = jax.random.normal(subkey, shape=(n_conc, len(rngs))) * jnp.array(sgms)

        min_vals = jnp.array(rngs)[:, 0]
        max_vals = jnp.array(rngs)[:, 1]
        
        key, subkey = jax.random.split(key)
        chld_center_inds = jax.random.choice(subkey, len(pt_losses), shape=(n_conc,), p=pt_losses[:, -1])
        
        conc_centers = pt_losses[chld_center_inds, :-1]
        min_vals = jnp.array(rngs)[:, 0]
        max_vals = jnp.array(rngs)[:, 1]
        conc_centers = jnp.clip(conc_centers, min_vals, max_vals)
        
        conc_pts += conc_centers

        pts = jnp.concatenate([expl_pts, conc_pts])
        key, subkey = jax.random.split(key)
        pts = jax.random.permutation(subkey, pts)

    return pts