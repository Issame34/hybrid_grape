import jax
import jax.numpy as jnp

from ..sampling import *
from .wigner_eval import *


def sample_with_wigner(key, sample_size,
                       complex_uniform_grid, d_complex_uniform_grid,
                       wigs):
    grid_pts = jnp.vstack([complex_uniform_grid.real, complex_uniform_grid.imag]).T
    grid_ds = jnp.array(d_complex_uniform_grid)
    densities = jnp.abs(wigs)
    smpl = sample_with_grid_density(key, sample_size, grid_pts, grid_ds, densities)
    return smpl[:, 0] + 1j * smpl[:, 1]


def parity_measurement(key, wigs):
    r = jax.random.uniform(key,
                           shape=(len(wigs),),
                           minval=-2/jnp.pi,
                           maxval=2/jnp.pi)
    pars = 2 * (r < wigs) - 1
    return pars


def infidelity_from_wigs(wigs_curr, wigs_tgt, norm_abs):
    f = wigs_tgt - wigs_curr
    infid = jnp.pi * (jnp.sign(wigs_tgt) * f).sum() \
            / len(wigs_tgt) * norm_abs
    return infid


def infidelity_from_pars(key, wigs_curr, wigs_tgt, norm_abs):
    pars = parity_measurement(key, wigs_curr)
    wigs_from_pars = (2 / jnp.pi) * pars
    infid = infidelity_from_wigs(wigs_from_pars, wigs_tgt, norm_abs)
    return infid


def infidelity_measurement(key,
                           rho_curr, rho_tgt,
                           sample_size, betas, d_betas,
                           N_cav, N_cav_shift):
    wigs = wigner_on_grid(rho_tgt, betas, N_cav, N_cav_shift)
    d2_beta = d_betas[0] * d_betas[1]
    norm_abs = jnp.abs(wigs).sum() * d2_beta
 
    key1, key2 = jax.random.split(key)
    betas_smp = sample_with_wigner(key1, sample_size, betas, d_betas, wigs)
    
    wigs_curr = wigner_on_grid(rho_curr, betas_smp, N_cav, N_cav_shift)
    wigs_tgt = wigner_on_grid(rho_tgt, betas_smp, N_cav, N_cav_shift)
    
    infid = infidelity_from_pars(key2, wigs_curr, wigs_tgt, norm_abs)
    return infid