import jax
import jax.numpy as jnp

from ..quantmech import *


def wigner(rho, beta, N_cav, N_cav_shift):
    """
    Compute the Wigner function for the state "rho"
        at the point "beta".
    N_cav --- the number of Fock states to be used
        for the state as it is.
    N_cav_shift --- the number of Fock states to be used
        when the state is shifted to "beta".
        
    Note:
    For the convension used here, the coherent state |alpha>
        will have a maximum Wigner at beta=alpha.
    """
    par = 1 - jnp.arange(N_cav_shift) % 2 * 2
    displp = displ(N_cav_shift, beta)[:N_cav, :]
    op = par * displp @ hconj(displp)
    wig = 2 / jnp.pi * jnp.trace(op @ rho).real
    return wig


wigner_on_grid = jax.vmap(wigner, in_axes=(None, 0, None, None))


def complex_uniform_grid(xampl, yampl, xbins):
    ybins = int(xbins / xampl * yampl)
    
    xc = jnp.linspace(-xampl, xampl, xbins+1)
    yc = jnp.linspace(-yampl, yampl, ybins+1)
    
    xc = (xc[1:] + xc[:-1]) / 2
    yc = (yc[1:] + yc[:-1]) / 2
    
    dx = 2 * xampl / xbins
    dy = 2 * yampl / ybins
    
    xc, yc = jnp.meshgrid(xc, yc)
    bc = (xc + 1j * yc).reshape(-1)
    
    return bc, dx, dy