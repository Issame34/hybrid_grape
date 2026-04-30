import jax
import jax.numpy as jnp

from .operators import *
from .ham_trees import *
from .unit_evol import *


def mesolve_norelax_step(uevol, rho_init, dt):
    rho_fin_norelax = uevol @ rho_init @ hconj(uevol)
    return rho_fin_norelax


def mesolve_norelax_htree(htree, rho_t0, dts, evol_hdt=evol_hdt_exp):
    """
    ME solution without relaxation for H-tree "htree" (see below)
        in finite intervals from ts[0] to ts[i] for all i > 0
        with initial state rho_t0 at ts[0].
    Time intervals in the input are defined as dts[i] = ts[i+1] - ts[i].
    H-tree is a list of 4-lists of the form:
        [
            operator,
            coeff (num or time-array),
            common coeff for operator,
            common coeff for hconj of operator
        ]
    """
    hmat = hmat_from_htree(htree)
    assert len(hmat) == len(dts)
    
    uevols = jax.vmap(evol_hdt)(hmat, dts)
    
    f = lambda rho, x: (
        mesolve_norelax_step(x[0], rho, x[1]), None
    )
    rho_fin, _ = jax.lax.scan(f, rho_t0, [uevols, dts])
    
    return rho_fin