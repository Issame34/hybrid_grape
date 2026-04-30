import jax
import jax.numpy as jnp

from .operators import *
from .ham_trees import *
from .unit_evol import *


def mesolve_step(uevol, c_ops, rho_init, dt):
    
    rho_fin_no_relax = uevol @ rho_init @ hconj(uevol)
    
    relax_term = 0
    for c_op in c_ops:
        num_op = hconj(c_op) @ c_op
        relax_term += c_op @ rho_init @ hconj(c_op) - \
                (num_op @ rho_init + rho_init @ num_op) / 2
        
    d_rho_relax = relax_term * dt
    
    rho_fin = rho_fin_no_relax + d_rho_relax

    return rho_fin


def mesolve_htree(htree, c_ops, rho_t0, dts, evol_hdt=evol_hdt_exp):
    """
    ME solution for H-tree "htree" (see below)
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
        mesolve_step(x[0], c_ops, rho, x[1]), None
    )
    rho_fin, _ = jax.lax.scan(f, rho_t0, [uevols, dts])
    
    return rho_fin