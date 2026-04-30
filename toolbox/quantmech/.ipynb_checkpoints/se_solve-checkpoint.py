import jax
import jax.numpy as jnp

from .ham_trees import *
from .unit_evol import *


def evol_hmat(hmat, dts, evol_hdt):
    """
    Evolution under H-matrix "hmat" (rank 3, time axis first)
        in finite intervals from ts[0] to ts[i] for all i > 0.
    Time intervals in the input are defined as dts[i] = ts[i+1] - ts[i].
    """
    U_dts = jax.vmap(evol_hdt)(hmat, dts)
    _, U_ts = jax.lax.scan(
        lambda U1, U2: (U2 @ U1, U2 @ U1),
        jnp.eye(U_dts.shape[-1], dtype=U_dts.dtype),
        U_dts
    )
    return U_ts


def sesolve_hmat(hmat, psi_t0, dts, evol_hdt):
    """
    SE solution for H-matrix "hmat"  (rank 3, time axis first)
        in finite intervals from ts[0] to ts[i] for all i > 0
        with initial state psi_t0 at ts[0].
    Time intervals in the input are defined as dts[i] = ts[i+1] - ts[i].    
    """
    U_ts = evol_hmat(hmat, dts, evol_hdt)
    psi_ts = U_ts @ psi_t0
    return psi_ts


def sesolve_htree(htree, psi_t0, dts, evol_hdt=evol_hdt_exp):
    """
    SE solution for H-tree "htree" (see below)
        in finite intervals from ts[0] to ts[i] for all i > 0
        with initial state psi_t0 at ts[0].
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
    
    return sesolve_hmat(hmat, psi_t0, dts, evol_hdt)