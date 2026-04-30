import jax
import jax.numpy as jnp
import numpy as np

from .operators import *


def hmat_from_htree(htree):
    """
    Constructs H-matrix from H-tree.
    H-tree is a list of 4-lists of the form:
        [
            operator,
            coeff (num or time array),
            common coeff for operator,
            common coeff for hconj of operator
        ]
    H-matrix has always rank 3 with a time axis coming first.
    """
    htree1 = jax.tree.map(
        lambda a: [jnp.tensordot(jnp.atleast_1d(a[1]), a[0], axes=0), a[2], a[3]],
        htree,
        is_leaf=lambda x: jax.tree_util.all_leaves(x)
    )
    htree2 = jax.tree.map(
        lambda a: a[1] * a[0] + a[2] * hconj(a[0]),
        htree1,
        is_leaf=lambda x: jax.tree_util.all_leaves(x)
    )
    return jax.tree_util.tree_reduce(jnp.add, htree2)


def get_htree_at_t(htree, i):
    htree_at_t = []
    
    for htree_term in htree:
        
        coeffs_arr = jnp.array(htree_term[1])
        if coeffs_arr.ndim == 0:
            coeff = coeffs_arr
        elif coeffs_arr.ndim == 1:
            coeff = coeffs_arr[i]

        htree_term_at_t = [
            htree_term[0],
            coeff,
            htree_term[2],
            htree_term[3]
        ]
        htree_at_t.append(htree_term_at_t)   
        
    return htree_at_t


def qutipify_htree(htree, **kwargs):
    """
    Converts an H-tree "htree" to QuTiP format.
    H-tree is a list of 4-lists of the form:
        [
            operator,
            coeff (num or time-array),
            common coeff for operator,
            common coeff for hconj of operator
        ]
    "kwargs" are arguments passed directly to QuTiP Qobj constructor.
    """
    import qutip as qt
    op_to_qt = lambda op: qt.Qobj(np.array(op), **kwargs)
    
    qutipified = []
    for op, ct, c1, c2 in htree:
        if jnp.array(ct).ndim == 0:
            opct = op * ct
            qutipified.append(
                op_to_qt(opct * c1 + hconj(opct) * c2)
            )
        elif jnp.array(ct).ndim == 1:
            if c1:
                qutipified.append(
                    [op_to_qt(op), np.array(ct * c1)]
                )
            if c2:
                qutipified.append(
                    [op_to_qt(hconj(op)), np.array(ct.conj() * c2)]
                )
        else:
            raise ValueError("Could not qutipify this tree because some time-dependent coefficient has a wrong dimension.")
    
    return qutipified