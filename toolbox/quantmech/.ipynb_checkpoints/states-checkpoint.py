import jax
import jax.numpy as jnp
import jax.scipy as jsp

from .operators import *


def basis(N, k=0):
    """
    Basis state in N-dimensional Hilbert space
    """
    one_hot = jax.nn.one_hot(k, N, dtype=complex)
    return one_hot.reshape(N, 1)


def displ(N, alpha):
    """
    Displacement operator
    """
    displ_arg = alpha * create(N) - jnp.conj(alpha) * destroy(N)
    displ_exp = jsp.linalg.expm(displ_arg)
    return displ_exp

coherent = lambda N, alpha: displ(N, alpha) @ basis(N)