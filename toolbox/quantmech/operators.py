from collections import namedtuple
import jax.numpy as jnp


hconj = lambda a: jnp.swapaxes(a.conj(), -1, -2)
identity = lambda N: jnp.eye(N, dtype=complex)
expect = lambda op, psi: jnp.squeeze(hconj(psi) @ op @ psi)


def ladder(N, *, dagger: bool):
    """
    N-dimensional ladder operator
    """
    values = jnp.sqrt(jnp.arange(1, N, dtype=complex))
    shift = -1 if dagger else 1
    return jnp.diag(values, k=shift)


create = lambda N: ladder(N, dagger=True)
destroy = lambda N: ladder(N, dagger=False)


def tensor(a, b):
    c = jnp.tensordot(a, b, axes=0)
    res_shape = (a.shape[0] * b.shape[0], a.shape[1] * b.shape[1])
    res = jnp.transpose(c, axes=(0, 2, 1, 3)).reshape(res_shape)
    return res


sigma = {
    "x": jnp.array([[0, 1], [1, 0]], dtype=complex),
    "y": jnp.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": jnp.array([[1, 0], [0, -1]], dtype=complex),
    "p": jnp.array([[0, 1], [0, 0]], dtype=complex),
    "m": jnp.array([[0, 0], [1, 0]], dtype=complex)
}
sigma = namedtuple('sigma', sigma.keys())(**sigma)