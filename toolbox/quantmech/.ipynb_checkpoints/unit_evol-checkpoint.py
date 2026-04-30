import jax.numpy as jnp
import jax.scipy as jsp


def evol_hdt_exp(H, dt):
    """
    "Exponential" (i. e. exact) evolution
        under Hamiltonian "H" (rank 2)
        in a small time interval "dt".
    """
    dL = -1j * H * dt
    U_dt = jsp.linalg.expm(dL)
    return U_dt


def evol_hdt_utrick(H, dt):
    """
    "Unitary trick" approximation
        of evolution under Hamiltonian "H" (rank 2)
        in a small time interval "dt".
    """
    dL = -1j * H * dt
    ot = jnp.eye(dL.shape[-1], dtype=dL.dtype)
    #U_dt = jnp.linalg.inv(ot - dL/2) @ (ot + dL/2)
    U_dt = jnp.linalg.solve(ot - dL/2, ot + dL/2)
    return U_dt