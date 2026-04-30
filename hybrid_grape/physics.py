from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from toolbox.quantmech.operators import destroy, hconj, identity, sigma, tensor
from toolbox.quantmech.states import basis
from toolbox.quantmech.unit_evol import evol_hdt_exp

from .config import chi_rad_per_us, self_kerr_rad_per_us


@dataclass(frozen=True)
class SimulationConfig:
    """Simulation quality and fixed experiment parameters."""

    n_cav: int = 25
    target_n: int = 2
    initial_cavity_n: int = 0
    initial_qubit_state: int = 0
    t_drive: float = 1.408
    ndt_drive: int = 80
    num_coeffs: int = 20
    spline_degree: int = 2
    spline_skip_left: int = 2
    spline_skip_right: int = 2
    param_clip: float = 2.0


@dataclass(frozen=True)
class PhysicsParams:
    """Parameters of the Hamiltonian model used by GRAPE."""

    chi: float = chi_rad_per_us()
    cavity_self_kerr: float = self_kerr_rad_per_us()
    cavity_detuning: float = 0.0
    qubit_detuning: float = 0.0
    mu_qub: float = 20.0
    mu_cav: float = 20.0
    grape_dispersive_frame: bool = True
    grape_cavity_iq: bool = True
    cavity_phase: float = 0.0


def bspline_knots_on_interval(
    t_left: float,
    t_right: float,
    num_coeffs: int,
    degree: int,
) -> jax.Array:
    knots_left = jnp.full((degree,), t_left)
    knots_mid = jnp.linspace(t_left, t_right, num_coeffs - degree + 1)
    knots_right = jnp.full((degree,), t_right)
    return jnp.concatenate([knots_left, knots_mid, knots_right])


def bspline_basis_on_interval(
    time_grid: jax.Array,
    t_left: float,
    t_right: float,
    num_coeffs: int,
    degree: int,
    skip_left: int = 0,
    skip_right: int = 0,
) -> jax.Array:
    total_splines = num_coeffs + skip_left + skip_right
    knots = bspline_knots_on_interval(t_left, t_right, total_splines, degree)
    time_grid = jnp.asarray(time_grid)
    basis_values = (
        (time_grid[None, :] >= knots[:-1, None])
        & (time_grid[None, :] < knots[1:, None])
    ).astype(time_grid.dtype)

    for spline_degree in range(1, degree + 1):
        new_count = basis_values.shape[0] - 1
        i = jnp.arange(new_count)

        left_den = knots[i + spline_degree] - knots[i]
        right_den = knots[i + spline_degree + 1] - knots[i + 1]

        left_num = time_grid[None, :] - knots[i, None]
        right_num = knots[i + spline_degree + 1, None] - time_grid[None, :]

        left = jnp.where(
            left_den[:, None] > 0.0,
            left_num / left_den[:, None] * basis_values[:-1],
            0.0,
        )
        right = jnp.where(
            right_den[:, None] > 0.0,
            right_num / right_den[:, None] * basis_values[1:],
            0.0,
        )
        basis_values = left + right

    return basis_values[skip_left : total_splines - skip_right]


def controls_from_coefficients(
    coefficients: jax.Array,
    bsplines: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    coeffs = jnp.asarray(coefficients).reshape((4, bsplines.shape[0]))
    controls_real = coeffs @ bsplines
    e_qub = controls_real[0] + 1j * controls_real[1]
    e_cav = controls_real[2] + 1j * controls_real[3]
    return e_qub, e_cav


def raw_from_bounded_controls(
    controls: jax.Array,
    *,
    param_clip: float = 2.0,
    eps: float = 1e-5,
) -> jax.Array:
    normalized = jnp.clip(jnp.asarray(controls) / param_clip, -1.0 + eps, 1.0 - eps)
    return jnp.arctanh(normalized)


def bounded_controls_from_raw(
    raw_controls: jax.Array,
    *,
    param_clip: float = 2.0,
) -> jax.Array:
    return param_clip * jnp.tanh(raw_controls)


class FockPhysicsModel:
    """JAX differentiable cavity-qubit simulator for one Fock reward."""

    def __init__(
        self,
        sim_config: SimulationConfig = SimulationConfig(),
        physics_params: PhysicsParams = PhysicsParams(),
    ):
        self.sim_config = sim_config
        self.physics_params = physics_params
        self._validate()

        q = sim_config
        self.t_edges = jnp.linspace(0.0, q.t_drive, q.ndt_drive + 1)
        self.t_mids = (self.t_edges[1:] + self.t_edges[:-1]) / 2
        self.dt = self.t_edges[1:] - self.t_edges[:-1]
        self.bsplines_mids = bspline_basis_on_interval(
            self.t_mids,
            0.0,
            q.t_drive,
            q.num_coeffs,
            q.spline_degree,
            q.spline_skip_left,
            q.spline_skip_right,
        )
        self.bsplines_edges = bspline_basis_on_interval(
            self.t_edges,
            0.0,
            q.t_drive,
            q.num_coeffs,
            q.spline_degree,
            q.spline_skip_left,
            q.spline_skip_right,
        )

        self.a = tensor(identity(2), destroy(q.n_cav))
        self.adag = hconj(self.a)
        self.n_phot = self.adag @ self.a
        self.n_phot_sq_minus_n = self.n_phot @ self.n_phot - self.n_phot
        self.sigz = tensor(sigma.z, identity(q.n_cav))
        self.sigp = tensor(sigma.p, identity(q.n_cav))
        self.sigm = hconj(self.sigp)
        self.one = identity(2 * q.n_cav)
        self.qubit_excited = 0.5 * (self.one - self.sigz)
        self.psi0 = tensor(
            basis(2, q.initial_qubit_state),
            basis(q.n_cav, q.initial_cavity_n),
        )

    @property
    def parameter_shape(self) -> tuple[int, int]:
        return 4, self.sim_config.num_coeffs

    @property
    def parameter_size(self) -> int:
        return 4 * self.sim_config.num_coeffs

    def h_drift(self) -> jax.Array:
        p = self.physics_params
        if p.grape_dispersive_frame:
            h_disp = (-0.5 * p.chi) * (self.n_phot @ (self.one - self.sigz))
        else:
            h_disp = (0.5 * p.chi) * (self.n_phot @ (self.sigz + self.one))
        return (
            h_disp
            + 0.5 * p.cavity_self_kerr * self.n_phot_sq_minus_n
            + p.cavity_detuning * self.n_phot
            + p.qubit_detuning * self.qubit_excited
        )

    def control_fields(self, controls: jax.Array) -> tuple[jax.Array, jax.Array]:
        return controls_from_coefficients(controls, self.bsplines_mids)

    def final_state(self, controls: jax.Array) -> jax.Array:
        p = self.physics_params
        e_qub, e_cav = self.control_fields(controls)
        cavity_phase = jnp.exp(1j * p.cavity_phase)
        if p.grape_cavity_iq:
            e_cav = cavity_phase * 1j * jnp.conj(e_cav)
        else:
            e_cav = cavity_phase * e_cav
        h_drift = self.h_drift()

        def step(psi, x):
            eq, ec, dt = x
            hmat = (
                h_drift
                + p.mu_qub * (eq * self.sigp + jnp.conj(eq) * self.sigm)
                + p.mu_cav * (ec * self.adag + jnp.conj(ec) * self.a)
            )
            return evol_hdt_exp(hmat, dt) @ psi, None

        psi_final, _ = jax.lax.scan(step, self.psi0, (e_qub, e_cav, self.dt))
        return psi_final

    def photon_probability(self, controls: jax.Array, n: int | None = None) -> jax.Array:
        target_n = self.sim_config.target_n if n is None else n
        psi_final = self.final_state(controls)
        psi_by_qubit = psi_final.reshape(2, self.sim_config.n_cav)
        return jnp.sum(jnp.abs(psi_by_qubit[:, target_n]) ** 2).real

    def population_probability(self, controls: jax.Array) -> jax.Array:
        return jax.vmap(self.photon_probability)(controls)

    def _validate(self) -> None:
        q = self.sim_config
        if not 0 <= q.target_n < q.n_cav:
            raise ValueError("target_n must be inside the cavity truncation.")
        if q.initial_qubit_state not in (0, 1):
            raise ValueError("initial_qubit_state must be 0 or 1.")
        if q.spline_degree != 2:
            raise ValueError("This project expects quadratic B-splines.")
        if q.spline_skip_left < 0 or q.spline_skip_right < 0:
            raise ValueError("Spline skip counts must be non-negative.")
