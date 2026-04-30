from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from .physics import (
    FockPhysicsModel,
    bounded_controls_from_raw,
    raw_from_bounded_controls,
)


@dataclass(frozen=True)
class GrapeConfig:
    maxiter: int = 120
    memory_size: int = 10
    noise_samples: int = 8
    control_noise_std: float = 0.02
    amplitude_l2: float = 1e-4
    smoothness_l2: float = 1e-4
    param_clip: float = 2.0
    grad_clip_norm: float = 20.0


def pulse_regularization(
    controls: jax.Array,
    *,
    amplitude_l2: float,
    smoothness_l2: float,
) -> jax.Array:
    coeffs = controls.reshape((4, -1))
    return amplitude_l2 * jnp.mean(coeffs**2) + smoothness_l2 * jnp.mean(
        jnp.diff(coeffs, axis=1) ** 2
    )


def noisy_physics_objective(
    raw_controls: jax.Array,
    physics_model: FockPhysicsModel,
    fixed_noise: jax.Array,
    config: GrapeConfig,
) -> tuple[jax.Array, jax.Array]:
    center_controls = bounded_controls_from_raw(
        raw_controls,
        param_clip=config.param_clip,
    )
    noisy_controls = jnp.clip(
        center_controls[None, :] + fixed_noise,
        -config.param_clip,
        config.param_clip,
    )

    def one(controls):
        physics_p = physics_model.photon_probability(controls)
        penalty = pulse_regularization(
            controls,
            amplitude_l2=config.amplitude_l2,
            smoothness_l2=config.smoothness_l2,
        )
        return physics_p - penalty, jnp.array([physics_p, penalty])

    utilities, stats = jax.vmap(one)(noisy_controls)
    objective = jnp.mean(utilities)
    center_stats = stats[0]
    summary = jnp.concatenate(
        [
            jnp.array([objective]),
            center_stats,
            jnp.array([jnp.mean(stats[:, 0]), jnp.std(stats[:, 0])]),
        ]
    )
    return objective, summary


def optimize_physics_grape(
    physics_model: FockPhysicsModel,
    initial_controls: jax.Array,
    key: jax.Array,
    config: GrapeConfig = GrapeConfig(),
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Warm-started AD-GRAPE directly optimizing the Hamiltonian model."""
    key, noise_key = jax.random.split(key)
    noise = config.control_noise_std * jax.random.normal(
        noise_key,
        (config.noise_samples, initial_controls.shape[0]),
    )
    noise = noise.at[0].set(jnp.zeros_like(initial_controls))
    raw_controls = raw_from_bounded_controls(
        initial_controls,
        param_clip=config.param_clip,
    )

    optimizer = optax.lbfgs(memory_size=config.memory_size)
    opt_state = optimizer.init(raw_controls)

    def loss_fn(raw):
        objective, aux = noisy_physics_objective(
            raw,
            physics_model,
            noise,
            config,
        )
        return -objective, aux

    value_and_grad = optax.value_and_grad_from_state(lambda raw: loss_fn(raw)[0])

    def step(carry, _):
        raw, opt_state = carry
        value, grad = value_and_grad(raw, state=opt_state)
        grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        grad_norm = jnp.linalg.norm(grad)
        grad = grad * jnp.minimum(1.0, config.grad_clip_norm / (grad_norm + 1e-12))
        updates, opt_state = optimizer.update(
            grad,
            opt_state,
            raw,
            value=value,
            grad=grad,
            value_fn=lambda z: loss_fn(z)[0],
        )
        raw = optax.apply_updates(raw, updates)
        raw = jnp.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        objective, summary = noisy_physics_objective(
            raw,
            physics_model,
            noise,
            config,
        )
        return (raw, opt_state), summary

    (raw_controls, opt_state), history = jax.lax.scan(
        step,
        (raw_controls, opt_state),
        xs=None,
        length=config.maxiter,
    )
    controls = bounded_controls_from_raw(raw_controls, param_clip=config.param_clip)
    objective, summary = noisy_physics_objective(
        raw_controls,
        physics_model,
        noise,
        config,
    )
    return controls, history, summary, key
