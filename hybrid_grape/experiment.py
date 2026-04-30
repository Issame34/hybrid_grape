from __future__ import annotations

import jax
import jax.numpy as jnp

from .physics import FockPhysicsModel


def sample_binomial_measurements(
    true_model: FockPhysicsModel,
    controls: jax.Array,
    key: jax.Array,
    *,
    shots: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Simulate an experiment: one physics evolution plus binomial shot sampling."""
    key, shot_key = jax.random.split(key)
    true_probability = true_model.population_probability(controls)
    successes = jax.random.binomial(
        shot_key,
        n=shots,
        p=true_probability,
        shape=true_probability.shape,
    )
    shot_counts = jnp.full((controls.shape[0],), shots, dtype=jnp.float32)
    return successes.astype(jnp.float32), shot_counts, true_probability, key


def append_dataset(
    controls: jax.Array | None,
    successes: jax.Array | None,
    shots: jax.Array | None,
    physics_probability: jax.Array | None,
    new_controls: jax.Array,
    new_successes: jax.Array,
    new_shots: jax.Array,
    new_physics_probability: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    if controls is None:
        return (
            jnp.asarray(new_controls, dtype=jnp.float32),
            jnp.asarray(new_successes, dtype=jnp.float32),
            jnp.asarray(new_shots, dtype=jnp.float32),
            jnp.asarray(new_physics_probability, dtype=jnp.float32),
        )
    return (
        jnp.concatenate([controls, jnp.asarray(new_controls, dtype=jnp.float32)], axis=0),
        jnp.concatenate([successes, jnp.asarray(new_successes, dtype=jnp.float32)], axis=0),
        jnp.concatenate([shots, jnp.asarray(new_shots, dtype=jnp.float32)], axis=0),
        jnp.concatenate(
            [physics_probability, jnp.asarray(new_physics_probability, dtype=jnp.float32)],
            axis=0,
        ),
    )


def make_local_experiment_batch(
    center_controls: jax.Array,
    key: jax.Array,
    *,
    batch_size: int,
    noise_std: float = 0.025,
    param_clip: float = 2.0,
    include_center: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Generate pulses to evaluate around the GRAPE optimum."""
    center_controls = jnp.asarray(center_controls).reshape((-1,))
    key, noise_key = jax.random.split(key)
    count = batch_size - 1 if include_center else batch_size
    noise = noise_std * jax.random.normal(noise_key, (count, center_controls.shape[0]))
    local = jnp.clip(center_controls[None, :] + noise, -param_clip, param_clip)
    if include_center:
        local = jnp.concatenate([center_controls[None, :], local], axis=0)
    return local, key
