from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class LowDimCMAESConfig:
    """Settings for black-box CMA-ES in a small pulse neighborhood."""

    num_directions: int = 8
    generations: int = 20
    population_size: int | None = None
    sigma0: float = 0.45
    alpha_clip: float = 2.0
    direction_rms: float = 0.05
    shots_per_candidate: int = 250
    param_clip: float = 2.0
    seed: int = 1234


@dataclass(frozen=True)
class CMAESResult:
    controls: jax.Array
    alpha: jax.Array
    directions: jax.Array
    history: jax.Array
    candidate_history: jax.Array
    best_measured: float


def make_smooth_directions(
    key: jax.Array,
    parameter_shape: tuple[int, int],
    *,
    num_directions: int,
    direction_rms: float,
) -> jax.Array:
    """Create smooth random coefficient directions around a GRAPE pulse."""
    channels, num_coeffs = parameter_shape
    key, subkey = jax.random.split(key)
    dirs = jax.random.normal(subkey, (num_directions, channels, num_coeffs))

    kernel = jnp.array([0.20, 0.60, 0.20], dtype=dirs.dtype)
    for _ in range(3):
        padded = jnp.pad(dirs, ((0, 0), (0, 0), (1, 1)), mode="edge")
        dirs = (
            kernel[0] * padded[:, :, :-2]
            + kernel[1] * padded[:, :, 1:-1]
            + kernel[2] * padded[:, :, 2:]
        )

    dirs = dirs.reshape((num_directions, channels * num_coeffs))
    dirs = dirs - jnp.mean(dirs, axis=1, keepdims=True)
    rms = jnp.sqrt(jnp.mean(dirs**2, axis=1, keepdims=True) + 1e-12)
    return direction_rms * dirs / rms


def pulses_from_alpha(
    center_controls: jax.Array,
    directions: jax.Array,
    alpha: jax.Array,
    *,
    param_clip: float,
) -> jax.Array:
    center_controls = jnp.asarray(center_controls).reshape((-1,))
    alpha = jnp.asarray(alpha)
    if alpha.ndim == 1:
        pulse = center_controls + alpha @ directions
    else:
        pulse = center_controls[None, :] + alpha @ directions
    return jnp.clip(pulse, -param_clip, param_clip)


def _candidate_count(dimension: int, population_size: int | None) -> int:
    if population_size is not None:
        return population_size
    return 4 + int(3 * np.log(dimension))


def optimize_lowdim_cma_es(
    center_controls: jax.Array,
    directions: jax.Array,
    measure_on_experiment: Callable,
    key: jax.Array,
    config: LowDimCMAESConfig = LowDimCMAESConfig(),
) -> tuple[CMAESResult, jax.Array]:
    """
    Maximize measured probability over u = u0 + V alpha.

    The measurement callable must accept `(controls, key, shots)` and return
    `(successes, shot_counts, diagnostics, key)`, matching experiment.py.
    """
    dimension = directions.shape[0]
    lam = _candidate_count(dimension, config.population_size)
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mueff = 1.0 / np.sum(weights**2)

    cc = (4.0 + mueff / dimension) / (dimension + 4.0 + 2.0 * mueff / dimension)
    cs = (mueff + 2.0) / (dimension + mueff + 5.0)
    c1 = 2.0 / ((dimension + 1.3) ** 2 + mueff)
    cmu = min(
        1.0 - c1,
        2.0
        * (mueff - 2.0 + 1.0 / mueff)
        / ((dimension + 2.0) ** 2 + mueff),
    )
    damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dimension + 1.0)) - 1.0) + cs
    chi_n = np.sqrt(dimension) * (1.0 - 1.0 / (4.0 * dimension) + 1.0 / (21.0 * dimension**2))

    mean = np.zeros(dimension)
    sigma = config.sigma0
    cov = np.eye(dimension)
    pc = np.zeros(dimension)
    ps = np.zeros(dimension)
    best_alpha = np.zeros(dimension)
    best_measured = -np.inf
    rows = []
    candidate_rows = []
    rng = np.random.default_rng(config.seed)

    for generation in range(config.generations):
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-12)
        transform = eigvecs @ np.diag(np.sqrt(eigvals))
        inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        z = rng.normal(size=(lam, dimension))
        y = z @ transform.T
        candidates = np.clip(mean[None, :] + sigma * y, -config.alpha_clip, config.alpha_clip)

        controls = pulses_from_alpha(
            center_controls,
            directions,
            jnp.asarray(candidates),
            param_clip=config.param_clip,
        )
        successes, shot_counts, diagnostics, key = measure_on_experiment(
            controls,
            key,
            config.shots_per_candidate,
        )
        measured = np.asarray(successes / shot_counts, dtype=float)
        successes_np = np.asarray(successes, dtype=float)
        shots_np = np.asarray(shot_counts, dtype=float)
        order = np.argsort(measured)[::-1]

        if measured[order[0]] > best_measured:
            best_measured = float(measured[order[0]])
            best_alpha = candidates[order[0]].copy()

        old_mean = mean.copy()
        selected = candidates[order[:mu]]
        mean = np.sum(selected * weights[:, None], axis=0)
        y_w = (mean - old_mean) / sigma

        ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (inv_sqrt @ y_w)
        hsig = float(
            np.linalg.norm(ps)
            / np.sqrt(1.0 - (1.0 - cs) ** (2.0 * (generation + 1)))
            / chi_n
            < (1.4 + 2.0 / (dimension + 1.0))
        )
        pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * y_w

        selected_steps = (selected - old_mean[None, :]) / sigma
        rank_mu = np.einsum("i,ij,ik->jk", weights, selected_steps, selected_steps)
        cov = (
            (1.0 - c1 - cmu) * cov
            + c1 * (np.outer(pc, pc) + (1.0 - hsig) * cc * (2.0 - cc) * cov)
            + cmu * rank_mu
        )
        cov = 0.5 * (cov + cov.T)
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1.0))

        true_best = np.nan
        if diagnostics is not None:
            diag = np.asarray(diagnostics)
            if diag.shape == measured.shape:
                true_best = float(diag[order[0]])
        else:
            diag = np.full_like(measured, np.nan)

        for candidate_index in range(lam):
            candidate_rows.append(
                [
                    generation,
                    candidate_index,
                    float(measured[candidate_index]),
                    float(successes_np[candidate_index]),
                    float(shots_np[candidate_index]),
                    float(diag[candidate_index]),
                    float(order.tolist().index(candidate_index)),
                ]
                + [float(x) for x in candidates[candidate_index]]
            )

        rows.append(
            [
                generation,
                float(np.max(measured)),
                float(np.mean(measured)),
                best_measured,
                true_best,
                float(sigma),
            ]
        )

    best_controls = pulses_from_alpha(
        center_controls,
        directions,
        jnp.asarray(best_alpha),
        param_clip=config.param_clip,
    )
    result = CMAESResult(
        controls=best_controls,
        alpha=jnp.asarray(best_alpha),
        directions=directions,
        history=jnp.asarray(rows),
        candidate_history=jnp.asarray(candidate_rows),
        best_measured=best_measured,
    )
    return result, key
