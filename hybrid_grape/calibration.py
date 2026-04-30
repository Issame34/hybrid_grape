from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from .config import khz_to_rad_per_us
from .physics import FockPhysicsModel, PhysicsParams, SimulationConfig


@dataclass(frozen=True)
class CalibrationConfig:
    """Low-dimensional physical parameter correction fit from measured pulses."""

    delta_chi_khz: float = 0.0
    delta_kerr_khz: float = 0.0
    cavity_detuning_khz: float = 0.0
    qubit_detuning_khz: float = 0.0
    qubit_drive_scale: float = 0.0
    cavity_drive_scale: float = 0.0
    cavity_phase: float = 0.0

    @classmethod
    def from_array(cls, values: np.ndarray | jax.Array) -> "CalibrationConfig":
        values = np.asarray(values, dtype=float)
        return cls(
            delta_chi_khz=float(values[0]),
            delta_kerr_khz=float(values[1]),
            cavity_detuning_khz=float(values[2]),
            qubit_detuning_khz=float(values[3]),
            qubit_drive_scale=float(values[4]),
            cavity_drive_scale=float(values[5]),
            cavity_phase=float(values[6]),
        )

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.delta_chi_khz,
                self.delta_kerr_khz,
                self.cavity_detuning_khz,
                self.qubit_detuning_khz,
                self.qubit_drive_scale,
                self.cavity_drive_scale,
                self.cavity_phase,
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class CalibrationFitResult:
    config: CalibrationConfig
    loss: float
    success: bool
    message: str
    predictions: jax.Array
    initial_loss: float


def calibrated_physics_params(
    base: PhysicsParams,
    calibration: CalibrationConfig,
) -> PhysicsParams:
    return PhysicsParams(
        chi=base.chi + khz_to_rad_per_us(calibration.delta_chi_khz),
        cavity_self_kerr=base.cavity_self_kerr
        + khz_to_rad_per_us(calibration.delta_kerr_khz),
        cavity_detuning=base.cavity_detuning
        + khz_to_rad_per_us(calibration.cavity_detuning_khz),
        qubit_detuning=base.qubit_detuning
        + khz_to_rad_per_us(calibration.qubit_detuning_khz),
        mu_qub=base.mu_qub * (1.0 + calibration.qubit_drive_scale),
        mu_cav=base.mu_cav * (1.0 + calibration.cavity_drive_scale),
        grape_dispersive_frame=base.grape_dispersive_frame,
        grape_cavity_iq=base.grape_cavity_iq,
        cavity_phase=base.cavity_phase + calibration.cavity_phase,
    )


def make_calibrated_model(
    sim_config: SimulationConfig,
    base_params: PhysicsParams,
    calibration: CalibrationConfig,
) -> FockPhysicsModel:
    return FockPhysicsModel(
        sim_config,
        calibrated_physics_params(base_params, calibration),
    )


def calibration_predictions(
    sim_config: SimulationConfig,
    base_params: PhysicsParams,
    calibration_values: np.ndarray | jax.Array,
    controls: jax.Array,
) -> jax.Array:
    calibration = CalibrationConfig.from_array(calibration_values)
    model = make_calibrated_model(sim_config, base_params, calibration)
    return model.population_probability(controls)


def fit_parametric_calibration(
    sim_config: SimulationConfig,
    base_params: PhysicsParams,
    controls: jax.Array,
    measured_probability: jax.Array,
    shots: jax.Array | None = None,
    *,
    initial: CalibrationConfig = CalibrationConfig(),
    bounds: tuple[tuple[float, float], ...] = (
        (-40.0, 40.0),
        (-2.0, 2.0),
        (-40.0, 40.0),
        (-40.0, 40.0),
        (-0.30, 0.30),
        (-0.30, 0.30),
        (-0.50, 0.50),
    ),
    num_restarts: int = 8,
    seed: int = 123,
) -> CalibrationFitResult:
    controls = jnp.asarray(controls)
    measured_probability_np = np.asarray(measured_probability, dtype=float)
    if shots is None:
        weights = np.ones_like(measured_probability_np)
    else:
        weights = np.asarray(shots, dtype=float)
        weights = weights / np.mean(weights)

    bounds_array = np.asarray(bounds, dtype=float)
    center = 0.5 * (bounds_array[:, 0] + bounds_array[:, 1])
    scale = 0.5 * (bounds_array[:, 1] - bounds_array[:, 0])

    def values_from_unit(unit_values: np.ndarray) -> np.ndarray:
        unit_values = np.clip(np.asarray(unit_values, dtype=float), -1.0, 1.0)
        return center + scale * unit_values

    def unit_from_values(values: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(values, dtype=float) - center) / scale, -1.0, 1.0)

    def objective_values(values: np.ndarray) -> float:
        pred = calibration_predictions(
            sim_config,
            base_params,
            values,
            controls,
        )
        pred_np = np.asarray(pred, dtype=float)
        err = pred_np - measured_probability_np
        return float(np.mean(weights * err**2))

    def objective_unit(unit_values: np.ndarray) -> float:
        return objective_values(values_from_unit(unit_values))

    initial_values = initial.as_array()
    initial_unit = unit_from_values(initial_values)
    initial_loss = objective_values(initial_values)

    starts = [initial_unit]
    rng = np.random.default_rng(seed)
    for _ in range(num_restarts):
        starts.append(rng.uniform(-0.75, 0.75, size=initial_unit.shape))

    best_result = None
    for start in starts:
        result = minimize(
            objective_unit,
            start,
            method="Powell",
            bounds=tuple((-1.0, 1.0) for _ in range(len(start))),
            options={
                "maxiter": 120,
                "xtol": 1e-3,
                "ftol": 1e-5,
                "disp": False,
            },
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    fitted_values = values_from_unit(best_result.x)
    fitted = CalibrationConfig.from_array(fitted_values)
    predictions = calibration_predictions(
        sim_config,
        base_params,
        fitted_values,
        controls,
    )
    return CalibrationFitResult(
        config=fitted,
        loss=float(best_result.fun),
        success=bool(best_result.success),
        message=str(best_result.message),
        predictions=predictions,
        initial_loss=float(initial_loss),
    )
