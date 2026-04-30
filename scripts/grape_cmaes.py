from __future__ import annotations

import jax
import jax.numpy as jnp

from hybrid_grape import (
    FockPhysicsModel,
    GrapeConfig,
    LowDimCMAESConfig,
    PhysicsParams,
    SimulationConfig,
    fit_parametric_calibration,
    make_calibrated_model,
    make_smooth_directions,
    optimize_physics_grape,
    optimize_lowdim_cma_es,
    sample_binomial_measurements,
)
from hybrid_grape.config import khz_to_rad_per_us


def main() -> None:
    jax.config.update("jax_enable_x64", False)
    key = jax.random.key(1234)

    sim_config = SimulationConfig(
        n_cav=25,
        target_n=2,
        initial_cavity_n=0,
        initial_qubit_state=0,
        t_drive=1.408,
        ndt_drive=80,
        num_coeffs=20,
        spline_degree=2,
        spline_skip_left=2,
        spline_skip_right=2,
        param_clip=2.0,
    )

    physics_params = PhysicsParams()
    physics_model = FockPhysicsModel(sim_config, physics_params)

    true_params = PhysicsParams(
        chi=physics_params.chi + khz_to_rad_per_us(18.0),
        cavity_self_kerr=physics_params.cavity_self_kerr + khz_to_rad_per_us(0.65),
        cavity_detuning=khz_to_rad_per_us(14.0),
        qubit_detuning=khz_to_rad_per_us(-18.0),
        mu_qub=physics_params.mu_qub * 1.080,
        mu_cav=physics_params.mu_cav * 0.900,
        cavity_phase=0.18,
    )
    true_model = FockPhysicsModel(sim_config, true_params)

    key, init_key = jax.random.split(key)
    initial_controls = 0.02 * jax.random.normal(init_key, (physics_model.parameter_size,))
    initial_controls = jnp.clip(initial_controls, -sim_config.param_clip, sim_config.param_clip)

    grape_controls, _, grape_summary, key = optimize_physics_grape(
        physics_model,
        initial_controls,
        key,
        GrapeConfig(
            maxiter=80,
            noise_samples=4,
            control_noise_std=0.0,
            param_clip=sim_config.param_clip,
        ),
    )

    print("GRAPE physics probability:", float(physics_model.photon_probability(grape_controls)))
    print("GRAPE hidden true probability:", float(true_model.photon_probability(grape_controls)))
    print("GRAPE objective summary:", grape_summary)

    key, direction_key = jax.random.split(key)
    directions = make_smooth_directions(
        direction_key,
        physics_model.parameter_shape,
        num_directions=8,
        direction_rms=0.035,
    )

    def measure_on_experiment(controls, key, shots):
        return sample_binomial_measurements(true_model, controls, key, shots=shots)

    cma_result, key = optimize_lowdim_cma_es(
        grape_controls,
        directions,
        measure_on_experiment,
        key,
        LowDimCMAESConfig(
            num_directions=8,
            generations=18,
            sigma0=0.40,
            alpha_clip=2.0,
            direction_rms=0.035,
            shots_per_candidate=250,
            param_clip=sim_config.param_clip,
            seed=7,
        ),
    )

    print("CMA-ES best measured probability:", cma_result.best_measured)
    print("CMA-ES hidden true probability:", float(true_model.photon_probability(cma_result.controls)))
    print("CMA-ES physics probability:", float(physics_model.photon_probability(cma_result.controls)))
    print("history columns: generation, batch_best, batch_mean, best_seen, true_batch_best, sigma")
    print(cma_result.history)

    candidate_history = cma_result.candidate_history
    alpha_history = candidate_history[:, 7:]
    from hybrid_grape.cmaes import pulses_from_alpha

    candidate_controls = pulses_from_alpha(
        grape_controls,
        directions,
        alpha_history,
        param_clip=sim_config.param_clip,
    )
    measured_probability = candidate_history[:, 2]
    shots = candidate_history[:, 4]

    calibration_fit = fit_parametric_calibration(
        sim_config,
        physics_params,
        candidate_controls,
        measured_probability,
        shots,
    )
    calibrated_model = make_calibrated_model(
        sim_config,
        physics_params,
        calibration_fit.config,
    )
    calibrated_controls, _, _, key = optimize_physics_grape(
        calibrated_model,
        grape_controls,
        key,
        GrapeConfig(
            maxiter=80,
            noise_samples=4,
            control_noise_std=0.0,
            param_clip=sim_config.param_clip,
        ),
    )

    print("calibration fit:", calibration_fit.config)
    print("calibrated GRAPE calibrated-model probability:", float(calibrated_model.photon_probability(calibrated_controls)))
    print("calibrated GRAPE hidden true probability:", float(true_model.photon_probability(calibrated_controls)))


if __name__ == "__main__":
    main()
