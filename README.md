# GRAPE + CMA-ES + Parametric Calibration

This project is a compact closed-loop control prototype for Fock-state
preparation with a cavity-qubit Hamiltonian model.

The goal is to separate three ideas clearly:

1. `GRAPE` optimizes the physics model.
2. `CMA-ES` improves the GRAPE pulse using measured/fake-experimental results.
3. `Parametric calibration` fits a small set of physical model errors, then
   reruns GRAPE with the corrected model.

There is no RBF or black-box residual model in this version.

## Core Idea

The project uses two physics models:

- `physics_model`: the model GRAPE believes.
- `experiment_model`: a deliberately mismatched hidden model used to mimic the
  real experiment.

In the fake experiment:

```text
P_physics(u) = probability predicted by the GRAPE model
P_hybrid(u)  = probability from the mismatched fake experiment model
P_exp(u)     = successes / shots sampled from P_hybrid(u)
```

So the loop mimics hardware:

```text
GRAPE optimizes P_physics(u)
        |
        v
get center pulse u0
        |
        v
CMA-ES searches u = u0 + V alpha
        |
        v
fake experiment returns P_exp = successes / shots
        |
        v
fit physical mismatch parameters
        |
        v
rerun GRAPE with calibrated model
```

## What Gets Fitted

Parametric calibration fits a low-dimensional physical correction:

```text
p = (
  delta_chi_khz,
  delta_kerr_khz,
  cavity_detuning_khz,
  qubit_detuning_khz,
  qubit_drive_scale,
  cavity_drive_scale,
  cavity_phase
)
```

The calibrated model is:

```text
P_calibrated(u) = P_physics(u; p_fit)
```

The fit minimizes measured prediction error over tested pulses:

```text
sum_k shots_k * (P_calibrated(u_k) - P_exp(u_k))^2
```

## Main Files

- `hybrid_grape/physics.py`: Hamiltonian model, B-spline controls, pulse
  evolution, photon-number probability.
- `hybrid_grape/grape.py`: physics-only GRAPE optimizer.
- `hybrid_grape/cmaes.py`: low-dimensional CMA-ES around a GRAPE pulse.
- `hybrid_grape/calibration.py`: parametric calibration of physical mismatch.
- `hybrid_grape/experiment.py`: fake binomial measurements from a model.
- `scripts/grape_cmaes.py`: command-line run of the full workflow.
- `grape_cmaes.ipynb`: notebook with metrics and plots.

## Run The Notebook

Open:

```text
grape_cmaes.ipynb
```

Then restart the kernel and run from the top.

Important notebook sections:

- `Sanity Check: The Two Models Are Different`
- `Run GRAPE Center Pulse`
- `Run Low-Dimensional CMA-ES`
- `Full Metrics Dashboard`
- `Parametric Calibration`
- `Calibrated GRAPE`

## Run From Command Line

```powershell
cd C:\Users\QCircuits\Documents\Codex\2026-04-30\hi-i-want-to-start-the\Hybrid_grape
uv run python scripts/grape_cmaes.py
```

The script prints:

- GRAPE probability under the physics model.
- GRAPE probability under the hidden fake experiment model.
- CMA-ES best measured probability.
- fitted calibration parameters.
- calibrated GRAPE probability under the calibrated model.
- calibrated GRAPE probability under the hidden fake experiment model.

## Current Workflow

1. Build `physics_model` from `PhysicsParams()`.
2. Build `experiment_model` from a deliberately mismatched `PhysicsParams`.
3. Run `optimize_physics_grape(...)` to get `grape_controls`.
4. Generate smooth directions with `make_smooth_directions(...)`.
5. Run `optimize_lowdim_cma_es(...)` on `u = u0 + V alpha`.
6. Reconstruct the tested pulse dataset from `cma_result.candidate_history`.
7. Fit physical corrections with `fit_parametric_calibration(...)`.
8. Build `calibrated_model`.
9. Run `optimize_physics_grape(...)` again using the calibrated model.

## Important Notes

- CMA-ES learns a better pulse.
- Parametric calibration learns a better model.
- These are different. The calibrated model can be reused for the next GRAPE
  round, while a CMA-ES pulse is just one optimized pulse.
- If calibration does not improve the loss, the dataset is probably not
  informative enough. Add more diverse calibration pulses, not just local
  pulses around one GRAPE solution.

## Metrics To Watch

The notebook plots:

- `P_physics` vs `P_hybrid`
- `P_hybrid` vs `P_exp`
- `P_physics` vs `P_exp`
- `P_calibrated` vs `P_exp`
- MSE before and after calibration
- CMA-ES shot history
- best measured pulse over generations
- GRAPE vs CMA-ES vs calibrated GRAPE

The most important checks are:

```text
mean |P_hybrid - P_physics|
MSE P_physics vs P_exp
MSE P_calibrated vs P_exp
hidden P_hybrid of calibrated GRAPE pulse
```
