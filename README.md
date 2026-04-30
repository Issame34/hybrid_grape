# GRAPE + Low-Dimensional CMA-ES

Self-contained JAX project for closed-loop Fock-state preparation.

This project is intentionally small. The loop is:

1. Run GRAPE with the fixed Hamiltonian model to get a good center pulse `u0`.
2. Build 5-10 smooth low-dimensional directions `V` in B-spline coefficient space.
3. Run black-box CMA-ES on `u = u0 + V alpha`.
4. Evaluate each candidate with the fake experiment now, then replace that hook with hardware.

This keeps the experimental optimizer low-dimensional while preserving the
physics-informed GRAPE pulse as the center of the search.

There is no RBF model in this version. The fake experiment is a second
mismatched Hamiltonian model, which mimics hardware disagreement with the GRAPE
model.

## Main Files

- `hybrid_grape/physics.py`: Hamiltonian model and B-spline controls
- `hybrid_grape/grape.py`: physics-only GRAPE optimizer
- `hybrid_grape/cmaes.py`: low-dimensional CMA-ES around the GRAPE pulse
- `hybrid_grape/experiment.py`: fake binomial experiment
- `scripts/grape_cmaes.py`: command-line run
- `grape_cmaes.ipynb`: notebook with plots and metrics

## Run

```powershell
uv run python scripts/grape_cmaes.py
```
