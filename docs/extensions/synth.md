# Optimization-based phase synthesis

Given a target far-field magnitude pattern, find per-cell phases that
approximately produce it. Three backends:

| Backend | Method | Always available? |
|---|---|:-:|
| `synth_phase_scipy` | L-BFGS-B with analytic gradients | ✓ |
| `synth_phase_cvxpy` | SOCP convex relaxation (\|c\|≤1) | needs `[synth]` |
| `synth_phase_jax` | Adam-style autodiff GD | needs `[synth]` |

```mermaid
graph LR
  Target[target_magnitude<br/>u-v grid] --> Loss[loss = Σ w·(\|E\| − target)²]
  Array[Reflectarray] --> Loss
  Loss --> Backend{Backend}
  Backend --> SC[scipy<br/>L-BFGS-B]
  Backend --> CV[cvxpy<br/>SOCP]
  Backend --> JX[jax<br/>autodiff]
  SC --> Phase[phase matrix]
  CV --> Phase
  JX --> Phase
```

## Broadside-target synthesis

![](../img/synth_broadside.png)

A Gaussian-bump target on the (u, v) grid; the synthesizer returns a 12×12
cell-phase matrix that focuses a 28 GHz reflectarray to broadside.

## API

```python
import numpy as np
import fresnelants as fa
from fresnelants.synth import synth_phase_scipy

array = fa.Reflectarray(focal_length=0.20, design_freq=28e9, nx=12, ny=12)

# Target: broadside-only Gaussian bump.
M = 32
u, v = np.meshgrid(np.linspace(-1, 1, M), np.linspace(-1, 1, M), indexing="xy")
target = np.exp(-200 * (u**2 + v**2))

result = synth_phase_scipy(target, array, freq=28e9, max_iter=80)
print(result.phase.shape)   # (12, 12)
print(result.final_loss)    # final fit error
```

## When to use which backend

- **scipy** — fast for arrays up to ~ 32×32; analytic gradients keep it
  competitive with autodiff at small sizes.
- **cvxpy** — exact convex relaxation; gives a magnitude-relaxed optimum.
  Best for proving lower bounds on achievable error.
- **jax** — > 32×32 arrays where the JIT-compiled gradient pays off; large
  initial cost (compile) but constant-time per step thereafter.
