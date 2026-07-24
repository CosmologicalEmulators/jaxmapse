# jaxmapse

JAX-based implementation of the MAtter Power Spectrum Emulator (Mapse), mirroring the [Mapse.jl](https://github.com/CosmologicalEmulators/Mapse.jl) repository.

## Overview

`jaxmapse` provides differentiable matter-power-spectrum emulators with JAX:

- artifact-backed loading of official trained emulators,
- PCA-compressed linear and nonlinear components,
- JAX-native Halofit for nonlinear total-matter spectra,
- automatic differentiation, JIT compilation, and vectorized redshift evaluation.

`jaxmapse` enables 64-bit JAX mode at import time because the cosmology kernels and emulator postprocessing are calibrated for Float64 precision.

## Installation

```bash
pip install .
```

## Quickstart

To use `jaxmapse`, resolve the emulator artifact path and load the desired transfer function component explicitly:

```python
import jax.numpy as jnp
import jaxmapse
from jaxmapse import w0waCDMCosmology

# Resolve path to the default mnuw0wacdm_class artifact and load components
root = jaxmapse.artifact_path(jaxmapse.DEFAULT_EMULATOR_ARTIFACT)
pmm = jaxmapse.load_emulator(root / "Pk_lin_mm")
pcb = jaxmapse.load_emulator(root / "Pk_lin_cb")

# Parameter order for the default mnuw0wacdm artifact:
# [ln10As, ns, H0, omega_b, omega_c, Mnu, w0, wa]
params = jnp.array([3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0])
z = 0.0

cosmo = w0waCDMCosmology(
    ln10As=params[0],
    ns=params[1],
    h=params[2] / 100.0,
    omega_b=params[3],
    omega_c=params[4],
    m_nu=params[5],
    w0=params[6],
    wa=params[7],
)
D = cosmo.D_z(z)

# Evaluate the linear total-matter power spectrum on its k-grid (300 points)
k_grid = pmm.k_grid
pk_linear = pmm(params, z, D)
```

To compute nonlinear `Pmm` from the linear emulator using JAX-native Halofit:

```python
k_halofit, pk_halofit = jaxmapse.halofit_pmm_from_emulator(
    params, z, D, linear_pmm_emu=pmm
)
```

Vector-redshift outputs use the jaxmapse convention `(len(z), len(k))`.



## Background cosmology

`jaxmapse` re-exports background cosmology helpers from `jaxace`, including:

- `w0waCDMCosmology`
- `D_z`, `f_z`, `E_z`, etc.
