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

The official default artifact is loaded into `jaxmapse.trained_emulators` when the package is imported, unless `JAXMAPSE_NO_AUTO_DOWNLOAD=1` is set.

```python
import jax.numpy as jnp
import jaxmapse
from jaxmapse import w0waCDMCosmology

emu = jaxmapse.trained_emulators[jaxmapse.DEFAULT_EMULATOR_ARTIFACT]

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

k = emu.k_grid
pk_nonlinear = emu.get_Pk(params, z, D)
```

For the default artifact the linear components are stored on a 300-point `k` grid and the nonlinear boost is stored on a 98-point `k` grid. The top-level `emu.get_Pk(...)` interpolates the linear `Pmm` prediction onto the boost grid and returns nonlinear `Pmm` on `emu.k_grid == emu.boost.k_grid`.

To compute nonlinear `Pmm` from the linear emulator with JAX-native Halofit:

```python
k_halofit, pk_halofit = emu.get_halofit_pmm(params, jnp.array([0.0, 0.5, 1.0]))
```

Vector-redshift outputs use the jaxmapse convention `(len(z), len(k))`.

## Artifact loading

`Artifacts.toml` is packaged inside the `jaxmapse` wheel and discovered through Python package resources. To disable eager artifact loading on import, set:

```bash
export JAXMAPSE_NO_AUTO_DOWNLOAD=1
```

This is useful for fast local unit tests or offline imports. Real-artifact tests are marked with the `artifact` pytest marker.

## Background cosmology

`jaxmapse` re-exports background cosmology helpers from `jaxace`, including:

- `w0waCDMCosmology`
- `D_z`, `f_z`, `E_z`, etc.
