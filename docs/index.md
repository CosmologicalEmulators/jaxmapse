# jaxmapse

JAX implementation of the Matter Power Spectrum Emulator (`Mapse.jl`).

`jaxmapse` provides differentiable emulator evaluation, artifact-backed loading of official trained emulators, and JAX-native Halofit utilities.

## Installation

```bash
pip install .
```

## Official artifact quickstart

The official artifact registry is packaged with the wheel. By default, official trained emulators are loaded into `jaxmapse.trained_emulators` when the package is imported.

```python
import jax.numpy as jnp
import jaxmapse
from jaxmapse import w0waCDMCosmology

emu = jaxmapse.trained_emulators[jaxmapse.DEFAULT_EMULATOR_ARTIFACT]

# Default mnuw0wacdm parameter order:
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

For the default artifact, the linear components use a 300-point grid and the nonlinear boost uses a 98-point grid. The top-level `emu.get_Pk(...)` returns nonlinear `Pmm` on the boost grid after interpolating linear `Pmm` onto that grid.

## Halofit

```python
k_halofit, pk_halofit = emu.get_halofit_pmm(params, jnp.array([0.0, 0.5, 1.0]))
```

Vector-redshift outputs use the jaxmapse convention `(len(z), len(k))`.

## Artifact loading policy

Set this environment variable to disable eager artifact loading on import:

```bash
export JAXMAPSE_NO_AUTO_DOWNLOAD=1
```

Real-artifact tests are marked with the `artifact` pytest marker.
