# jaxmapse

JAX-based implementation of the MAtter Power Spectrum Emulator (Mapse), mirroring the [Mapse.jl](https://github.com/CosmologicalEmulators/Mapse.jl) repository.

## Overview

`jaxmapse` provides a high-performance, differentiable emulator for the matter power spectrum. By leveraging JAX, it supports:
- **Automatic Differentiation**: Compute gradients of the power spectrum with respect to cosmological parameters.
- **Just-In-Time (JIT) Compilation**: Near-native execution speeds.
- **Vectorization**: Efficiently process batches of cosmological parameters or redshifts using `vmap`.

## Installation

```bash
# Since this relies on a local jaxace, ensure it is accessible
pip install -e .
```

## Usage

```python
import jax.numpy as jnp
from jaxmapse import load_emulator, w0waCDMCosmology, D_z

# Load the composite emulator
emu = load_emulator("path/to/model", structure="PkEmulator")

# Define cosmology and compute background
cosmo_params = jnp.array([0.3, 0.7, 0.05, 0.96, 0.67]) # example input
cosmo_bg = w0waCDMCosmology(h=0.67, ωb=0.022, ωc=0.12)
z = 1.0
D = D_z(z, cosmo_bg)

# Get P(k)
pk = emu.get_Pk(cosmo_params, z, D)
```

## Background Cosmology

`jaxmapse` re-exports background cosmology functions from `jaxace` for convenience:
- `w0waCDMCosmology`
- `D_z`, `f_z`, `E_z`, etc.