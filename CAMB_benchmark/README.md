# CAMB HMCode references

This directory contains fixed CAMB HMCode2020 references and comparison scripts
for the JAX implementation.

## Fiducial reference

`generate_fixtures.py` reproduces the original fiducial DMO and feedback files:

```bash
python CAMB_benchmark/generate_fixtures.py
```

The grid has 150 redshifts over `0 <= z <= 3.5` and 128 wavenumbers over
`1e-3 <= k/(h Mpc^-1) <= 10`.

## Ten-cosmology reference set

`multicosmo/cases.json` records ten cosmologies spanning density parameters,
neutrino mass, Hubble parameter, primordial parameters, `w0`, `wa`, and
feedback temperature. Every case satisfies the emulator trust condition
`w0 + wa < 0`.

Each `multicosmo/case_*.txt` file has shape `(150, 256)`:

- columns `0:128`: CAMB `mead2020` DMO `P(k,z)` in Mpc^3;
- columns `128:256`: CAMB `mead2020_feedback` `P(k,z)` in Mpc^3.

The redshift and case-dependent physical wavenumber grids are defined in the
manifest. Regenerate the references only when intentionally changing the CAMB
reference configuration:

```bash
python CAMB_benchmark/generate_multicosmo_fixtures.py
```

Compare the direct and production `N_coarse=24` smart pipelines against the
saved files with:

```bash
python CAMB_benchmark/compare_multicosmo.py
```

The comparison reports direct-versus-CAMB, smart-versus-direct, and
smart-versus-CAMB statistics. CI requires direct-versus-CAMB and
smart-versus-CAMB errors below 1.1%, and smart-versus-direct errors below 0.5%.
Detailed results are written to `multicosmo/comparison_results.json`.

## Timing

`benchmark.py` reports the existing fiducial smart-path timings. It must be run
from this repository checkout so the local `jaxmapse` implementation is used.
