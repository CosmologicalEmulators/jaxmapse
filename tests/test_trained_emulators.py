import os

import jax
import jax.numpy as jnp
import pytest

import jaxmapse
from jaxmapse import w0waCDMCosmology

pytestmark = pytest.mark.artifact

DEFAULT_PARAMS = jnp.array([3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0])


@pytest.fixture
def emulator_setup():
    if os.environ.get("JAXMAPSE_NO_AUTO_DOWNLOAD"):
        pytest.skip("artifact tests require auto-download enabled")

    try:
        root = jaxmapse.artifact_path(jaxmapse.DEFAULT_EMULATOR_ARTIFACT)
        pmm = jaxmapse.load_emulator(str(root / "Pk_lin_mm"))
        pcb = jaxmapse.load_emulator(str(root / "Pk_lin_cb"))
    except Exception as e:
        pytest.fail(f"Failed to load emulator: {e}")

    z = 1.0

    h = DEFAULT_PARAMS[2] / 100.0
    cosmo = w0waCDMCosmology(
        ln10As=DEFAULT_PARAMS[0],
        ns=DEFAULT_PARAMS[1],
        h=h,
        omega_b=DEFAULT_PARAMS[3],
        omega_c=DEFAULT_PARAMS[4],
        m_nu=DEFAULT_PARAMS[5],
        w0=DEFAULT_PARAMS[6],
        wa=DEFAULT_PARAMS[7],
    )
    D = cosmo.D_z(z)

    return pmm, pcb, DEFAULT_PARAMS, z, D


def test_linear_pmm_single_z(emulator_setup):
    """Test running linear Pmm component with a single redshift."""
    pmm, _, params, z, D = emulator_setup

    pk = pmm.get_Pk(params, z, D)

    assert pk.ndim == 1
    assert pk.shape[0] > 0
    assert jnp.all(pk > 0)


def test_linear_pkcb_single_z(emulator_setup):
    """Test running linear Pkcb component with a single redshift."""
    _, pcb, params, z, D = emulator_setup

    pk = pcb.get_Pk(params, z, D)

    assert pk.ndim == 1
    assert pk.shape[0] > 0
    assert jnp.all(pk > 0)


def test_linear_pmm_multiple_z(emulator_setup):
    """Test running linear Pmm component with multiple redshifts."""
    pmm, _, params, _, _ = emulator_setup

    z_vec = jnp.array([0.0, 0.5, 1.0])

    h = params[2] / 100.0
    cosmo = w0waCDMCosmology(
        ln10As=params[0],
        ns=params[1],
        h=h,
        omega_b=params[3],
        omega_c=params[4],
        m_nu=params[5],
        w0=params[6],
        wa=params[7],
    )
    D_vec = cosmo.D_z(z_vec)

    pk_vec = pmm.get_Pk(params, z_vec, D_vec)

    n_k = pmm.k_grid.shape[0]
    assert pk_vec.shape == (len(z_vec), n_k)
    assert jnp.all(pk_vec > 0)


def test_jit_compilation(emulator_setup):
    """Test that emulator methods can be JIT compiled."""
    pmm, _, params, z, D = emulator_setup

    @jax.jit
    def run_emu(p, z, d):
        return pmm.get_Pk(p, z, d)

    pk1 = run_emu(params, z, D)
    pk2 = run_emu(params, z, D)

    assert jnp.allclose(pk1, pk2)


def test_differentiability(emulator_setup):
    """Test that emulator methods are differentiable."""
    pmm, _, params, z, D = emulator_setup

    def loss(p):
        pk = pmm.get_Pk(p, z, D)
        return jnp.sum(pk)

    grad_fn = jax.grad(loss)
    grad = grad_fn(params)

    assert grad.shape == params.shape
    assert jnp.all(jnp.isfinite(grad))
    assert not jnp.allclose(grad, 0.0)


@pytest.fixture
def default_emulator():
    if os.environ.get("JAXMAPSE_NO_AUTO_DOWNLOAD"):
        pytest.skip("artifact tests require auto-download enabled")

    name = jaxmapse.DEFAULT_EMULATOR_ARTIFACT
    try:
        root = jaxmapse.artifact_path(name)
        pmm = jaxmapse.load_emulator(str(root / "Pk_lin_mm"))
        pcb = jaxmapse.load_emulator(str(root / "Pk_lin_cb"))
    except Exception as e:
        pytest.fail(f"Failed to load default emulator '{name}': {e}")
    return pmm, pcb


def _growth_for_default(params, z):
    h = params[2] / 100.0
    cosmo = w0waCDMCosmology(
        ln10As=params[0],
        ns=params[1],
        h=h,
        omega_b=params[3],
        omega_c=params[4],
        m_nu=params[5],
        w0=params[6],
        wa=params[7],
    )
    return cosmo.D_z(z)


def test_default_artifact_component_grids(default_emulator):
    pmm, pcb = default_emulator
    assert pmm.k_grid.shape == (300,)
    assert pcb.k_grid.shape == (300,)


def test_default_artifact_public_workflow_scalar_z(default_emulator):
    pmm, pcb = default_emulator
    z = 0.0
    D = _growth_for_default(DEFAULT_PARAMS, z)

    pk_mm = pmm.get_Pk(DEFAULT_PARAMS, z, D)
    pk_cb = pcb.get_Pk(DEFAULT_PARAMS, z, D)

    assert pk_mm.shape == (300,)
    assert pk_cb.shape == (300,)
    assert jnp.all(jnp.isfinite(pk_mm))
    assert jnp.all(pk_mm > 0.0)


def test_default_artifact_halofit_scalar_z(default_emulator):
    pmm, _ = default_emulator
    z = 0.0

    k_halofit, pk_halofit = jaxmapse.halofit_pmm_from_emulator(
        DEFAULT_PARAMS, z, linear_pmm_emu=pmm
    )

    assert k_halofit.shape == (300,)
    assert pk_halofit.shape == (300,)
    assert jnp.all(jnp.isfinite(pk_halofit))
    assert jnp.all(pk_halofit > 0.0)


def test_default_artifact_halofit_vector_z(default_emulator):
    pmm, _ = default_emulator
    z = jnp.array([0.0, 0.5, 1.0])

    k_halofit, pk_halofit = jaxmapse.halofit_pmm_from_emulator(
        DEFAULT_PARAMS, z, linear_pmm_emu=pmm
    )

    assert k_halofit.shape == (300,)
    assert pk_halofit.shape == (len(z), 300)
    assert jnp.all(jnp.isfinite(pk_halofit))
    assert jnp.all(pk_halofit > 0.0)
