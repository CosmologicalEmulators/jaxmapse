import os

import jax
import jax.numpy as jnp
import pytest

import jaxmapse
from jaxmapse import w0waCDMCosmology

pytestmark = pytest.mark.artifact

EMULATOR_NAME = "trained_mapse_class_hmcode_mnuw0waOkcdm"
DEFAULT_PARAMS = jnp.array([3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0])


@pytest.fixture
def emulator_setup():
    if os.environ.get("JAXMAPSE_NO_AUTO_DOWNLOAD"):
        pytest.skip("artifact tests require auto-download enabled")

    # Strict check: Emulator MUST be present
    if EMULATOR_NAME not in jaxmapse.trained_emulators:
        pytest.fail(
            f"Emulator '{EMULATOR_NAME}' is not in the trained_emulators dictionary."
        )

    emu = jaxmapse.trained_emulators[EMULATOR_NAME]

    if emu is None:
        pytest.fail(f"Emulator '{EMULATOR_NAME}' failed to load (value is None).")

    # Standard Planck 2018 + log10T_heat parameters
    # Order: [ln10As, ns, H0, ombh2, omch2, Mν, w0, wa, Omega_k, log10T_heat]
    p_hmcode = jnp.array(
        [3.044, 0.9665, 67.66, 0.02242, 0.11933, 0.06, -1.0, 0.0, 0.0, 7.8]
    )

    z = 1.0

    h = p_hmcode[2] / 100.0
    cosmo = w0waCDMCosmology(
        ln10As=p_hmcode[0],
        ns=p_hmcode[1],
        h=h,
        omega_b=p_hmcode[3],
        omega_c=p_hmcode[4],
        m_nu=p_hmcode[5],
        w0=p_hmcode[6],
        wa=p_hmcode[7],
        omega_k=p_hmcode[8] * h**2,
    )
    D = cosmo.D_z(z)

    return emu, p_hmcode, z, D


@pytest.mark.parametrize(
    "component_method", ["get_linear_pmm", "get_linear_pkcb", "get_Pk"]  # Composite
)
def test_emulator_components_single_z(emulator_setup, component_method):
    """Test running each emulator component with a single redshift."""
    emu, p_hmcode, z, D = emulator_setup

    method = getattr(emu, component_method)
    pk = method(p_hmcode, z, D)

    assert pk.ndim == 1
    assert pk.shape[0] > 0
    assert jnp.all(pk > 0)


def test_boost_component_single_z(emulator_setup):
    """Test running the Boost component specifically (different method signature on Boost object)."""
    emu, p_hmcode, z, D = emulator_setup

    pk = emu.boost.get_Pk(p_hmcode, z, D)

    assert pk.ndim == 1
    assert pk.shape[0] > 0
    assert jnp.all(pk > 0)


@pytest.mark.parametrize(
    "component_method", ["get_linear_pmm", "get_linear_pkcb", "get_Pk"]  # Composite
)
def test_emulator_components_multiple_z(emulator_setup, component_method):
    """Test running each emulator component with multiple redshifts."""
    emu, p_hmcode, _, _ = emulator_setup

    z_vec = jnp.array([0.0, 0.5, 1.0])

    h = p_hmcode[2] / 100.0
    cosmo = w0waCDMCosmology(
        ln10As=p_hmcode[0],
        ns=p_hmcode[1],
        h=h,
        omega_b=p_hmcode[3],
        omega_c=p_hmcode[4],
        m_nu=p_hmcode[5],
        w0=p_hmcode[6],
        wa=p_hmcode[7],
        omega_k=p_hmcode[8] * h**2,
    )
    D_vec = cosmo.D_z(z_vec)

    method = getattr(emu, component_method)
    pk_vec = method(p_hmcode, z_vec, D_vec)

    n_k = emu.linear_pmm.k_grid.shape[0]
    assert pk_vec.shape == (len(z_vec), n_k)
    assert jnp.all(pk_vec > 0)


def test_boost_component_multiple_z(emulator_setup):
    """Test running the Boost component with multiple redshifts."""
    emu, p_hmcode, _, _ = emulator_setup

    z_vec = jnp.array([0.0, 0.5, 1.0])
    h = p_hmcode[2] / 100.0
    cosmo = w0waCDMCosmology(
        ln10As=p_hmcode[0],
        ns=p_hmcode[1],
        h=h,
        omega_b=p_hmcode[3],
        omega_c=p_hmcode[4],
        m_nu=p_hmcode[5],
        w0=p_hmcode[6],
        wa=p_hmcode[7],
        omega_k=p_hmcode[8] * h**2,
    )
    D_vec = cosmo.D_z(z_vec)

    pk_vec = emu.boost.get_Pk(p_hmcode, z_vec, D_vec)

    n_k = emu.boost.k_grid.shape[0]
    assert pk_vec.shape == (len(z_vec), n_k)
    assert jnp.all(pk_vec > 0)


@pytest.mark.parametrize(
    "component_method", ["get_linear_pmm", "get_linear_pkcb", "get_Pk"]
)
def test_jit_compilation(emulator_setup, component_method):
    """Test that emulator methods can be JIT compiled."""
    emu, p_hmcode, z, D = emulator_setup

    method = getattr(emu, component_method)

    # Define a jittable wrapper
    @jax.jit
    def run_emu(p, z, d):
        return method(p, z, d)

    # Run once to compile
    pk1 = run_emu(p_hmcode, z, D)
    # Run again
    pk2 = run_emu(p_hmcode, z, D)

    assert jnp.allclose(pk1, pk2)


def test_boost_jit_compilation(emulator_setup):
    """Test that Boost component can be JIT compiled."""
    emu, p_hmcode, z, D = emulator_setup

    @jax.jit
    def run_boost(p, z, d):
        return emu.boost.get_Pk(p, z, d)

    pk1 = run_boost(p_hmcode, z, D)
    pk2 = run_boost(p_hmcode, z, D)

    assert jnp.allclose(pk1, pk2)


@pytest.mark.parametrize(
    "component_method", ["get_linear_pmm", "get_linear_pkcb", "get_Pk"]
)
def test_differentiability(emulator_setup, component_method):
    """Test that emulator methods are differentiable with respect to input parameters."""
    emu, p_hmcode, z, D = emulator_setup
    method = getattr(emu, component_method)

    def loss(p):
        pk = method(p, z, D)
        return jnp.sum(pk)

    grad_fn = jax.grad(loss)
    grad = grad_fn(p_hmcode)

    assert grad.shape == p_hmcode.shape
    assert jnp.all(jnp.isfinite(grad))
    # Check that gradient is not all zeros (which would mean no sensitivity)
    assert not jnp.allclose(grad, 0.0)


def test_boost_differentiability(emulator_setup):
    """Test that Boost component is differentiable."""
    emu, p_hmcode, z, D = emulator_setup

    def loss(p):
        pk = emu.boost.get_Pk(p, z, D)
        return jnp.sum(pk)

    grad_fn = jax.grad(loss)
    grad = grad_fn(p_hmcode)

    assert grad.shape == p_hmcode.shape
    assert jnp.all(jnp.isfinite(grad))
    assert not jnp.allclose(grad, 0.0)


@pytest.fixture
def default_emulator():
    if os.environ.get("JAXMAPSE_NO_AUTO_DOWNLOAD"):
        pytest.skip("artifact tests require auto-download enabled")

    name = jaxmapse.DEFAULT_EMULATOR_ARTIFACT
    if name not in jaxmapse.trained_emulators:
        pytest.fail(f"Default emulator '{name}' is not in trained_emulators.")

    emu = jaxmapse.trained_emulators[name]
    if emu is None:
        pytest.fail(f"Default emulator '{name}' failed to load (value is None).")
    return emu


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
    assert default_emulator.linear_pmm.k_grid.shape == (300,)
    assert default_emulator.linear_pkcb.k_grid.shape == (300,)
    assert default_emulator.boost.k_grid.shape == (98,)
    assert default_emulator.k_grid.shape == default_emulator.boost.k_grid.shape


def test_default_artifact_public_workflow_scalar_z(default_emulator):
    z = 0.0
    D = _growth_for_default(DEFAULT_PARAMS, z)

    pmm = default_emulator.get_linear_pmm(DEFAULT_PARAMS, z, D)
    pkcb = default_emulator.get_linear_pkcb(DEFAULT_PARAMS, z, D)
    boost = default_emulator.boost.get_Pk(DEFAULT_PARAMS, z, D)
    pk = default_emulator.get_Pk(DEFAULT_PARAMS, z, D)

    assert pmm.shape == (300,)
    assert pkcb.shape == (300,)
    assert boost.shape == (98,)
    assert pk.shape == (98,)
    assert jnp.all(jnp.isfinite(pk))
    assert jnp.all(pk > 0.0)

    expected = (
        jnp.interp(
            default_emulator.boost.k_grid, default_emulator.linear_pmm.k_grid, pmm
        )
        * boost
    )
    assert jnp.allclose(pk, expected, rtol=1.0e-10, atol=1.0e-10)


def test_default_artifact_public_workflow_vector_z(default_emulator):
    z = jnp.array([0.0, 0.5, 1.0])
    D = _growth_for_default(DEFAULT_PARAMS, z)

    pk = default_emulator.get_Pk(DEFAULT_PARAMS, z, D)
    k_halofit, pk_halofit = default_emulator.get_halofit_pmm(DEFAULT_PARAMS, z)

    assert pk.shape == (len(z), 98)
    assert k_halofit.shape == (300,)
    assert pk_halofit.shape == (len(z), 300)
    assert jnp.all(jnp.isfinite(pk))
    assert jnp.all(jnp.isfinite(pk_halofit))
    assert jnp.all(pk > 0.0)
    assert jnp.all(pk_halofit > 0.0)
