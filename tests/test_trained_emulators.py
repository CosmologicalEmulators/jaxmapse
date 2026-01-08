import pytest
import jax
import jax.numpy as jnp
import jaxmapse
from jaxmapse import w0waCDMCosmology, D_z

EMULATOR_NAME = "trained_mapse_class_hmcode_mnuw0waOkcdm"

@pytest.fixture
def emulator_setup():
    # Strict check: Emulator MUST be present
    if EMULATOR_NAME not in jaxmapse.trained_emulators:
        pytest.fail(f"Emulator '{EMULATOR_NAME}' is not in the trained_emulators dictionary.")
    
    emu = jaxmapse.trained_emulators[EMULATOR_NAME]
    
    if emu is None:
        pytest.fail(f"Emulator '{EMULATOR_NAME}' failed to load (value is None).")
    
    # Standard Planck 2018 + log10T_heat parameters
    # Order: [ln10As, ns, H0, ombh2, omch2, Mν, w0, wa, Omega_k, log10T_heat]
    p_hmcode = jnp.array([3.044, 0.9665, 67.66, 0.02242, 0.11933, 0.06, -1.0, 0.0, 0.0, 7.8])
    
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
        omega_k=p_hmcode[8] * h**2
    )
    D = cosmo.D_z(z)
    
    return emu, p_hmcode, z, D

@pytest.mark.parametrize("component_method", [
    "get_linear_pmm",
    "get_linear_pkcb",
    "get_Pk" # Composite
])
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

@pytest.mark.parametrize("component_method", [
    "get_linear_pmm",
    "get_linear_pkcb",
    "get_Pk" # Composite
])
def test_emulator_components_multiple_z(emulator_setup, component_method):
    """Test running each emulator component with multiple redshifts."""
    emu, p_hmcode, _, _ = emulator_setup
    
    z_vec = jnp.array([0.0, 0.5, 1.0])
    
    h = p_hmcode[2] / 100.0
    cosmo = w0waCDMCosmology(
        ln10As=p_hmcode[0], ns=p_hmcode[1], h=h,
        omega_b=p_hmcode[3], omega_c=p_hmcode[4], m_nu=p_hmcode[5],
        w0=p_hmcode[6], wa=p_hmcode[7], omega_k=p_hmcode[8] * h**2
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
        ln10As=p_hmcode[0], ns=p_hmcode[1], h=h,
        omega_b=p_hmcode[3], omega_c=p_hmcode[4], m_nu=p_hmcode[5],
        w0=p_hmcode[6], wa=p_hmcode[7], omega_k=p_hmcode[8] * h**2
    )
    D_vec = cosmo.D_z(z_vec)
    
    pk_vec = emu.boost.get_Pk(p_hmcode, z_vec, D_vec)
    
    n_k = emu.boost.k_grid.shape[0]
    assert pk_vec.shape == (len(z_vec), n_k)
    assert jnp.all(pk_vec > 0)

@pytest.mark.parametrize("component_method", [
    "get_linear_pmm",
    "get_linear_pkcb",
    "get_Pk"
])
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

@pytest.mark.parametrize("component_method", [
    "get_linear_pmm",
    "get_linear_pkcb",
    "get_Pk"
])
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
