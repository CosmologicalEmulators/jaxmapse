import jax.numpy as jnp
from jax import jit
from typing import Union, Any

@jit
def primordial_Pk(
    As: Union[float, jnp.ndarray], 
    ns: Union[float, jnp.ndarray], 
    k: Union[float, jnp.ndarray]
) -> Union[float, jnp.ndarray]:
    """
    Compute the primordial power spectrum.
    
    Matches Mapse.jl implementation:
    return @. As * (k/0.05)^(ns-1)/k^3 * (k^2*c0^2)^2
    where c0 = 2.99792458E8 (m/s)
    
    Args:
        As: Scalar amplitude
        ns: Scalar spectral index
        k: Wavenumber (or array of wavenumbers)
        
    Returns:
        Primordial power spectrum
    """
    c0 = 2.99792458e8
    k_pivot = 0.05
    
    # Implementation matching Mapse.jl exactly
    # As * (k/0.05)^(ns-1) / k^3 * (k^2 * c0^2)^2
    return As * (k / k_pivot)**(ns - 1) / k**3 * (k**2 * c0**2)**2
