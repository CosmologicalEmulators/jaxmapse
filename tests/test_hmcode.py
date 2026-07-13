from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxmapse import HMCodeCosmology, hmcode_boost, hmcode_pmm, hmcode_pmm_jax


def _reference_case():
    data = np.loadtxt(Path(__file__).parent / "data" / "hmcode_camb_reference.txt")
    z_values = np.unique(data[:, 0])
    k_values = data[data[:, 0] == z_values[0], 1]

    # jaxmapse uses redshift-first spectra, the transpose of Mapse.jl's (k, z)
    # convention used in the source Julia tests.
    pk_mm_zk = np.stack([data[data[:, 0] == z, 2] for z in z_values])
    pk_cb_zk = np.stack([data[data[:, 0] == z, 3] for z in z_values])
    pk_nl_ref_zk = np.stack([data[data[:, 0] == z, 4] for z in z_values])
    boost_ref_zk = np.stack([data[data[:, 0] == z, 5] for z in z_values])

    h = 0.6736
    omega_b = 0.02237 / h**2
    omega_nu = 0.06 / (93.14 * h**2)
    omega_m = omega_b + 0.12 / h**2 + omega_nu
    cosmo = HMCodeCosmology(
        Omega_m=omega_m,
        Omega_b=omega_b,
        h=h,
        n_s=0.9649,
        sigma_8=0.8109118,
        w0=-1.0,
        wa=0.0,
        Omega_nu=omega_nu,
        Omega_k=0.0,
    )

    return (
        cosmo,
        jnp.asarray(z_values),
        jnp.asarray(k_values),
        jnp.asarray(pk_mm_zk),
        jnp.asarray(pk_cb_zk),
        jnp.asarray(pk_nl_ref_zk),
        jnp.asarray(boost_ref_zk),
    )


def _curved_parity_case():
    data = np.loadtxt(Path(__file__).parent / "data" / "hmcode_curved_parity_reference.txt")
    z_values = np.unique(data[:, 0])
    k_values = data[data[:, 0] == z_values[0], 1]
    pmm_zk = np.stack([data[data[:, 0] == z, 2] for z in z_values])
    pcb_zk = np.stack([data[data[:, 0] == z, 3] for z in z_values])
    dmo_zk = np.stack([data[data[:, 0] == z, 4] for z in z_values])
    feedback_zk = np.stack([data[data[:, 0] == z, 5] for z in z_values])

    h = 0.6736
    omega_b = 0.02237 / h**2
    omega_nu = 0.06 / (93.14 * h**2)
    omega_m = omega_b + 0.12 / h**2 + omega_nu
    cosmo = HMCodeCosmology(
        Omega_m=omega_m,
        Omega_b=omega_b,
        h=h,
        n_s=0.9649,
        sigma_8=0.8109118,
        w0=-0.9,
        wa=0.2,
        Omega_nu=omega_nu,
        Omega_k=0.01,
    )
    return (
        cosmo,
        jnp.asarray(z_values),
        jnp.asarray(k_values),
        jnp.asarray(pmm_zk),
        jnp.asarray(pcb_zk),
        jnp.asarray(dmo_zk),
        jnp.asarray(feedback_zk),
    )


def test_hmcode_curved_neutrino_parity_fixture_matches_native_julia():
    cosmo, z, k, pmm_zk, pcb_zk, dmo_reference, feedback_reference = _curved_parity_case()

    dmo = hmcode_pmm(cosmo, z, k, pmm_zk, pcb_zk, T_AGN=None, nM=32)
    feedback = hmcode_pmm(cosmo, z, k, pmm_zk, pcb_zk, T_AGN=10.0**7.8, nM=32)

    # Native Julia uses adaptive root finding/integration while JAX uses fixed
    # compiled grids, so this deliberately has a looser cross-backend tolerance.
    assert jnp.allclose(dmo, dmo_reference, rtol=1.5e-2, atol=0.0)
    assert jnp.allclose(feedback, feedback_reference, rtol=1.5e-2, atol=0.0)


def test_hmcode_boost_matches_camb_reference_table():
    cosmo, z, k, pk_mm_zk, pk_cb_zk, _, boost_ref_zk = _reference_case()

    boost = hmcode_boost(cosmo, z, k, pk_mm_zk, pk_cb_zk, nM=256)

    assert boost.shape == pk_mm_zk.shape
    assert jnp.all(jnp.isfinite(boost))
    assert jnp.allclose(boost, boost_ref_zk, rtol=3.0e-3, atol=0.0)


def test_hmcode_scalar_pmm_and_boost_are_consistent():
    cosmo, _, k, pk_mm_zk, pk_cb_zk, *_ = _reference_case()

    pmm = hmcode_pmm(cosmo, 0.0, k, pk_mm_zk[0], pk_cb_zk[0], nM=64)
    boost = hmcode_boost(cosmo, 0.0, k, pk_mm_zk[0], pk_cb_zk[0], nM=64)

    assert pmm.shape == k.shape
    assert boost.shape == k.shape
    assert jnp.all(jnp.isfinite(pmm))
    assert jnp.all(jnp.isfinite(boost))
    assert jnp.allclose(pmm / pk_mm_zk[0], boost, rtol=1.0e-10, atol=1.0e-10)


def test_hmcode_keyword_and_positional_cb_paths_match():
    cosmo, z, k, pk_mm_zk, pk_cb_zk, *_ = _reference_case()

    boost_keyword = hmcode_boost(cosmo, z, k, pk_mm_zk, pk_cb_z=pk_cb_zk, nM=64)
    boost_positional = hmcode_boost(cosmo, z, k, pk_mm_zk, pk_cb_zk, nM=64)

    assert jnp.allclose(boost_keyword, boost_positional, rtol=1.0e-12, atol=1.0e-12)


def test_hmcode_support_grid_matches_same_grid_call():
    cosmo, z, k, pk_mm_zk, pk_cb_zk, *_ = _reference_case()

    boost_support = hmcode_boost(
        cosmo,
        z,
        k,
        pk_mm_zk,
        k_support=k,
        pk_cb_support_z=pk_cb_zk,
        nM=64,
    )
    boost_same_grid = hmcode_boost(cosmo, z, k, pk_mm_zk, pk_cb_zk, nM=64)

    assert jnp.allclose(boost_support, boost_same_grid, rtol=1.0e-12, atol=1.0e-12)


def test_hmcode_support_output_grid_and_scalar_support_call():
    cosmo, z, k, pk_mm_zk, pk_cb_zk, *_ = _reference_case()
    k_out = k[::2]

    pmm_kout = hmcode_pmm(
        cosmo,
        z,
        k_out,
        pk_mm_zk,
        k_support=k,
        pk_cb_support_z=pk_cb_zk,
        nM=64,
    )
    boost_kout = hmcode_boost(
        cosmo,
        z,
        k_out,
        pk_mm_zk,
        k_support=k,
        pk_cb_support_z=pk_cb_zk,
        nM=64,
    )

    assert pmm_kout.shape == (len(z), len(k_out))
    assert boost_kout.shape == (len(z), len(k_out))
    assert jnp.all(jnp.isfinite(pmm_kout))
    assert jnp.allclose(boost_kout, pmm_kout / pk_mm_zk[:, ::2], rtol=1.0e-10, atol=1.0e-10)

    scalar_pmm = hmcode_pmm(
        cosmo,
        0.0,
        k_out,
        pk_mm_zk[0],
        k_support=k,
        pk_cb_support_z=pk_cb_zk[0],
        nM=64,
    )
    assert jnp.allclose(scalar_pmm, pmm_kout[0], rtol=1.0e-10, atol=1.0e-10)


@pytest.mark.parametrize("nM", [32])
def test_hmcode_pure_jax_kernel_is_jittable(nM):
    cosmo, z_all, k_support, pk_mm_all, pk_cb_all, *_ = _reference_case()
    z = z_all[:1]
    k_out = k_support[::8]
    pk_mm = pk_mm_all[:1]
    pk_cb = pk_cb_all[:1]

    compiled = jax.jit(
        hmcode_pmm_jax,
        static_argnames=("nM", "include_feedback"),
    )
    pmm_jit = compiled(
        cosmo,
        z,
        k_out,
        k_support,
        pk_mm,
        pk_cb,
        nM=nM,
        include_feedback=True,
    )
    pmm_host = hmcode_pmm(
        cosmo,
        z,
        k_out,
        pk_mm,
        k_support=k_support,
        pk_cb_support_z=pk_cb,
        nM=nM,
    )

    assert pmm_jit.shape == pmm_host.shape
    assert jnp.all(jnp.isfinite(pmm_jit))
    assert jnp.allclose(pmm_jit, pmm_host, rtol=3.0e-3, atol=0.0)


def test_hmcode_rejects_invalid_inputs_like_mapse_jl():
    cosmo, z, k, pk_mm_zk, pk_cb_zk, *_ = _reference_case()
    k_out = k[::2]

    with pytest.raises(ValueError, match="sorted in ascending order"):
        hmcode_pmm(cosmo, z, k[::-1], pk_mm_zk)

    irregular_k = k.at[10].set(0.5 * (k[9] + k[10]))
    with pytest.raises(ValueError, match="uniformly spaced in log"):
        hmcode_pmm(cosmo, z, irregular_k, pk_mm_zk)

    with pytest.raises(ValueError, match="pk_mm_z must have shape"):
        hmcode_pmm(cosmo, jnp.array([0.0, 1.0]), k, pk_mm_zk[:1])

    with pytest.raises(ValueError, match="pk_cb_z must have shape"):
        hmcode_pmm(cosmo, z, k, pk_mm_zk, pk_cb_zk[:, :-1])

    with pytest.raises(ValueError, match="at least 2"):
        hmcode_boost(cosmo, z, k, pk_mm_zk, pk_cb_zk, nM=0)

    with pytest.raises(ValueError, match="at least 2"):
        hmcode_boost(cosmo, z, k, pk_mm_zk, pk_cb_zk, nM=1)

    with pytest.raises(ValueError, match="output k range"):
        hmcode_boost(
            cosmo,
            z,
            jnp.concatenate([k_out, jnp.array([20.0])]),
            pk_mm_zk,
            k_support=k,
            pk_cb_support_z=pk_cb_zk,
        )
