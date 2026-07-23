import os
import time
import numpy as np
import jax
import jax.numpy as jnp

from jaxmapse import (
    load_trained_emulators,
    trained_emulators,
    hmcode_pmm_dmo_smart,
    hmcode_pmm_baryonic_smart,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

def main():
    print("=" * 60)
    print("JAXMapse Smart Path Benchmark vs CAMB")
    print("=" * 60)
    
    # Load CAMB reference data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ref_dmo = np.loadtxt(os.path.join(base_dir, "camb_pk_hmcode_dmo.txt"))
    ref_fb = np.loadtxt(os.path.join(base_dir, "camb_pk_hmcode_fb.txt"))

    print(f"Loaded CAMB DMO ref shape: {ref_dmo.shape}")
    print(f"Loaded CAMB FB ref shape:  {ref_fb.shape}")

    # Load emulators
    from jaxmapse import DEFAULT_EMULATOR_ARTIFACT
    load_trained_emulators()
    bundle = trained_emulators[DEFAULT_EMULATOR_ARTIFACT]
    linear_pmm_emu = bundle.linear_pmm
    linear_pcb_emu = bundle.linear_pcb

    # Cosmology parameters matching CAMB test
    H = 0.6736
    omega_b = 0.02237
    omega_c = 0.12
    m_nu = 0.06
    n_s = 0.9649
    
    # input_params array for the ACE emulators (typically 8 parameters)
    # [ln10As, n_s, H0, omega_b, omega_c, m_nu, w0, wa]
    input_params = jnp.array([3.044, n_s, H * 100, omega_b, omega_c, m_nu, -1.0, 0.0])

    k_label = 10.0 ** jnp.linspace(-3.0, 1.0, 128)
    k_out = k_label * H
    z_fine = jnp.linspace(0.0, 3.5, 150)
    N_coarse_list = [24, 32, 40]
    nM = 128
    T_AGN = 10.0**7.8

    print("\nStarting JIT compilations & Warmup...")
    
    def time_pipeline(func, name, n_coarse, *args, **kwargs):
        # We do not jit the wrapper because it contains Python control flow.
        # The inner kernels are already compiled.
        def eager_func():
            return func(*args, N_coarse=n_coarse, **kwargs)
            
        # Warmup
        t0 = time.time()
        res = jax.block_until_ready(eager_func())
        t1 = time.time()
        print(f"  Warmup {name}[{n_coarse}]: {(t1-t0)*1000:.1f} ms")
        
        # Benchmark
        reps = 10
        times = []
        for _ in range(reps):
            t0 = time.time()
            _ = jax.block_until_ready(eager_func())
            t1 = time.time()
            times.append(t1 - t0)
        
        times_ms = np.array(times) * 1000.0
        return res, np.median(times_ms), np.min(times_ms)

    for n in N_coarse_list:
        print(f"\n--- N_coarse = {n} ---")
        
        # DMO
        res_dmo, dmo_med, dmo_min = time_pipeline(
            hmcode_pmm_dmo_smart,
            "Smart DMO",
            n,
            input_params=input_params,
            z_fine=z_fine,
            linear_pmm_emu=linear_pmm_emu,
            linear_pcb_emu=linear_pcb_emu,
            nM=nM,
            k_out=k_out
        )
        # Calculate error
        smart_dmo_np = np.array(res_dmo[1])
        dmo_err = np.abs(smart_dmo_np - ref_dmo) / np.maximum(np.abs(ref_dmo), 1e-30)
        dmo_max_err = np.max(dmo_err) * 100.0
        
        print(f"  Smart DMO[{n}] median: {dmo_med:.3f} ms | max error vs CAMB: {dmo_max_err:.3f}%")
        
        # Feedback
        res_fb, fb_med, fb_min = time_pipeline(
            hmcode_pmm_baryonic_smart,
            "Smart Feedback",
            n,
            input_params=input_params,
            z_fine=z_fine,
            linear_pmm_emu=linear_pmm_emu,
            linear_pcb_emu=linear_pcb_emu,
            T_AGN=T_AGN,
            nM=nM,
            k_out=k_out
        )
        smart_fb_np = np.array(res_fb[1])
        fb_err = np.abs(smart_fb_np - ref_fb) / np.maximum(np.abs(ref_fb), 1e-30)
        fb_max_err = np.max(fb_err) * 100.0
        
        print(f"  Smart Feedback[{n}] median: {fb_med:.3f} ms | max error vs CAMB: {fb_max_err:.3f}%")

if __name__ == "__main__":
    main()
