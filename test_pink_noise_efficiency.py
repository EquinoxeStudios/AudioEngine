#!/usr/bin/env python3
"""Test pink noise generation efficiency improvements"""

import time
import numpy as np
from baby_noise_engine import BabyNoiseEngine

def test_pink_noise_algorithms():
    """Test different pink noise generation methods"""
    print("Testing Pink Noise Generation Efficiency")
    print("=" * 60)
    
    # Test short duration (Voss-McCartney)
    print("\nShort Duration (30 seconds) - Should use Voss-McCartney:")
    print("-" * 40)
    
    start_time = time.time()
    engine_short = BabyNoiseEngine(
        noise_type='pink',
        duration_str='30 secs',
        sample_rate=48000,
        bit_depth=24
    )
    pink_short = engine_short.generate_pink_noise()
    short_time = time.time() - start_time
    
    print(f"Generation time: {short_time:.2f} seconds")
    print(f"Samples: {len(pink_short):,}")
    print(f"Peak value: {np.max(np.abs(pink_short)):.3f}")
    print(f"RMS: {np.sqrt(np.mean(pink_short**2)):.3f}")
    
    # Test long duration (filtered white noise)
    print("\nLong Duration (90 minutes) - Should use filtered white noise:")
    print("-" * 40)
    
    start_time = time.time()
    engine_long = BabyNoiseEngine(
        noise_type='pink',
        duration_str='90 mins',
        sample_rate=48000,
        bit_depth=24
    )
    # Just generate a small portion to test
    engine_long.total_samples = 48000 * 60 * 5  # 5 minutes for testing
    pink_long = engine_long.generate_pink_noise()
    long_time = time.time() - start_time
    
    print(f"Generation time (5 min sample): {long_time:.2f} seconds")
    print(f"Samples: {len(pink_long):,}")
    print(f"Peak value: {np.max(np.abs(pink_long)):.3f}")
    print(f"RMS: {np.sqrt(np.mean(pink_long**2)):.3f}")
    
    # Efficiency comparison
    print("\nEfficiency Analysis:")
    print("-" * 40)
    samples_per_sec_short = len(pink_short) / short_time
    samples_per_sec_long = len(pink_long) / long_time
    
    print(f"Voss-McCartney: {samples_per_sec_short:,.0f} samples/sec")
    print(f"Filtered white: {samples_per_sec_long:,.0f} samples/sec")
    print(f"Speed improvement: {samples_per_sec_long/samples_per_sec_short:.1f}x")

def test_spectral_characteristics():
    """Verify both methods produce proper 1/f spectrum"""
    print("\n" + "=" * 60)
    print("Spectral Characteristics Test")
    print("=" * 60)
    
    # Generate samples with both methods
    engine = BabyNoiseEngine(noise_type='pink', duration_str='10 secs')
    
    # Force Voss-McCartney
    engine.total_samples = 48000 * 10
    pink_vm = engine.generate_pink_noise()
    
    # Force filtered white noise
    engine.total_samples = 48000 * 60 * 61  # Just over threshold
    pink_filtered = engine.generate_pink_noise()
    engine.total_samples = 48000 * 10  # Reset for analysis
    
    # Analyze spectrum (simplified)
    def analyze_spectrum(signal, name):
        # Take FFT of middle portion
        fft_size = 8192
        start = len(signal) // 2 - fft_size // 2
        segment = signal[start:start+fft_size, 0]
        
        fft = np.fft.rfft(segment)
        magnitude = np.abs(fft)
        power = magnitude ** 2
        
        # Check slope in log-log space (simplified)
        freqs = np.fft.rfftfreq(fft_size, 1/48000)
        # Avoid DC and very low frequencies
        valid = (freqs > 20) & (freqs < 20000)
        
        log_freq = np.log10(freqs[valid])
        log_power = np.log10(power[valid] + 1e-10)
        
        # Linear regression for slope
        slope = np.polyfit(log_freq, log_power, 1)[0]
        
        print(f"\n{name}:")
        print(f"  Spectral slope: {slope:.2f} (ideal: -1.0 for 1/f)")
        print(f"  RMS: {np.sqrt(np.mean(signal**2)):.3f}")
        
        return slope
    
    slope_vm = analyze_spectrum(pink_vm[:48000*10], "Voss-McCartney")
    slope_filtered = analyze_spectrum(pink_filtered[:48000*10], "Filtered White")
    
    print(f"\nSlope difference: {abs(slope_vm - slope_filtered):.3f}")
    if abs(slope_vm - slope_filtered) < 0.2:
        print("[OK] Both methods produce similar spectral characteristics")
    else:
        print("[WARN] Methods produce different spectral characteristics")

def test_gpu_residency():
    """Test that GPU operations stay on GPU"""
    print("\n" + "=" * 60)
    print("GPU Residency Test")
    print("=" * 60)
    
    try:
        import cupy as cp
        print("CuPy available - testing GPU residency")
        
        # Monitor memory transfers
        engine = BabyNoiseEngine(
            noise_type='pink',
            duration_str='5 mins',
            sample_rate=48000
        )
        
        # Track GPU memory usage
        mempool = cp.get_default_memory_pool()
        initial_bytes = mempool.used_bytes()
        
        # Generate pink noise
        pink = engine.generate_pink_noise()
        
        final_bytes = mempool.used_bytes()
        
        print(f"GPU memory used: {(final_bytes - initial_bytes) / 1024 / 1024:.1f} MB")
        print(f"Is result on GPU: {isinstance(pink, cp.ndarray)}")
        
        if isinstance(pink, cp.ndarray):
            print("[OK] Pink noise generation maintains GPU residency")
        else:
            print("[FAIL] Pink noise was transferred to CPU")
            
    except ImportError:
        print("CuPy not available - skipping GPU residency test")

if __name__ == "__main__":
    test_pink_noise_algorithms()
    test_spectral_characteristics()
    test_gpu_residency()