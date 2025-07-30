#!/usr/bin/env python3
"""Test noise generation consistency and levels"""

from baby_noise_engine import BabyNoiseEngine
import numpy as np

def analyze_noise_characteristics():
    """Analyze the characteristics of each noise type"""
    print("Analyzing Noise Generation Consistency")
    print("=" * 60)
    
    noise_types = ['white', 'pink', 'brown']
    duration = '5 secs'
    
    for noise_type in noise_types:
        print(f"\n{noise_type.upper()} NOISE:")
        print("-" * 30)
        
        # Create engine
        engine = BabyNoiseEngine(
            noise_type=noise_type,
            duration_str=duration,
            sample_rate=48000,
            bit_depth=24
        )
        
        # Generate noise (before processing)
        if noise_type == 'white':
            raw_noise = engine.generate_white_noise()
        elif noise_type == 'pink':
            raw_noise = engine.generate_pink_noise()
        else:
            raw_noise = engine.generate_brown_noise()
        
        # Calculate statistics
        mean = np.mean(raw_noise)
        std = np.std(raw_noise)
        rms = np.sqrt(np.mean(raw_noise**2))
        peak = np.max(np.abs(raw_noise))
        
        # Convert to dB
        rms_db = 20 * np.log10(rms) if rms > 0 else -np.inf
        peak_db = 20 * np.log10(peak) if peak > 0 else -np.inf
        
        # Theoretical values
        if noise_type == 'white':
            theoretical_rms = 0.1  # Our scaling factor
            crest_factor = np.sqrt(3)  # ~4.77 dB for Gaussian
        elif noise_type == 'pink':
            theoretical_rms = 0.1  # After scaling
            crest_factor = np.sqrt(3.5)  # Slightly higher than white
        else:  # brown
            theoretical_rms = 0.05  # After normalization in brown noise
            crest_factor = np.sqrt(4)  # Higher due to integration
        
        print(f"  Mean: {mean:.6f} (should be ~0)")
        print(f"  Std Dev: {std:.6f}")
        print(f"  RMS: {rms:.6f} ({rms_db:.1f} dB)")
        print(f"  Peak: {peak:.6f} ({peak_db:.1f} dB)")
        print(f"  Crest Factor: {peak/rms:.2f} ({peak_db - rms_db:.1f} dB)")
        print(f"  Theoretical RMS: ~{theoretical_rms:.3f}")
        
        # Check if levels are reasonable
        if abs(rms - theoretical_rms) / theoretical_rms < 0.5:
            print("  [OK] RMS level is consistent")
        else:
            print("  [WARN] RMS level differs from expected")

def explain_unity_variance_approach():
    """Explain the benefits of starting with unity variance"""
    print("\n" + "=" * 60)
    print("Unity Variance Approach Benefits")
    print("=" * 60)
    print("""
1. Mathematical Consistency:
   - Standard normal: mean=0, std=1 is the canonical form
   - All noise types start from the same basis
   - Predictable behavior in signal processing

2. LUFS Normalization:
   - Starting level doesn't matter - LUFS will normalize
   - But consistent starting point = predictable gain
   - Unity variance ~ 0 dBFS RMS (very loud)
   - Scale by 0.1 ~ -20 dB ~ -20 LUFS (reasonable)

3. Headroom Management:
   - Unity variance would clip immediately
   - 0.1 scaling provides 20 dB headroom
   - Room for EQ boosts and compression
   - Prevents saturation in processing chain

4. Cross-Platform Consistency:
   - NumPy and CuPy both use standard_normal
   - No need to specify different parameters
   - Identical results on CPU and GPU
""")

def test_processing_headroom():
    """Test that we have adequate headroom for processing"""
    print("\n" + "=" * 60)
    print("Processing Headroom Test")
    print("=" * 60)
    
    # Generate white noise with unity variance approach
    engine = BabyNoiseEngine(noise_type='white', duration_str='2 secs')
    
    # Get noise at each stage
    raw_noise = engine.generate_white_noise()
    
    # Full processing
    processed = engine.generate()
    
    print(f"Raw noise peak: {np.max(np.abs(raw_noise)):.3f}")
    print(f"Processed peak: {np.max(np.abs(processed)):.3f}")
    print(f"Headroom used: {20 * np.log10(np.max(np.abs(processed)) / np.max(np.abs(raw_noise))):.1f} dB")

if __name__ == "__main__":
    analyze_noise_characteristics()
    explain_unity_variance_approach()
    test_processing_headroom()