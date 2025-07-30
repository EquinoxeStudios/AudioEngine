#!/usr/bin/env python3
"""Test bit depth handling and dithering"""

from baby_noise_engine import BabyNoiseEngine
import numpy as np

def test_bit_depths():
    """Test different bit depths"""
    print("Testing bit depth handling...\n")
    
    for bit_depth in [16, 24, 32]:
        print(f"=== Testing {bit_depth}-bit ===")
        
        try:
            # Create engine with specific bit depth
            engine = BabyNoiseEngine(
                noise_type='white',
                duration_str='2 secs',
                sample_rate=48000,
                bit_depth=bit_depth
            )
            
            # Generate audio
            audio = engine.generate()
            
            # Save with proper bit depth
            filename = f"test_{bit_depth}bit.flac"
            engine.save_to_flac(audio, filename)
            
            print(f"Successfully generated {bit_depth}-bit file\n")
            
        except Exception as e:
            print(f"Error with {bit_depth}-bit: {e}\n")

def verify_lsb_calculation():
    """Verify LSB calculation for different bit depths"""
    print("\n=== LSB Calculation Verification ===")
    print("Audio range: -1.0 to +1.0 (2.0 total)")
    
    for bit_depth in [16, 24, 32]:
        # Correct LSB calculation
        lsb = 1.0 / (2 ** (bit_depth - 1))
        
        # Number of quantization levels
        levels = 2 ** bit_depth
        
        # Step size verification
        step_size = 2.0 / levels  # Should equal LSB
        
        print(f"\n{bit_depth}-bit:")
        print(f"  LSB = 1 / 2^{bit_depth-1} = 1 / {2**(bit_depth-1)} = {lsb:.2e}")
        print(f"  Quantization levels: {levels}")
        print(f"  Step size: 2.0 / {levels} = {step_size:.2e}")
        print(f"  Verification: LSB == step_size? {np.isclose(lsb, step_size)}")

if __name__ == "__main__":
    test_bit_depths()
    verify_lsb_calculation()