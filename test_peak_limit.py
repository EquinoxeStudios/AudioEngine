#!/usr/bin/env python3
"""Test true peak limiting with different settings"""

from baby_noise_engine import BabyNoiseEngine
import numpy as np
import soundfile as sf

def measure_true_peak(audio_file):
    """Measure the true peak of an audio file"""
    audio, sr = sf.read(audio_file)
    
    # Simple peak measurement (not oversampled)
    peak_linear = np.max(np.abs(audio))
    peak_db = 20 * np.log10(peak_linear)
    
    return peak_db

def test_peak_limits():
    """Test different true peak limit settings"""
    print("Testing True Peak Limiting\n")
    print("=" * 50)
    
    test_limits = [-1.0, -2.0, -3.0]
    
    for limit in test_limits:
        print(f"\nTesting with true peak limit: {limit} dBTP")
        
        # Create engine with specific limit
        engine = BabyNoiseEngine(
            noise_type='white',
            duration_str='3 secs',
            true_peak_limit=limit
        )
        
        # Generate audio
        audio = engine.generate()
        
        # Save to file
        filename = f"test_peak_{abs(limit):.0f}dB.flac"
        engine.save_to_flac(audio, filename)
        
        # Measure actual peak
        actual_peak = measure_true_peak(filename)
        
        print(f"  Target limit: {limit:.1f} dBTP")
        print(f"  Actual peak: {actual_peak:.2f} dB")
        print(f"  Headroom: {actual_peak:.2f} dB")
        
        # Note: actual peak should be at or below the limit
        if actual_peak <= limit + 0.1:  # Allow 0.1 dB tolerance
            print("  [OK] Peak limiting working correctly")
        else:
            print("  [FAIL] Peak exceeded limit!")

def explain_streaming_headroom():
    """Explain why -2 dBTP is recommended"""
    print("\n" + "=" * 50)
    print("Why -2 dBTP for Streaming?")
    print("=" * 50)
    print("""
1. Lossy Codec Overshoot:
   - AAC encoding can add 0.5-1.5 dB peaks
   - Opus encoding can add 0.5-1.0 dB peaks
   - MP3 encoding can add up to 2 dB peaks

2. Platform Requirements:
   - YouTube: -2 dBTP max after normalization
   - Spotify: -2 dBTP recommended
   - Apple Music: -1 dBTP (but -2 safer)

3. Benefits of -2 dBTP:
   - Prevents clipping after transcoding
   - Maintains audio quality
   - Ensures compatibility across platforms
   - Future-proof for new codecs

4. For Baby Noise:
   - Consistent levels important for sleep
   - No sudden peaks to wake baby
   - Clean playback on all devices
""")

if __name__ == "__main__":
    test_peak_limits()
    explain_streaming_headroom()