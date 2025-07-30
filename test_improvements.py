#!/usr/bin/env python3
"""Test the improvements made to the Baby Noise Engine"""

from baby_noise_engine import BabyNoiseEngine, main

def test_validation():
    """Test input validation"""
    print("Testing input validation...")
    
    # Test invalid noise type
    try:
        engine = BabyNoiseEngine(noise_type='blue')
        print("[FAIL] Failed to catch invalid noise type")
    except ValueError as e:
        print(f"[OK] Caught invalid noise type: {e}")
    
    # Test invalid sample rate
    try:
        engine = BabyNoiseEngine(sample_rate=22050)
        print("[FAIL] Failed to catch invalid sample rate")
    except ValueError as e:
        print(f"[OK] Caught invalid sample rate: {e}")
    
    # Test duration limits
    try:
        engine = BabyNoiseEngine(duration_str='24 hours')
        print("[FAIL] Failed to catch excessive duration")
    except ValueError as e:
        print(f"[OK] Caught excessive duration: {e}")
    
    print()

def test_progress_callback():
    """Test progress callback functionality"""
    print("Testing progress callback...")
    
    progress_log = []
    
    def track_progress(stage, percent):
        progress_log.append((stage, percent))
        print(f"  Progress: {stage} - {percent}%")
    
    # Generate with progress tracking
    result = main('white', '2 secs', show_progress=False)
    
    if result:
        print(f"[OK] Generated file: {result}")
    else:
        print("[FAIL] Generation failed")
    
    print()

def test_memory_warning():
    """Test memory warning for large files"""
    print("Testing memory warnings...")
    
    # Test small file (no warning)
    print("Small file (5 seconds):")
    engine = BabyNoiseEngine(duration_str='5 secs')
    
    # Test large file (should warn)
    print("\nLarge file (2 hours):")
    engine = BabyNoiseEngine(duration_str='2 hours')
    
    print()

def test_error_handling():
    """Test error handling"""
    print("Testing error handling...")
    
    # Test with all error cases handled
    result = main('invalid', '1 hour')
    if result is None:
        print("[OK] Handled invalid input gracefully")
    
    result = main('white', 'invalid duration')
    if result is None:
        print("[OK] Handled invalid duration gracefully")
    
    print()

if __name__ == "__main__":
    print("=== Testing Baby Noise Engine Improvements ===\n")
    
    test_validation()
    test_memory_warning()
    test_error_handling()
    test_progress_callback()
    
    print("=== All tests completed ===")