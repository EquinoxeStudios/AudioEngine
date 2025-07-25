#!/usr/bin/env python3
"""
Basic Usage Example - Audio Engine

Demonstrates basic therapeutic noise generation for YouTube content.
Perfect for getting started with the Audio Engine.
"""

import sys
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_engine import NoiseGenerator


def main():
    """Basic usage demonstration."""
    print("🎵 Audio Engine - Basic Usage Example")
    print("=" * 50)
    
    # Initialize the noise generator
    print("Initializing Audio Engine...")
    generator = NoiseGenerator(
        sample_rate=48000,      # YouTube native
        bit_depth=24,           # Professional quality
        target_lufs=-14.0,      # YouTube reference
        use_cuda=True,          # GPU acceleration
        therapeutic_eq=True,    # Infant optimization
        fade_duration=5.0       # 5-second fades
    )
    
    # Check system capabilities
    print(f"CUDA Available: {generator.cuda_accelerator.is_available()}")
    print(f"Device: {generator.cuda_accelerator.device}")
    print()
    
    # Generate different types of therapeutic noise
    noise_types = [
        ("white", "White Noise - Equal energy across all frequencies"),
        ("pink", "Pink Noise - Perceptually uniform across octaves"),
        ("brown", "Brown Noise - Deeper, warmer sound")
    ]
    
    for noise_type, description in noise_types:
        print(f"Generating {description}...")
        
        # Generate 30-second sample
        if noise_type == "white":
            audio = generator.generate_white_noise(duration_minutes=0.5)  # 30 seconds
        elif noise_type == "pink":
            audio = generator.generate_pink_noise(duration_minutes=0.5)
        elif noise_type == "brown":
            audio = generator.generate_brown_noise(duration_minutes=0.5)
        
        # Export to FLAC
        output_file = f"therapeutic_{noise_type}_noise_30s.flac"
        generator.export_flac(
            output_file, 
            audio, 
            noise_type=noise_type,
            duration_minutes=0.5
        )
        
        print(f"✅ Generated: {output_file}")
        print()
    
    # Display generation statistics
    stats = generator.get_generation_stats()
    print("Generation Statistics:")
    print(f"  Total Generated: {stats['total_generated']} files")
    print(f"  Last Generation Time: {stats.get('last_generation_time', 0):.2f}s")
    print(f"  CUDA Enabled: {stats['cuda_device']}")
    print(f"  Therapeutic Processing: {stats['therapeutic_processing']}")
    
    print("\n🎉 Basic usage complete! Check the generated FLAC files.")


if __name__ == "__main__":
    main()