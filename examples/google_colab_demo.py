#!/usr/bin/env python3
"""
Google Colab Demo - Audio Engine

Optimized demonstration for Google Colab environment with GPU acceleration
and interactive widgets for generating therapeutic noise content.
"""

import sys
from pathlib import Path
import time

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_engine import NoiseGenerator


def setup_colab_environment():
    """Setup Google Colab environment with necessary installations."""
    print("🚀 Setting up Google Colab Environment")
    print("=" * 50)
    
    # Check if running in Colab
    try:
        import google.colab
        print("✅ Google Colab detected")
        IN_COLAB = True
    except ImportError:
        print("⚠️  Not running in Google Colab")
        IN_COLAB = False
    
    return IN_COLAB


def demonstrate_gpu_acceleration():
    """Demonstrate GPU acceleration capabilities."""
    print("\n🖥️  GPU Acceleration Test")
    print("-" * 30)
    
    # Test with CUDA enabled
    generator_gpu = NoiseGenerator(use_cuda=True)
    device_info = generator_gpu.cuda_accelerator.get_device_info()
    
    print(f"CUDA Available: {device_info['cuda_available']}")
    if device_info['cuda_available']:
        print(f"GPU: {device_info['device_name']}")
        print(f"GPU Memory: {device_info['total_memory_gb']:.1f} GB")
        print(f"GPU Memory Free: {device_info['memory_free_gb']:.1f} GB")
    
    # Performance comparison
    print("\nPerformance Comparison (30-second generation):")
    
    # GPU timing
    if device_info['cuda_available']:
        start_time = time.time()
        audio_gpu = generator_gpu.generate_pink_noise(duration_minutes=0.5)
        gpu_time = time.time() - start_time
        print(f"GPU Generation Time: {gpu_time:.2f}s")
    else:
        gpu_time = float('inf')
        print("GPU: Not available")
    
    # CPU timing
    generator_cpu = NoiseGenerator(use_cuda=False)
    start_time = time.time()
    audio_cpu = generator_cpu.generate_pink_noise(duration_minutes=0.5)
    cpu_time = time.time() - start_time
    print(f"CPU Generation Time: {cpu_time:.2f}s")
    
    if gpu_time != float('inf'):
        speedup = cpu_time / gpu_time
        print(f"GPU Speedup: {speedup:.1f}x faster")
    
    return generator_gpu if device_info['cuda_available'] else generator_cpu


def generate_youtube_content(generator):
    """Generate various durations of content for YouTube."""
    print("\n📺 YouTube Content Generation")
    print("-" * 35)
    
    # Common YouTube durations
    durations = [
        (30, "30 minutes - Short meditation"),
        (60, "1 hour - Standard sleep aid"),
        (120, "2 hours - Deep sleep session"),
        (360, "6 hours - Full night sleep"),
        (600, "10 hours - Extended sleep aid")
    ]
    
    print("Available YouTube durations:")
    for i, (minutes, description) in enumerate(durations, 1):
        print(f"  {i}. {description}")
    
    # For demo, generate shorter samples
    print("\nGenerating demo samples (10 seconds each)...")
    
    noise_types = ["white", "pink", "brown"]
    
    for noise_type in noise_types:
        print(f"Generating {noise_type} noise...")
        
        start_time = time.time()
        
        if noise_type == "white":
            audio = generator.generate_white_noise(duration_minutes=10/60)  # 10 seconds
        elif noise_type == "pink":
            audio = generator.generate_pink_noise(duration_minutes=10/60)
        elif noise_type == "brown":
            audio = generator.generate_brown_noise(duration_minutes=10/60)
        
        generation_time = time.time() - start_time
        
        # Export with YouTube-optimized settings
        filename = f"youtube_{noise_type}_noise_demo.flac"
        generator.export_flac(
            filename,
            audio,
            noise_type=noise_type,
            duration_minutes=10/60
        )
        
        print(f"  ✅ {filename} ({generation_time:.2f}s)")
    
    # Validate YouTube compliance
    print("\n🔍 YouTube Compliance Check")
    validation = generator.loudness_processor.validate_youtube_compliance(audio)
    
    print(f"LUFS Compliant: {'✅' if validation['lufs_compliant'] else '❌'}")
    print(f"True Peak Compliant: {'✅' if validation['true_peak_compliant'] else '❌'}")
    print(f"Measured LUFS: {validation['measured_lufs']:.1f}")
    print(f"True Peak: {validation['measured_true_peak_dbtp']:.1f} dBTP")


def interactive_generation():
    """Interactive noise generation with user input."""
    print("\n🎛️  Interactive Generation")
    print("-" * 25)
    
    try:
        # Try to import ipywidgets for Colab
        from IPython.display import display
        import ipywidgets as widgets
        
        # Create interactive widgets
        noise_type_widget = widgets.Dropdown(
            options=['white', 'pink', 'brown'],
            value='pink',
            description='Noise Type:'
        )
        
        duration_widget = widgets.IntSlider(
            value=30,
            min=10,
            max=600,
            step=10,
            description='Duration (min):'
        )
        
        therapeutic_widget = widgets.Checkbox(
            value=True,
            description='Therapeutic EQ'
        )
        
        generate_button = widgets.Button(
            description='Generate Audio',
            button_style='success'
        )
        
        output_widget = widgets.Output()
        
        def on_generate_click(b):
            with output_widget:
                output_widget.clear_output()
                print(f"Generating {noise_type_widget.value} noise...")
                print(f"Duration: {duration_widget.value} minutes")
                print(f"Therapeutic EQ: {therapeutic_widget.value}")
                
                # Generate audio
                generator = NoiseGenerator(therapeutic_eq=therapeutic_widget.value)
                
                if noise_type_widget.value == "white":
                    audio = generator.generate_white_noise(duration_widget.value)
                elif noise_type_widget.value == "pink":
                    audio = generator.generate_pink_noise(duration_widget.value)
                elif noise_type_widget.value == "brown":
                    audio = generator.generate_brown_noise(duration_widget.value)
                
                filename = f"interactive_{noise_type_widget.value}_{duration_widget.value}min.flac"
                generator.export_flac(filename, audio, noise_type_widget.value, duration_widget.value)
                
                print(f"✅ Generated: {filename}")
        
        generate_button.on_click(on_generate_click)
        
        # Display widgets
        display(widgets.VBox([
            noise_type_widget,
            duration_widget,
            therapeutic_widget,
            generate_button,
            output_widget
        ]))
        
    except ImportError:
        # Fallback for non-Colab environments
        print("Interactive widgets not available. Using command-line interface.")
        
        noise_type = input("Enter noise type (white/pink/brown): ").lower()
        if noise_type not in ['white', 'pink', 'brown']:
            noise_type = 'pink'
        
        duration = int(input("Enter duration in minutes (1-60): ") or "1")
        duration = max(1, min(60, duration))
        
        print(f"Generating {noise_type} noise for {duration} minutes...")
        
        generator = NoiseGenerator()
        
        if noise_type == "white":
            audio = generator.generate_white_noise(duration)
        elif noise_type == "pink":
            audio = generator.generate_pink_noise(duration)
        elif noise_type == "brown":
            audio = generator.generate_brown_noise(duration)
        
        filename = f"interactive_{noise_type}_{duration}min.flac"
        generator.export_flac(filename, audio, noise_type, duration)
        
        print(f"✅ Generated: {filename}")


def main():
    """Main demonstration function."""
    print("🎵 Audio Engine - Google Colab Demo")
    print("=" * 50)
    
    # Setup environment
    in_colab = setup_colab_environment()
    
    # Demonstrate GPU acceleration
    generator = demonstrate_gpu_acceleration()
    
    # Generate YouTube content
    generate_youtube_content(generator)
    
    # Interactive generation
    if in_colab:
        interactive_generation()
    else:
        print("\n💡 Tip: Run this in Google Colab for interactive widgets!")
    
    # Performance report
    print("\n📊 Performance Report")
    print("-" * 20)
    report = generator.cuda_accelerator.get_performance_report()
    
    print(f"Device: {report['device_info']['device_type']}")
    if report['device_info']['cuda_available']:
        print(f"GPU: {report['device_info']['device_name']}")
        print(f"Memory Usage: {report['memory_usage']['gpu_memory_percent']:.1f}%")
    
    print(f"Operations: {report['performance_stats']['operations_count']}")
    print(f"Avg Processing Time: {report['performance_stats']['average_processing_time']:.2f}s")
    
    print("\n🎉 Google Colab demo complete!")
    print("Check the generated FLAC files for your therapeutic audio content.")


if __name__ == "__main__":
    main()