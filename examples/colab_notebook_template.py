#!/usr/bin/env python3
"""
Google Colab Notebook Template - Audio Engine

Template for creating Jupyter notebooks in Google Colab for therapeutic
noise generation. Copy this code into Colab cells for interactive use.
"""

# CELL 1: Installation and Setup
"""
# Audio Engine Installation and Setup
Run this cell first to install dependencies and setup the environment.
"""

# Install required packages
# !pip install torch torchaudio soundfile librosa mutagen psutil

# Clone the Audio Engine repository (replace with your actual repo)
# !git clone https://github.com/your-repo/AudioEngine.git

# Change to the AudioEngine directory
import os
# os.chdir('/content/AudioEngine')

# Import the Audio Engine
import sys
sys.path.append('/content/AudioEngine')

from audio_engine import NoiseGenerator
import torch
import time

print("🎵 Audio Engine - Google Colab Setup Complete!")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# CELL 2: Interactive Configuration
"""
# Interactive Configuration
Configure your noise generation settings using interactive widgets.
"""

try:
    from IPython.display import display
    import ipywidgets as widgets
    
    # Configuration widgets
    noise_type_widget = widgets.Dropdown(
        options=[
            ('White Noise - Equal energy across frequencies', 'white'),
            ('Pink Noise - Perceptually uniform (recommended)', 'pink'),
            ('Brown Noise - Deeper, warmer sound', 'brown')
        ],
        value='pink',
        description='Noise Type:',
        style={'description_width': 'initial'}
    )
    
    duration_widget = widgets.IntSlider(
        value=60,
        min=1,
        max=600,
        step=1,
        description='Duration (minutes):',
        style={'description_width': 'initial'}
    )
    
    therapeutic_widget = widgets.Checkbox(
        value=True,
        description='Therapeutic EQ (infant-optimized)',
        style={'description_width': 'initial'}
    )
    
    cuda_widget = widgets.Checkbox(
        value=torch.cuda.is_available(),
        description='GPU Acceleration',
        disabled=not torch.cuda.is_available(),
        style={'description_width': 'initial'}
    )
    
    # Display configuration
    config_box = widgets.VBox([
        widgets.HTML("<h3>🎛️ Audio Generation Configuration</h3>"),
        noise_type_widget,
        duration_widget,
        therapeutic_widget,
        cuda_widget
    ])
    
    display(config_box)
    
except ImportError:
    print("Interactive widgets not available. Using default settings.")
    noise_type_widget = type('obj', (object,), {'value': 'pink'})
    duration_widget = type('obj', (object,), {'value': 60})
    therapeutic_widget = type('obj', (object,), {'value': True})
    cuda_widget = type('obj', (object,), {'value': torch.cuda.is_available()})


# CELL 3: Audio Generation
"""
# Audio Generation
Generate therapeutic noise based on your configuration.
"""

# Initialize generator with settings
# OPTIMIZED FOR LONG DURATIONS: oversampling_factor=1
generator = NoiseGenerator(
    sample_rate=48000,
    bit_depth=24,
    target_lufs=-14.0,
    use_cuda=cuda_widget.value,
    therapeutic_eq=therapeutic_widget.value,
    fade_duration=5.0,
    oversampling_factor=1  # Disabled for faster generation
)

print("🎵 Generating Therapeutic Audio...")
print(f"Type: {noise_type_widget.value.title()} Noise")
print(f"Duration: {duration_widget.value} minutes")
print(f"Therapeutic EQ: {'Enabled' if therapeutic_widget.value else 'Disabled'}")
print(f"GPU Acceleration: {'Enabled' if cuda_widget.value else 'Disabled'}")

# Generate audio
start_time = time.time()

if noise_type_widget.value == 'white':
    audio = generator.generate_white_noise(duration_widget.value)
elif noise_type_widget.value == 'pink':
    audio = generator.generate_pink_noise(duration_widget.value)
elif noise_type_widget.value == 'brown':
    audio = generator.generate_brown_noise(duration_widget.value)

generation_time = time.time() - start_time

print(f"✅ Generation completed in {generation_time:.2f} seconds")

# Export to FLAC
filename = f"therapeutic_{noise_type_widget.value}_noise_{duration_widget.value}min.flac"
generator.export_flac(
    filename,
    audio,
    noise_type=noise_type_widget.value,
    duration_minutes=duration_widget.value
)

print(f"💾 Audio exported as: {filename}")


# CELL 4: Quality Analysis
"""
# Quality Analysis and Validation
Analyze the generated audio for YouTube compliance and therapeutic quality.
"""

# Loudness analysis
print("📊 Audio Quality Analysis")
print("-" * 30)

# LUFS measurement
measured_lufs = generator.loudness_processor.measure_lufs(audio)
true_peak = generator.loudness_processor.measure_true_peak(audio)

print(f"Measured LUFS: {measured_lufs:.2f}")
print(f"Target LUFS: -14.0 (YouTube standard)")
print(f"True Peak: {true_peak:.2f} dBTP")
print(f"True Peak Limit: -1.0 dBTP (YouTube safe)")

# YouTube compliance check
compliance = generator.loudness_processor.validate_youtube_compliance(audio)

print("\n🎯 YouTube Compliance Check:")
print(f"LUFS Compliant: {'✅ PASS' if compliance['lufs_compliant'] else '❌ FAIL'}")
print(f"True Peak Compliant: {'✅ PASS' if compliance['true_peak_compliant'] else '❌ FAIL'}")
print(f"Overall Status: {'✅ READY FOR YOUTUBE' if compliance['overall_compliant'] else '❌ NEEDS ADJUSTMENT'}")

# Therapeutic validation
if therapeutic_widget.value:
    print("\n🏥 Therapeutic Quality:")
    processor_info = generator.therapeutic_processor.get_processor_info()
    print(f"Infant-optimized EQ: ✅ Applied")
    print(f"Harshness reduction: {processor_info['harsh_reduction']} dB at {processor_info['harsh_freq_center']} Hz")
    print(f"Low-end warmth: +{processor_info['low_shelf_gain']} dB below {processor_info['low_shelf_freq']} Hz")

# Audio characteristics
print(f"\n📈 Audio Characteristics:")
print(f"Sample Rate: {generator.sample_rate} Hz (YouTube native)")
print(f"Bit Depth: {generator.bit_depth}-bit (Professional)")
print(f"Channels: Stereo")
print(f"Duration: {duration_widget.value} minutes")
print(f"File Size: ~{(audio.numel() * 4 / 1024 / 1024):.1f} MB")


# CELL 5: Download and Metadata
"""
# Download Files and View Metadata
Download the generated audio and view detailed metadata.
"""

# Generate metadata report
from audio_engine.utils import MetadataHandler

metadata_handler = MetadataHandler()
metadata_report = metadata_handler.generate_metadata_report(filename)

print("📄 Metadata Report Generated")
print("-" * 25)

# Display key metadata
summary = metadata_report.get('metadata_summary', {})
print("📋 Content Summary:")
for key, value in summary.items():
    if value != "Unknown":
        print(f"  {key.replace('_', ' ').title()}: {value}")

# Display technical specs
tech_specs = metadata_report.get('technical_specs', {})
if tech_specs:
    print("\n⚙️ Technical Specifications:")
    for key, value in tech_specs.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")

# Export metadata report
report_filename = f"metadata_report_{noise_type_widget.value}_{duration_widget.value}min.json"
metadata_handler.export_metadata_json(filename, report_filename)

print(f"\n💾 Files ready for download:")
print(f"  📁 {filename} - Audio file")
print(f"  📄 {report_filename} - Metadata report")

# Download files (in Colab)
try:
    from google.colab import files
    
    print("\n⬇️ Starting downloads...")
    files.download(filename)
    files.download(report_filename)
    print("✅ Downloads completed!")
    
except ImportError:
    print("\n💡 In Google Colab, run files.download(filename) to download")


# CELL 6: Batch Generation (Optional)
"""
# Batch Generation for Multiple Durations
Generate multiple durations for different use cases.
"""

batch_durations = [30, 60, 120, 360, 600]  # 30min, 1hr, 2hr, 6hr, 10hr
batch_files = []

print("🔄 Batch Generation for YouTube Content")
print("-" * 40)

for duration in batch_durations:
    print(f"Generating {duration}-minute {noise_type_widget.value} noise...")
    
    if noise_type_widget.value == 'white':
        batch_audio = generator.generate_white_noise(duration)
    elif noise_type_widget.value == 'pink':
        batch_audio = generator.generate_pink_noise(duration)
    elif noise_type_widget.value == 'brown':
        batch_audio = generator.generate_brown_noise(duration)
    
    batch_filename = f"youtube_{noise_type_widget.value}_noise_{duration}min.flac"
    generator.export_flac(
        batch_filename,
        batch_audio,
        noise_type=noise_type_widget.value,
        duration_minutes=duration
    )
    
    batch_files.append(batch_filename)
    print(f"  ✅ {batch_filename}")

print(f"\n🎉 Batch generation complete! Generated {len(batch_files)} files.")

# Download all batch files
try:
    from google.colab import files
    
    download_batch = input("Download all batch files? (y/n): ").lower() == 'y'
    if download_batch:
        for batch_file in batch_files:
            files.download(batch_file)
        print("✅ All batch files downloaded!")
        
except ImportError:
    print("💡 Run files.download() for each file to download in Colab")


# CELL 7: Performance Monitoring
"""
# Performance Monitoring and System Info
Monitor generation performance and system resources.
"""

# Get performance statistics
stats = generator.get_generation_stats()
performance_report = generator.cuda_accelerator.get_performance_report()

print("⚡ Performance Statistics")
print("-" * 25)

print(f"Total Files Generated: {stats['total_generated']}")
print(f"Last Generation Time: {stats.get('last_generation_time', 0):.2f} seconds")
print(f"Device Used: {stats['cuda_device']}")

# System info
device_info = performance_report['device_info']
print(f"\n💻 System Information:")
print(f"  Device Type: {device_info['device_type']}")
print(f"  CUDA Available: {device_info['cuda_available']}")

if device_info['cuda_available']:
    print(f"  GPU: {device_info['device_name']}")
    print(f"  GPU Memory: {device_info['total_memory_gb']:.1f} GB")
    print(f"  Memory Used: {device_info['memory_allocated_gb']:.1f} GB")
    print(f"  Memory Free: {device_info['memory_free_gb']:.1f} GB")

# Memory usage
memory_usage = performance_report['memory_usage']
print(f"\n💾 Memory Usage:")
print(f"  System Memory: {memory_usage['system_memory_percent']:.1f}%")
if device_info['cuda_available']:
    print(f"  GPU Memory: {memory_usage.get('gpu_memory_percent', 0):.1f}%")

print("\n🎉 Audio Engine session complete!")
print("Your therapeutic audio files are ready for YouTube upload!")


def create_colab_notebook():
    """Helper function to create a .ipynb notebook file."""
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🎵 Audio Engine - Therapeutic Noise Generator\n",
                    "\n",
                    "Professional therapeutic noise generation optimized for YouTube content creation.\n",
                    "\n",
                    "## Features:\n",
                    "- Studio-quality white, pink, and brown noise\n",
                    "- YouTube LUFS compliance (-14 LUFS)\n",
                    "- Therapeutic frequency shaping for infant comfort\n",
                    "- GPU acceleration with CUDA\n",
                    "- Professional metadata embedding\n",
                    "\n",
                    "---"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": ["# Installation and Setup\n", "# (Copy Cell 1 code here)"]
            },
            {
                "cell_type": "code", 
                "execution_count": None,
                "metadata": {},
                "source": ["# Interactive Configuration\n", "# (Copy Cell 2 code here)"]
            },
            {
                "cell_type": "code",
                "execution_count": None, 
                "metadata": {},
                "source": ["# Audio Generation\n", "# (Copy Cell 3 code here)"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": ["# Quality Analysis\n", "# (Copy Cell 4 code here)"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": ["# Download and Metadata\n", "# (Copy Cell 5 code here)"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": ["# Batch Generation (Optional)\n", "# (Copy Cell 6 code here)"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": ["# Performance Monitoring\n", "# (Copy Cell 7 code here)"]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    import json
    with open("AudioEngine_Colab_Template.ipynb", "w") as f:
        json.dump(notebook_content, f, indent=2)
    
    print("✅ Colab notebook template created: AudioEngine_Colab_Template.ipynb")


if __name__ == "__main__":
    create_colab_notebook()