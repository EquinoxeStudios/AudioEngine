#!/usr/bin/env python3
"""
Advanced Features Demo - Audio Engine

Demonstrates advanced features including batch processing, custom EQ,
professional metadata handling, and quality validation.
"""

import sys
from pathlib import Path
import json
import time

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_engine import NoiseGenerator
from audio_engine.processors import TherapeuticProcessor, LoudnessProcessor
from audio_engine.utils import MetadataHandler, CUDAAccelerator


def demonstrate_batch_processing():
    """Demonstrate batch processing for multiple audio files."""
    print("🔄 Batch Processing Demo")
    print("-" * 25)
    
    generator = NoiseGenerator()
    
    # Define batch jobs
    batch_jobs = [
        {"type": "white", "duration": 30, "name": "meditation_white"},
        {"type": "pink", "duration": 60, "name": "sleep_pink"},
        {"type": "brown", "duration": 120, "name": "deep_sleep_brown"},
        {"type": "pink", "duration": 30, "name": "focus_pink"},
        {"type": "brown", "duration": 60, "name": "relaxation_brown"}
    ]
    
    print(f"Processing {len(batch_jobs)} audio files...")
    
    start_time = time.time()
    
    for i, job in enumerate(batch_jobs, 1):
        print(f"  [{i}/{len(batch_jobs)}] Generating {job['name']}...")
        
        # Generate audio based on type
        if job["type"] == "white":
            audio = generator.generate_white_noise(job["duration"])
        elif job["type"] == "pink":
            audio = generator.generate_pink_noise(job["duration"])
        elif job["type"] == "brown":
            audio = generator.generate_brown_noise(job["duration"])
        
        # Export with descriptive filename
        filename = f"batch_{job['name']}_{job['duration']}min.flac"
        generator.export_flac(
            filename,
            audio,
            noise_type=job["type"],
            duration_minutes=job["duration"]
        )
        
        print(f"    ✅ {filename}")
    
    total_time = time.time() - start_time
    print(f"\n📊 Batch completed in {total_time:.1f}s")
    print(f"Average per file: {total_time/len(batch_jobs):.1f}s")


def demonstrate_custom_processing():
    """Demonstrate custom therapeutic processing settings."""
    print("\n🎛️  Custom Processing Demo")
    print("-" * 30)
    
    # Create custom therapeutic processor
    therapeutic_processor = TherapeuticProcessor(
        sample_rate=48000,
        use_cuda=True,
        enabled=True
    )
    
    # Generate base audio
    generator = NoiseGenerator(therapeutic_eq=False)  # Disable built-in EQ
    base_audio = generator.generate_pink_noise(duration_minutes=0.5)
    
    print("Processing with custom therapeutic settings...")
    
    # Apply custom processing
    processed_audio = therapeutic_processor.process(base_audio)
    
    # Get processor info
    processor_info = therapeutic_processor.get_processor_info()
    print(f"Low shelf frequency: {processor_info['low_shelf_freq']} Hz")
    print(f"Harsh frequency reduction: {processor_info['harsh_reduction']} dB")
    print(f"Coherence threshold: {processor_info['coherence_threshold']}")
    
    # Export comparison files
    generator.export_flac("custom_original.flac", base_audio, "pink", 0.5)
    generator.export_flac("custom_processed.flac", processed_audio, "pink", 0.5)
    
    print("✅ Generated comparison files:")
    print("  - custom_original.flac (no therapeutic processing)")
    print("  - custom_processed.flac (with therapeutic processing)")


def demonstrate_loudness_analysis():
    """Demonstrate professional loudness analysis and compliance."""
    print("\n📊 Loudness Analysis Demo")
    print("-" * 28)
    
    generator = NoiseGenerator()
    
    # Generate test audio
    audio = generator.generate_brown_noise(duration_minutes=1.0)
    
    # Detailed loudness analysis
    loudness_processor = generator.loudness_processor
    
    # Measure current LUFS
    measured_lufs = loudness_processor.measure_lufs(audio)
    true_peak = loudness_processor.measure_true_peak(audio)
    
    print(f"Measured LUFS: {measured_lufs:.2f}")
    print(f"Target LUFS: {loudness_processor.target_lufs:.2f}")
    print(f"True Peak: {true_peak:.2f} dBTP")
    print(f"True Peak Limit: {loudness_processor.true_peak_limit:.2f} dBTP")
    
    # YouTube compliance check
    compliance = loudness_processor.validate_youtube_compliance(audio)
    
    print("\n🎯 YouTube Compliance:")
    print(f"  LUFS Compliant: {'✅' if compliance['lufs_compliant'] else '❌'}")
    print(f"  True Peak Compliant: {'✅' if compliance['true_peak_compliant'] else '❌'}")
    print(f"  Overall Compliant: {'✅' if compliance['overall_compliant'] else '❌'}")
    
    if not compliance['overall_compliant']:
        print(f"  LUFS Error: {compliance['lufs_error']:.2f} dB")
    
    # Export with analysis metadata
    generator.export_flac("loudness_analysis_demo.flac", audio, "brown", 1.0)
    
    # Get detailed measurements
    measurements = loudness_processor.get_measurements()
    if measurements:
        print(f"\n📈 Processing Details:")
        print(f"  Original LUFS: {measurements.get('original_lufs', 'N/A'):.2f}")
        print(f"  Gain Applied: {measurements.get('gain_applied_db', 'N/A'):.2f} dB")
        print(f"  Final LUFS: {measurements.get('final_lufs', 'N/A'):.2f}")


def demonstrate_metadata_handling():
    """Demonstrate professional metadata handling."""
    print("\n📝 Metadata Handling Demo")
    print("-" * 27)
    
    # Generate audio with custom metadata
    generator = NoiseGenerator()
    audio = generator.generate_white_noise(duration_minutes=0.5)
    
    # Custom metadata
    custom_metadata = {
        "noise_type": "white",
        "duration_minutes": 0.5,
        "lufs_target": -14.0,
        "sample_rate": 48000,
        "bit_depth": 24,
        "therapeutic": True,
        "youtube_optimized": True,
        "therapeutic_description": "Optimized for infant sleep and relaxation",
        "intended_use": "Sleep aid, focus enhancement, tinnitus masking",
        "generation_settings": {
            "algorithm": "Mersenne Twister + Box-Muller",
            "oversampling": "4x with anti-aliasing",
            "fade_duration": 5.0
        }
    }
    
    # Export with metadata
    filename = "metadata_demo.flac"
    generator.export_flac(filename, audio, **custom_metadata)
    
    # Demonstrate metadata reading
    metadata_handler = MetadataHandler()
    
    print("Reading embedded metadata...")
    metadata = metadata_handler.read_metadata(filename)
    
    if metadata:
        print("✅ Successfully read metadata:")
        
        # Display therapeutic metadata
        therapeutic_fields = {k: v for k, v in metadata.items() if k.startswith("THERAPEUTIC_")}
        if therapeutic_fields:
            print("\n🏥 Therapeutic Metadata:")
            for key, value in therapeutic_fields.items():
                clean_key = key.replace("THERAPEUTIC_", "").replace("_", " ").title()
                print(f"  {clean_key}: {value}")
        
        # Display technical metadata
        technical_fields = {k: v for k, v in metadata.items() if k.startswith("TECHNICAL_")}
        if technical_fields:
            print("\n⚙️  Technical Metadata:")
            for key, value in technical_fields.items():
                clean_key = key.replace("TECHNICAL_", "").replace("_", " ").title()
                print(f"  {clean_key}: {value}")
    
    # Generate comprehensive metadata report
    print("\n📄 Generating metadata report...")
    report = metadata_handler.generate_metadata_report(filename)
    
    # Export report to JSON
    report_filename = "metadata_report.json"
    metadata_handler.export_metadata_json(filename, report_filename)
    
    print(f"✅ Metadata report exported to: {report_filename}")
    
    # Display validation results
    validation = report.get("validation", {})
    print(f"\n✅ Metadata Validation:")
    print(f"  Valid: {'✅' if validation.get('valid', False) else '❌'}")
    print(f"  Therapeutic Compliant: {'✅' if validation.get('therapeutic_compliant', False) else '❌'}")
    print(f"  Technical Compliant: {'✅' if validation.get('technical_compliant', False) else '❌'}")
    
    if validation.get('errors'):
        print(f"  Errors: {len(validation['errors'])}")
    if validation.get('warnings'):
        print(f"  Warnings: {len(validation['warnings'])}")


def demonstrate_performance_profiling():
    """Demonstrate performance profiling and optimization."""
    print("\n⚡ Performance Profiling Demo")
    print("-" * 32)
    
    cuda_accelerator = CUDAAccelerator(use_cuda=True)
    generator = NoiseGenerator()
    
    # Profile different operations
    operations = [
        ("White Noise Generation", lambda: generator.generate_white_noise(0.5)),
        ("Pink Noise Generation", lambda: generator.generate_pink_noise(0.5)),
        ("Brown Noise Generation", lambda: generator.generate_brown_noise(0.5))
    ]
    
    print("Profiling operations...")
    
    for operation_name, operation_func in operations:
        result, profiling_info = cuda_accelerator.profile_operation(operation_func)
        
        print(f"\n📊 {operation_name}:")
        print(f"  Processing Time: {profiling_info['processing_time_seconds']:.3f}s")
        print(f"  Memory Used: {profiling_info['memory_used_mb']:.1f} MB")
        print(f"  Peak Memory: {profiling_info['memory_peak_mb']:.1f} MB")
        print(f"  Device: {profiling_info['device_type']}")
    
    # System performance report
    performance_report = cuda_accelerator.get_performance_report()
    
    print(f"\n💻 System Performance:")
    print(f"  Operations Count: {performance_report['performance_stats']['operations_count']}")
    print(f"  Average Time: {performance_report['performance_stats']['average_processing_time']:.3f}s")
    
    # Memory usage
    memory_usage = performance_report['memory_usage']
    print(f"\n💾 Memory Usage:")
    print(f"  System Memory: {memory_usage['system_memory_percent']:.1f}%")
    if cuda_accelerator.is_available():
        print(f"  GPU Memory: {memory_usage.get('gpu_memory_percent', 0):.1f}%")
    
    # Export performance report
    with open("performance_report.json", "w") as f:
        json.dump(performance_report, f, indent=2)
    
    print(f"✅ Performance report exported to: performance_report.json")


def main():
    """Main demonstration function for advanced features."""
    print("🎵 Audio Engine - Advanced Features Demo")
    print("=" * 50)
    
    # Demonstrate each advanced feature
    demonstrate_batch_processing()
    demonstrate_custom_processing()
    demonstrate_loudness_analysis()
    demonstrate_metadata_handling()
    demonstrate_performance_profiling()
    
    print("\n🎉 Advanced features demo complete!")
    print("\nGenerated files:")
    print("- Batch processing: batch_*.flac")
    print("- Custom processing: custom_*.flac")
    print("- Loudness analysis: loudness_analysis_demo.flac")
    print("- Metadata demo: metadata_demo.flac")
    print("- Reports: *.json")


if __name__ == "__main__":
    main()