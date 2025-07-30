"""
Test suite for YouTube compliance, LUFS processing, and professional audio standards.
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_engine.processors.loudness_processor import LoudnessProcessor
from audio_engine.core.noise_generator import NoiseGenerator


class TestLoudnessProcessor(unittest.TestCase):
    """Test LUFS processing and loudness compliance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = LoudnessProcessor(
            sample_rate=48000,
            target_lufs=-14.0,
            true_peak_limit=-1.0,
            use_cuda=False
        )
        self.sample_rate = 48000
    
    def generate_test_audio(self, duration_seconds=2.0, amplitude=0.3):
        """Generate test audio for loudness testing."""
        samples = int(duration_seconds * self.sample_rate)
        return torch.randn(2, samples) * amplitude
    
    def test_processor_initialization(self):
        """Test proper initialization of loudness processor."""
        self.assertEqual(self.processor.sample_rate, 48000)
        self.assertEqual(self.processor.target_lufs, -14.0)
        self.assertEqual(self.processor.true_peak_limit, -1.0)
    
    def test_lufs_measurement(self):
        """Test LUFS measurement functionality."""
        # Generate test audio with known characteristics
        test_audio = self.generate_test_audio(duration_seconds=3.0, amplitude=0.1)
        
        # Measure LUFS
        measured_lufs = self.processor.measure_lufs(test_audio)
        
        # LUFS should be a reasonable value (not NaN or infinite)
        self.assertFalse(np.isnan(measured_lufs), "LUFS measurement should not be NaN")
        self.assertFalse(np.isinf(measured_lufs), "LUFS measurement should not be infinite")
        
        # Should be within a reasonable range for the test signal
        self.assertGreater(measured_lufs, -60.0, "LUFS should be above silence threshold")
        self.assertLess(measured_lufs, 0.0, "LUFS should be below 0 dB")
    
    def test_true_peak_measurement(self):
        """Test true peak measurement with oversampling."""
        # Generate test audio with known peak
        test_audio = self.generate_test_audio(amplitude=0.5)
        
        # Measure true peak
        true_peak = self.processor.measure_true_peak(test_audio)
        
        # True peak should be reasonable
        self.assertFalse(np.isnan(true_peak), "True peak should not be NaN")
        self.assertFalse(np.isinf(true_peak), "True peak should not be infinite")
        
        # Should be related to the input amplitude
        expected_peak_db = 20 * np.log10(0.5)  # Convert amplitude to dB
        self.assertLess(abs(true_peak - expected_peak_db), 10.0, 
                       "True peak should be approximately correct")
    
    def test_loudness_processing(self):
        """Test complete loudness processing pipeline."""
        # Generate test audio that needs adjustment
        test_audio = self.generate_test_audio(amplitude=0.1)  # Quiet audio
        
        # Process for LUFS compliance
        processed_audio = self.processor.process(test_audio)
        
        # Measure final LUFS
        final_lufs = self.processor.measure_lufs(processed_audio)
        
        # Should be close to target LUFS
        lufs_error = abs(final_lufs - self.processor.target_lufs)
        self.assertLess(lufs_error, 1.0, 
                       f"Final LUFS should be close to target: {final_lufs:.2f} vs {self.processor.target_lufs:.2f}")
        
        # Should not clip
        max_amplitude = torch.max(torch.abs(processed_audio)).item()
        self.assertLess(max_amplitude, 0.99, "Processing should not cause clipping")
    
    def test_true_peak_limiting(self):
        """Test true peak limiting functionality."""
        # Generate audio that might exceed true peak limit
        loud_audio = self.generate_test_audio(amplitude=0.9)
        
        # Process with limiting
        processed_audio = self.processor.process(loud_audio)
        
        # Measure true peak
        final_peak = self.processor.measure_true_peak(processed_audio)
        
        # Should be below limit with some tolerance
        self.assertLessEqual(final_peak, self.processor.true_peak_limit + 0.5,
                            f"True peak should be limited: {final_peak:.2f} dBTP")
    
    def test_youtube_compliance_validation(self):
        """Test YouTube compliance validation."""
        # Generate and process audio
        test_audio = self.generate_test_audio(duration_seconds=5.0)
        processed_audio = self.processor.process(test_audio)
        
        # Validate YouTube compliance
        compliance = self.processor.validate_youtube_compliance(processed_audio)
        
        # Check compliance structure
        required_keys = ['lufs_compliant', 'true_peak_compliant', 'measured_lufs', 
                        'measured_true_peak_dbtp', 'overall_compliant']
        for key in required_keys:
            self.assertIn(key, compliance, f"Compliance report should include {key}")
        
        # Should be compliant after processing
        self.assertTrue(compliance['overall_compliant'], 
                       "Processed audio should be YouTube compliant")
        
        # LUFS should be close to -14
        self.assertTrue(compliance['lufs_compliant'],
                       f"LUFS should be compliant: {compliance['measured_lufs']:.2f}")
        
        # True peak should be safe
        self.assertTrue(compliance['true_peak_compliant'],
                       f"True peak should be compliant: {compliance['measured_true_peak_dbtp']:.2f}")
    
    def test_gating_behavior(self):
        """Test loudness gating for accurate measurement."""
        # Create audio with quiet and loud sections
        duration_samples = 5 * self.sample_rate
        test_audio = torch.zeros(2, duration_samples)
        
        # Loud section in the middle
        start_loud = duration_samples // 3
        end_loud = 2 * duration_samples // 3
        test_audio[:, start_loud:end_loud] = torch.randn(2, end_loud - start_loud) * 0.3
        
        # Quiet sections at beginning and end (should be gated out)
        test_audio[:, :start_loud] = torch.randn(2, start_loud) * 0.01
        test_audio[:, end_loud:] = torch.randn(2, duration_samples - end_loud) * 0.01
        
        # Measure LUFS (gating should ignore quiet sections)
        measured_lufs = self.processor.measure_lufs(test_audio)
        
        # Should give reasonable measurement despite quiet sections
        self.assertGreater(measured_lufs, -40.0, "Gating should ignore quiet sections")
    
    def test_bs1770_filter_application(self):
        """Test that BS.1770-4 filters are applied correctly."""
        # Generate test audio
        test_audio = self.generate_test_audio(duration_seconds=2.0)
        
        # Apply BS.1770 filtering manually
        filtered_audio = self.processor._apply_bs1770_filtering(test_audio)
        
        # Filtered audio should be different from original
        self.assertFalse(torch.allclose(test_audio, filtered_audio, atol=1e-6),
                        "BS.1770 filtering should modify the audio")
        
        # Should maintain same shape
        self.assertEqual(filtered_audio.shape, test_audio.shape)
    
    def test_state_consistency(self):
        """Test filter state consistency across multiple calls."""
        audio1 = self.generate_test_audio(duration_seconds=1.0)
        audio2 = self.generate_test_audio(duration_seconds=1.0)
        
        # Process separately
        lufs1 = self.processor.measure_lufs(audio1)
        lufs2 = self.processor.measure_lufs(audio2)
        
        # Reset state and process again  
        self.processor.reset_state()
        lufs1_reset = self.processor.measure_lufs(audio1)
        
        # Should get consistent results after state reset
        self.assertAlmostEqual(lufs1, lufs1_reset, places=2,
                              msg="State reset should provide consistent results")


class TestYouTubeCompliance(unittest.TestCase):
    """Test complete YouTube compliance in the main generator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = NoiseGenerator(
            sample_rate=48000,      # YouTube native
            bit_depth=24,           # Professional
            target_lufs=-14.0,      # YouTube standard
            use_cuda=False
        )
    
    def test_sample_rate_compliance(self):
        """Test that audio is generated at YouTube-native sample rate."""
        self.assertEqual(self.generator.sample_rate, 48000,
                        "Should use YouTube-native 48kHz sample rate")
    
    def test_generated_audio_compliance(self):
        """Test that generated audio meets YouTube standards."""
        # Test different noise types
        noise_types = ['white', 'pink', 'brown']
        
        for noise_type in noise_types:
            with self.subTest(noise_type=noise_type):
                if noise_type == 'white':
                    audio = self.generator.generate_white_noise(0.1)
                elif noise_type == 'pink':
                    audio = self.generator.generate_pink_noise(0.1)
                elif noise_type == 'brown':
                    audio = self.generator.generate_brown_noise(0.1)
                
                # Validate compliance
                compliance = self.generator.loudness_processor.validate_youtube_compliance(audio)
                
                self.assertTrue(compliance['overall_compliant'],
                               f"{noise_type} noise should be YouTube compliant")
                
                # Check specific requirements
                self.assertTrue(compliance['lufs_compliant'],
                               f"{noise_type} LUFS: {compliance['measured_lufs']:.2f}")
                self.assertTrue(compliance['true_peak_compliant'],
                               f"{noise_type} True Peak: {compliance['measured_true_peak_dbtp']:.2f}")
    
    def test_long_form_content_compliance(self):
        """Test compliance for longer YouTube content."""
        # Test longer duration (simulate YouTube long-form content)
        audio = self.generator.generate_pink_noise(0.5)  # 30 seconds
        
        compliance = self.generator.loudness_processor.validate_youtube_compliance(audio)
        
        self.assertTrue(compliance['overall_compliant'],
                       "Long-form content should maintain YouTube compliance")
        
        # Check LUFS tolerance is reasonable
        lufs_error = abs(compliance['measured_lufs'] - (-14.0))
        self.assertLess(lufs_error, 0.5, 
                       f"LUFS should be within ±0.5 of target: {compliance['measured_lufs']:.2f}")
    
    def test_stereo_compliance(self):
        """Test that stereo audio maintains compliance."""
        stereo_audio = self.generator.generate_brown_noise(0.1, channels=2)
        
        compliance = self.generator.loudness_processor.validate_youtube_compliance(stereo_audio)
        
        self.assertTrue(compliance['overall_compliant'],
                       "Stereo audio should be YouTube compliant")
        
        # Both channels should contribute to loudness measurement
        measured_lufs = compliance['measured_lufs']
        self.assertGreater(measured_lufs, -30.0, "Stereo should provide adequate loudness")
    
    def test_metadata_youtube_optimization(self):
        """Test that metadata indicates YouTube optimization."""
        audio = self.generator.generate_white_noise(0.05)
        
        # Export with metadata
        test_file = "test_youtube_metadata.flac"
        self.generator.export_flac(test_file, audio, "white", 0.05)
        
        # Read and verify metadata
        from audio_engine.utils.metadata_handler import MetadataHandler
        metadata_handler = MetadataHandler()
        metadata = metadata_handler.read_metadata(test_file)
        
        if metadata:
            # Check for YouTube optimization indicators
            youtube_fields = {k: v for k, v in metadata.items() if k.startswith("YOUTUBE_")}
            self.assertTrue(len(youtube_fields) > 0, "Should have YouTube metadata")
            
            # Check specific YouTube optimization fields
            if "YOUTUBE_OPTIMIZED" in metadata:
                self.assertEqual(metadata["YOUTUBE_OPTIMIZED"], "True")
        
        # Cleanup
        Path(test_file).unlink(missing_ok=True)
    
    def test_normalization_headroom(self):
        """Test that audio has appropriate headroom for YouTube normalization."""
        audio = self.generator.generate_pink_noise(0.1)
        
        # Measure true peak
        true_peak = self.generator.loudness_processor.measure_true_peak(audio)
        
        # Should have adequate headroom (≤ -1 dBTP)
        self.assertLessEqual(true_peak, -1.0,
                            f"Should have normalization headroom: {true_peak:.2f} dBTP")
        
        # Should not be overly conservative
        self.assertGreater(true_peak, -6.0,
                          f"Should not be overly quiet: {true_peak:.2f} dBTP")


class TestProfessionalStandards(unittest.TestCase):
    """Test adherence to professional audio standards."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = NoiseGenerator(use_cuda=False)
    
    def test_bit_depth_compliance(self):
        """Test bit depth meets professional standards."""
        self.assertEqual(self.generator.bit_depth, 24,
                        "Should use professional 24-bit depth")
    
    def test_dynamic_range(self):
        """Test that audio maintains good dynamic range."""
        audio = self.generator.generate_brown_noise(0.2)
        
        # Calculate dynamic range (difference between peak and RMS)
        peak_level = torch.max(torch.abs(audio)).item()
        rms_level = torch.sqrt(torch.mean(audio**2)).item()
        
        dynamic_range_db = 20 * np.log10(peak_level / (rms_level + 1e-8))
        
        # Should have reasonable dynamic range
        self.assertGreater(dynamic_range_db, 6.0, "Should maintain adequate dynamic range")
        self.assertLess(dynamic_range_db, 30.0, "Dynamic range should be reasonable")
    
    def test_frequency_response_flatness(self):
        """Test frequency response characteristics."""
        white_audio = self.generator.generate_white_noise(0.2)
        
        # Analyze frequency response
        audio_np = white_audio[0].numpy()
        fft = np.fft.fft(audio_np)
        freqs = np.fft.fftfreq(len(audio_np), 1/self.generator.sample_rate)
        
        # Focus on audible range (20Hz - 20kHz)
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft[:len(fft)//2])
        
        audible_mask = (positive_freqs >= 20) & (positive_freqs <= 20000)
        audible_response = positive_fft[audible_mask]
        
        if len(audible_response) > 0:
            # Calculate response variation
            response_db = 20 * np.log10(audible_response + 1e-8)
            response_variation = np.std(response_db)
            
            # White noise should have relatively flat response after processing
            # (some variation is expected due to therapeutic EQ)
            self.assertLess(response_variation, 15.0, 
                           "Frequency response should be reasonably controlled")
    
    def test_thd_noise_performance(self):
        """Test THD+N (Total Harmonic Distortion + Noise) characteristics."""
        # Generate pure tone test signal
        duration = 0.1
        samples = int(duration * self.generator.sample_rate)
        
        # Since we're generating noise, we test that our processing doesn't add significant artifacts
        original_noise = torch.randn(2, samples) * 0.1
        
        # Process through our pipeline (without noise generation, just processing)
        processed_noise = self.generator.therapeutic_processor.process(original_noise)
        processed_noise = self.generator.loudness_processor.process(processed_noise)
        
        # Calculate difference (represents processing artifacts)
        difference = processed_noise - original_noise
        difference_rms = torch.sqrt(torch.mean(difference**2)).item()
        original_rms = torch.sqrt(torch.mean(original_noise**2)).item()
        
        # Processing artifacts should be minimal
        artifact_ratio = difference_rms / (original_rms + 1e-8)
        self.assertLess(artifact_ratio, 0.5, 
                       "Processing should not introduce significant artifacts")


if __name__ == '__main__':
    unittest.main(verbosity=2)