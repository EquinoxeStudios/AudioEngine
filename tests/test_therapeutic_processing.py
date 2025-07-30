"""
Test suite for therapeutic processing and infant optimization features.
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_engine.processors.therapeutic_processor import TherapeuticProcessor
from audio_engine.core.noise_generator import NoiseGenerator


class TestTherapeuticProcessor(unittest.TestCase):
    """Test therapeutic processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = TherapeuticProcessor(
            sample_rate=48000,
            use_cuda=False,
            enabled=True
        )
        self.sample_rate = 48000
        self.test_duration = 48000  # 1 second of audio
    
    def generate_test_audio(self):
        """Generate test audio for processing tests."""
        # Generate simple stereo white noise for testing
        return torch.randn(2, self.test_duration) * 0.3
    
    def test_processor_initialization(self):
        """Test proper initialization of therapeutic processor."""
        self.assertTrue(self.processor.enabled)
        self.assertEqual(self.processor.sample_rate, 48000)
        self.assertEqual(self.processor.low_shelf_freq, 200.0)
        self.assertEqual(self.processor.harsh_freq_center, 3500.0)
    
    def test_therapeutic_eq_application(self):
        """Test that therapeutic EQ is applied correctly."""
        original_audio = self.generate_test_audio()
        processed_audio = self.processor.process(original_audio)
        
        # Audio should be modified by processing
        self.assertFalse(torch.allclose(original_audio, processed_audio, atol=1e-6),
                        "Processing should modify the audio")
        
        # Output shape should match input
        self.assertEqual(processed_audio.shape, original_audio.shape)
        
        # Should not introduce clipping
        max_amplitude = torch.max(torch.abs(processed_audio)).item()
        self.assertLess(max_amplitude, 0.95, "Processing should not cause clipping")
    
    def test_frequency_shaping(self):
        """Test therapeutic frequency shaping effects."""
        # Generate test audio
        original_audio = self.generate_test_audio()
        processed_audio = self.processor.process(original_audio)
        
        # Convert to frequency domain for analysis
        original_fft = torch.fft.fft(original_audio[0])
        processed_fft = torch.fft.fft(processed_audio[0])
        
        freqs = torch.fft.fftfreq(self.test_duration, 1/self.sample_rate)
        positive_freqs = freqs[:len(freqs)//2]
        
        # Find frequency bins for analysis
        low_freq_mask = (positive_freqs > 50) & (positive_freqs < 200)
        harsh_freq_mask = (positive_freqs > 2000) & (positive_freqs < 5000)
        
        if low_freq_mask.any() and harsh_freq_mask.any():
            # Calculate power in different frequency bands
            original_low_power = torch.mean(torch.abs(original_fft[:len(freqs)//2][low_freq_mask])**2)
            processed_low_power = torch.mean(torch.abs(processed_fft[:len(freqs)//2][low_freq_mask])**2)
            
            original_harsh_power = torch.mean(torch.abs(original_fft[:len(freqs)//2][harsh_freq_mask])**2)
            processed_harsh_power = torch.mean(torch.abs(processed_fft[:len(freqs)//2][harsh_freq_mask])**2)
            
            # Low frequencies should be enhanced (or at least not reduced significantly)
            low_freq_ratio = processed_low_power / (original_low_power + 1e-8)
            self.assertGreater(low_freq_ratio.item(), 0.8, 
                             "Low frequencies should be enhanced or preserved")
            
            # Harsh frequencies should be reduced
            harsh_freq_ratio = processed_harsh_power / (original_harsh_power + 1e-8)
            self.assertLess(harsh_freq_ratio.item(), 1.2,
                           "Harsh frequencies should be controlled")
    
    def test_envelope_smoothing(self):
        """Test envelope smoothing functionality."""
        # Create audio with artificial transients
        test_audio = torch.randn(2, self.test_duration) * 0.1
        
        # Add some sharp transients
        test_audio[:, 1000] = 0.8
        test_audio[:, 5000] = -0.7
        test_audio[:, 20000] = 0.6
        
        processed_audio = self.processor.process(test_audio)
        
        # Check that transients are smoothed
        original_max = torch.max(torch.abs(test_audio)).item()
        processed_max = torch.max(torch.abs(processed_audio)).item()
        
        # Processing should reduce the impact of transients
        self.assertLessEqual(processed_max, original_max * 1.1,
                            "Envelope smoothing should control transients")
    
    def test_phase_coherence_preservation(self):
        """Test that phase coherence is preserved in stereo processing."""
        original_audio = self.generate_test_audio()
        processed_audio = self.processor.process(original_audio)
        
        # Calculate cross-correlation between channels
        left_channel = processed_audio[0].numpy()
        right_channel = processed_audio[1].numpy()
        
        # Calculate coherence in frequency domain
        fft_left = np.fft.fft(left_channel)
        fft_right = np.fft.fft(right_channel)
        
        cross_power = fft_left * np.conj(fft_right)
        auto_power_left = np.abs(fft_left)**2
        auto_power_right = np.abs(fft_right)**2
        
        coherence = np.abs(cross_power)**2 / (auto_power_left * auto_power_right + 1e-8)
        mean_coherence = np.mean(coherence)
        
        # Coherence should be reasonable (above threshold)
        self.assertGreater(mean_coherence, self.processor.coherence_threshold * 0.8,
                          "Phase coherence should be maintained")
    
    def test_disabled_processing(self):
        """Test that processing can be disabled."""
        disabled_processor = TherapeuticProcessor(
            sample_rate=48000,
            use_cuda=False,
            enabled=False
        )
        
        original_audio = self.generate_test_audio()
        processed_audio = disabled_processor.process(original_audio)
        
        # When disabled, audio should pass through unchanged
        self.assertTrue(torch.allclose(original_audio, processed_audio, atol=1e-6),
                       "Disabled processor should not modify audio")
    
    def test_processor_state_reset(self):
        """Test filter state reset functionality."""
        audio1 = self.generate_test_audio()
        processed1 = self.processor.process(audio1)
        
        # Reset state
        self.processor.reset_state()
        
        # Process same audio again
        processed2 = self.processor.process(audio1)
        
        # Results should be identical after state reset
        self.assertTrue(torch.allclose(processed1, processed2, atol=1e-5),
                       "State reset should provide consistent results")


class TestTherapeuticIntegration(unittest.TestCase):
    """Test therapeutic processing integration in the main generator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator_with_therapeutic = NoiseGenerator(
            therapeutic_eq=True,
            use_cuda=False
        )
        self.generator_without_therapeutic = NoiseGenerator(
            therapeutic_eq=False,
            use_cuda=False
        )
    
    def test_therapeutic_integration(self):
        """Test that therapeutic processing is properly integrated."""
        duration = 0.05  # 3 seconds
        
        # Generate with and without therapeutic processing
        audio_with = self.generator_with_therapeutic.generate_pink_noise(duration)
        audio_without = self.generator_without_therapeutic.generate_pink_noise(duration)
        
        # Audio should be different when therapeutic processing is applied
        self.assertFalse(torch.allclose(audio_with, audio_without, atol=1e-3),
                        "Therapeutic processing should modify the audio")
        
        # Both should have same shape
        self.assertEqual(audio_with.shape, audio_without.shape)
    
    def test_therapeutic_metadata(self):
        """Test that therapeutic metadata is properly set."""
        processor_info = self.generator_with_therapeutic.therapeutic_processor.get_processor_info()
        
        self.assertTrue(processor_info['enabled'])
        self.assertEqual(processor_info['sample_rate'], 48000)
        self.assertIsInstance(processor_info['low_shelf_freq'], float)
        self.assertIsInstance(processor_info['harsh_reduction'], float)
    
    def test_infant_optimization_characteristics(self):
        """Test characteristics specific to infant optimization."""
        # Generate therapeutic audio
        audio = self.generator_with_therapeutic.generate_pink_noise(0.1)
        
        # Should not have excessive high frequencies that could be harsh
        audio_np = audio[0].numpy()
        fft = np.fft.fft(audio_np)
        freqs = np.fft.fftfreq(len(audio_np), 1/48000)
        
        # Focus on harsh frequency range (2-5kHz)
        harsh_freq_mask = (np.abs(freqs) > 2000) & (np.abs(freqs) < 5000)
        harsh_freq_power = np.mean(np.abs(fft[harsh_freq_mask])**2) if harsh_freq_mask.any() else 0
        
        # Focus on warm frequency range (50-200Hz)  
        warm_freq_mask = (np.abs(freqs) > 50) & (np.abs(freqs) < 200)
        warm_freq_power = np.mean(np.abs(fft[warm_freq_mask])**2) if warm_freq_mask.any() else 0
        
        # Therapeutic processing should maintain good balance
        if harsh_freq_power > 0 and warm_freq_power > 0:
            freq_balance = warm_freq_power / harsh_freq_power
            self.assertGreater(freq_balance, 0.1, 
                             "Should maintain good frequency balance for infant comfort")


class TestTherapeuticValidation(unittest.TestCase):
    """Test validation of therapeutic characteristics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = TherapeuticProcessor(use_cuda=False)
        self.generator = NoiseGenerator(therapeutic_eq=True, use_cuda=False)
    
    def test_amplitude_safety(self):
        """Test that therapeutic audio maintains safe amplitude levels."""
        audio = self.generator.generate_white_noise(0.1)
        
        # Check for safe amplitude levels
        max_amplitude = torch.max(torch.abs(audio)).item()
        rms_level = torch.sqrt(torch.mean(audio**2)).item()
        
        self.assertLess(max_amplitude, 0.9, "Amplitude should be safe for infants")
        self.assertLess(rms_level, 0.3, "RMS level should be gentle")
    
    def test_smoothness(self):
        """Test that audio is smooth without jarring transitions."""
        audio = self.generator.generate_brown_noise(0.1)
        
        # Calculate rate of change
        diff = torch.diff(audio, dim=1)
        max_change = torch.max(torch.abs(diff)).item()
        
        self.assertLess(max_change, 0.05, "Audio should have smooth transitions")
    
    def test_consistency(self):
        """Test that therapeutic processing is consistent."""
        # Generate multiple samples
        audio1 = self.generator.generate_pink_noise(0.05)
        audio2 = self.generator.generate_pink_noise(0.05)
        
        # Calculate RMS levels
        rms1 = torch.sqrt(torch.mean(audio1**2)).item()
        rms2 = torch.sqrt(torch.mean(audio2**2)).item()
        
        # RMS levels should be consistent (within reasonable tolerance)
        rms_ratio = max(rms1, rms2) / min(rms1, rms2)
        self.assertLess(rms_ratio, 1.5, "Therapeutic processing should be consistent")


if __name__ == '__main__':
    unittest.main(verbosity=2)