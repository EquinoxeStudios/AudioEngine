"""
Test suite for noise generation algorithms and core functionality.
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_engine.core.noise_generator import NoiseGenerator
from audio_engine.algorithms.white_noise import WhiteNoiseAlgorithm
from audio_engine.algorithms.pink_noise import PinkNoiseAlgorithm
from audio_engine.algorithms.brown_noise import BrownNoiseAlgorithm


class TestNoiseAlgorithms(unittest.TestCase):
    """Test individual noise generation algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_rate = 48000
        self.test_duration = 1000  # samples
        self.channels = 2
        
    def test_white_noise_algorithm(self):
        """Test white noise algorithm properties."""
        algo = WhiteNoiseAlgorithm(self.sample_rate, use_cuda=False)
        noise = algo.generate(self.test_duration, self.channels)
        
        # Test output shape
        self.assertEqual(noise.shape, (self.channels, self.test_duration))
        
        # Test Gaussian properties
        mean = torch.mean(noise).item()
        std = torch.std(noise).item()
        
        self.assertLess(abs(mean), 0.1, "Mean should be close to 0")
        self.assertGreater(std, 0.8, "Standard deviation should be close to 1")
        self.assertLess(std, 1.2, "Standard deviation should be close to 1")
        
        # Test that it's actually random (not all zeros)
        self.assertGreater(torch.var(noise).item(), 0.1)
    
    def test_pink_noise_algorithm(self):
        """Test pink noise algorithm properties."""
        algo = PinkNoiseAlgorithm(self.sample_rate, use_cuda=False)
        noise = algo.generate(self.test_duration, self.channels)
        
        # Test output shape
        self.assertEqual(noise.shape, (self.channels, self.test_duration))
        
        # Test spectral properties
        spectral_info = algo.verify_spectral_density(noise)
        self.assertGreater(spectral_info["spectral_correlation"], 0.7, 
                          "Pink noise should have good 1/f correlation")
        
        # Test dynamic range
        self.assertGreater(torch.var(noise).item(), 0.01)
    
    def test_brown_noise_algorithm(self):
        """Test brown noise algorithm properties."""
        algo = BrownNoiseAlgorithm(self.sample_rate, use_cuda=False)
        noise = algo.generate(self.test_duration, self.channels)
        
        # Test output shape
        self.assertEqual(noise.shape, (self.channels, self.test_duration))
        
        # Test spectral properties
        spectral_info = algo.verify_spectral_density(noise)
        self.assertGreater(spectral_info["spectral_correlation"], 0.7,
                          "Brown noise should have good 1/f² correlation")
        
        # Brown noise should have lower high-frequency content
        self.assertLess(abs(spectral_info["spectral_slope"] + 2.0), 0.5,
                       "Spectral slope should be close to -2")


class TestNoiseGenerator(unittest.TestCase):
    """Test the main NoiseGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = NoiseGenerator(
            sample_rate=48000,
            bit_depth=24,
            target_lufs=-14.0,
            use_cuda=False,  # Use CPU for consistent testing
            therapeutic_eq=True,
            fade_duration=1.0
        )
    
    def test_generator_initialization(self):
        """Test proper initialization of the generator."""
        self.assertEqual(self.generator.sample_rate, 48000)
        self.assertEqual(self.generator.bit_depth, 24)
        self.assertEqual(self.generator.target_lufs, -14.0)
        self.assertTrue(self.generator.therapeutic_processor.enabled)
    
    def test_white_noise_generation(self):
        """Test white noise generation through main interface."""
        duration_minutes = 0.1  # 6 seconds
        audio = self.generator.generate_white_noise(duration_minutes, channels=2)
        
        expected_samples = int(duration_minutes * 60 * self.generator.sample_rate)
        self.assertEqual(audio.shape, (2, expected_samples))
        
        # Test audio is not silent
        rms = torch.sqrt(torch.mean(audio**2)).item()
        self.assertGreater(rms, 0.01, "Generated audio should not be silent")
    
    def test_pink_noise_generation(self):
        """Test pink noise generation through main interface."""
        duration_minutes = 0.1
        audio = self.generator.generate_pink_noise(duration_minutes, channels=2)
        
        expected_samples = int(duration_minutes * 60 * self.generator.sample_rate)
        self.assertEqual(audio.shape, (2, expected_samples))
        
        # Test audio characteristics
        rms = torch.sqrt(torch.mean(audio**2)).item()
        self.assertGreater(rms, 0.01, "Generated audio should not be silent")
        
        # Test no clipping
        max_amplitude = torch.max(torch.abs(audio)).item()
        self.assertLess(max_amplitude, 1.0, "Audio should not clip")
    
    def test_brown_noise_generation(self):
        """Test brown noise generation through main interface."""
        duration_minutes = 0.1
        audio = self.generator.generate_brown_noise(duration_minutes, channels=2)
        
        expected_samples = int(duration_minutes * 60 * self.generator.sample_rate)
        self.assertEqual(audio.shape, (2, expected_samples))
        
        # Test audio characteristics
        rms = torch.sqrt(torch.mean(audio**2)).item()
        self.assertGreater(rms, 0.01, "Generated audio should not be silent")
    
    def test_fade_application(self):
        """Test that fades are properly applied."""
        duration_minutes = 0.1
        audio = self.generator.generate_white_noise(duration_minutes, channels=2)
        
        fade_samples = int(self.generator.fade_duration * self.generator.sample_rate)
        
        # Test fade-in (should start quiet and increase)
        start_amplitude = torch.abs(audio[:, 0]).mean().item()
        after_fade_amplitude = torch.abs(audio[:, fade_samples]).mean().item()
        self.assertLess(start_amplitude, after_fade_amplitude, 
                       "Fade-in should increase amplitude")
        
        # Test fade-out (should end quiet)
        end_amplitude = torch.abs(audio[:, -1]).mean().item()
        before_fade_amplitude = torch.abs(audio[:, -fade_samples-1]).mean().item()
        self.assertLess(end_amplitude, before_fade_amplitude,
                       "Fade-out should decrease amplitude")
    
    def test_dc_offset_removal(self):
        """Test DC offset removal."""
        duration_minutes = 0.1
        audio = self.generator.generate_white_noise(duration_minutes, channels=2)
        
        dc_offset = torch.mean(audio).item()
        self.assertLess(abs(dc_offset), 0.01, 
                       "DC offset should be minimal after processing")
    
    def test_stereo_generation(self):
        """Test stereo audio generation."""
        duration_minutes = 0.05
        audio = self.generator.generate_pink_noise(duration_minutes, channels=2)
        
        # Channels should be different (decorrelated)
        correlation = torch.corrcoef(audio)[0, 1].item()
        self.assertLess(correlation, 0.9, 
                       "Stereo channels should be decorrelated")
        self.assertGreater(correlation, 0.1,
                          "Stereo channels should have some correlation")
    
    def test_mono_generation(self):
        """Test mono audio generation."""
        duration_minutes = 0.05
        audio = self.generator.generate_white_noise(duration_minutes, channels=1)
        
        self.assertEqual(audio.shape[0], 1, "Should generate mono audio")


class TestAudioQuality(unittest.TestCase):
    """Test audio quality metrics and validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = NoiseGenerator(use_cuda=False)
    
    def test_amplitude_levels(self):
        """Test that audio levels are appropriate."""
        audio = self.generator.generate_pink_noise(0.05, channels=2)
        
        # Test no clipping
        max_amplitude = torch.max(torch.abs(audio)).item()
        self.assertLess(max_amplitude, 0.95, "Audio should not approach clipping")
        
        # Test reasonable RMS level
        rms = torch.sqrt(torch.mean(audio**2)).item()
        self.assertGreater(rms, 0.01, "RMS should be reasonable")
        self.assertLess(rms, 0.5, "RMS should not be too high")
    
    def test_frequency_content(self):
        """Test frequency content is reasonable."""
        audio = self.generator.generate_white_noise(0.1, channels=1)
        
        # Convert to numpy for FFT
        audio_np = audio[0].numpy()
        fft = np.fft.fft(audio_np)
        freqs = np.fft.fftfreq(len(audio_np), 1/self.generator.sample_rate)
        
        # Test that we have content across the frequency spectrum
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft[:len(fft)//2])
        
        # Should have significant energy in multiple frequency bands
        low_freq_energy = np.mean(positive_fft[:len(positive_fft)//4])
        mid_freq_energy = np.mean(positive_fft[len(positive_fft)//4:3*len(positive_fft)//4])
        high_freq_energy = np.mean(positive_fft[3*len(positive_fft)//4:])
        
        self.assertGreater(low_freq_energy, 0, "Should have low frequency content")
        self.assertGreater(mid_freq_energy, 0, "Should have mid frequency content")
        self.assertGreater(high_freq_energy, 0, "Should have high frequency content")
    
    def test_no_artifacts(self):
        """Test for common audio artifacts."""
        audio = self.generator.generate_brown_noise(0.1, channels=2)
        
        # Test for sudden jumps (discontinuities)
        diff = torch.diff(audio, dim=1)
        max_diff = torch.max(torch.abs(diff)).item()
        self.assertLess(max_diff, 0.1, "Should not have sudden amplitude jumps")
        
        # Test for NaN or infinite values
        self.assertFalse(torch.isnan(audio).any(), "Should not contain NaN values")
        self.assertFalse(torch.isinf(audio).any(), "Should not contain infinite values")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)