"""
White Noise Algorithm Implementation

Uses Mersenne Twister PRNG with verified Gaussian distribution
for high-quality white noise generation.
"""

import torch
import numpy as np
from typing import Tuple, Optional


class WhiteNoiseAlgorithm:
    """High-quality white noise generator using Mersenne Twister."""
    
    def __init__(self, sample_rate: int = 48000, use_cuda: bool = False):
        """
        Initialize white noise algorithm.
        
        Args:
            sample_rate: Target sample rate in Hz
            use_cuda: Enable CUDA acceleration if available
        """
        self.sample_rate = sample_rate
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        # Set up Mersenne Twister generator
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(42)  # Reproducible results
        
    def generate(self, duration_samples: int, channels: int = 2) -> torch.Tensor:
        """
        Generate white noise with Gaussian distribution.
        
        Args:
            duration_samples: Number of samples to generate
            channels: Number of output channels (1=mono, 2=stereo)
            
        Returns:
            torch.Tensor: White noise audio data [channels, samples]
        """
        # Generate Gaussian white noise using Box-Muller transform
        # This ensures true Gaussian distribution, not just uniform
        shape = (channels, duration_samples)
        
        # Use PyTorch's randn for proper Gaussian distribution
        noise = torch.randn(shape, generator=self.generator, device=self.device)
        
        # Verify Gaussian distribution properties
        self._verify_gaussian_properties(noise)
        
        return noise
    
    def _verify_gaussian_properties(self, noise: torch.Tensor) -> None:
        """
        Verify that generated noise has proper Gaussian properties.
        
        Args:
            noise: Generated noise tensor to verify
        """
        # Check mean is close to 0
        mean = torch.mean(noise).item()
        assert abs(mean) < 0.01, f"Mean deviation too high: {mean}"
        
        # Check standard deviation is close to 1
        std = torch.std(noise).item()
        assert 0.95 < std < 1.05, f"Standard deviation out of range: {std}"
        
        # Check that 99.7% of samples are within 3 standard deviations (3-sigma rule)
        within_3sigma = torch.sum(torch.abs(noise) <= 3.0).item()
        total_samples = noise.numel()
        percentage = within_3sigma / total_samples
        assert percentage > 0.995, f"3-sigma rule violation: {percentage:.3f}"
    
    def get_spectral_density(self) -> float:
        """
        Get theoretical spectral density for white noise.
        
        Returns:
            float: Flat spectral density across all frequencies
        """
        return 1.0  # White noise has flat spectral density
    
    def get_algorithm_info(self) -> dict:
        """
        Get information about the algorithm implementation.
        
        Returns:
            dict: Algorithm metadata
        """
        return {
            "name": "White Noise",
            "algorithm": "Mersenne Twister + Box-Muller",
            "distribution": "Gaussian",
            "spectral_density": "Flat (1/f^0)",
            "quality": "High",
            "cuda_enabled": self.use_cuda
        }