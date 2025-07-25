"""
Pink Noise Algorithm Implementation

Implements the Voss-McCartney algorithm for perceptually uniform
distribution across octaves with proper 1/f spectral density.
"""

import torch
import numpy as np
from typing import Tuple, Optional


class PinkNoiseAlgorithm:
    """Pink noise generator using Voss-McCartney algorithm."""
    
    def __init__(self, sample_rate: int = 48000, use_cuda: bool = False):
        """
        Initialize pink noise algorithm.
        
        Args:
            sample_rate: Target sample rate in Hz
            use_cuda: Enable CUDA acceleration if available
        """
        self.sample_rate = sample_rate
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        # Voss-McCartney algorithm parameters
        self.num_sources = 16  # Number of white noise sources
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(42)
        
        # Initialize state for each source
        self.sources = torch.zeros(self.num_sources, device=self.device)
        self.counters = torch.zeros(self.num_sources, dtype=torch.long, device=self.device)
        
    def generate(self, duration_samples: int, channels: int = 2) -> torch.Tensor:
        """
        Generate pink noise using Voss-McCartney algorithm.
        
        Args:
            duration_samples: Number of samples to generate
            channels: Number of output channels (1=mono, 2=stereo)
            
        Returns:
            torch.Tensor: Pink noise audio data [channels, samples]
        """
        # Generate base pink noise for one channel
        pink_mono = self._generate_voss_mccartney(duration_samples)
        
        if channels == 1:
            return pink_mono.unsqueeze(0)
        elif channels == 2:
            # Create decorrelated stereo channels
            pink_left = pink_mono
            pink_right = self._generate_voss_mccartney(duration_samples)
            
            # Apply slight correlation for natural stereo field
            correlation = 0.3
            pink_right = correlation * pink_left + np.sqrt(1 - correlation**2) * pink_right
            
            return torch.stack([pink_left, pink_right])
        else:
            raise ValueError(f"Unsupported channel count: {channels}")
    
    def _generate_voss_mccartney(self, duration_samples: int) -> torch.Tensor:
        """
        Generate pink noise using the Voss-McCartney algorithm.
        
        Args:
            duration_samples: Number of samples to generate
            
        Returns:
            torch.Tensor: Pink noise samples
        """
        output = torch.zeros(duration_samples, device=self.device)
        
        # Reset sources for consistent generation
        self.sources.fill_(0.0)
        self.counters.fill_(0)
        
        for i in range(duration_samples):
            # Update sources based on bit patterns
            for j in range(self.num_sources):
                if (i >> j) & 1 != (max(0, i-1) >> j) & 1:
                    self.sources[j] = torch.randn(1, generator=self.generator, device=self.device)
            
            # Sum all sources to create pink noise
            output[i] = torch.sum(self.sources)
        
        # Normalize to prevent clipping
        output = output / torch.sqrt(torch.tensor(self.num_sources, device=self.device))
        
        # Apply additional filtering for better 1/f characteristic
        output = self._apply_pink_filter(output)
        
        return output
    
    def _apply_pink_filter(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply additional filtering to improve 1/f characteristic.
        
        Args:
            signal: Input signal to filter
            
        Returns:
            torch.Tensor: Filtered pink noise
        """
        # Simple first-order high-pass to improve low-frequency rolloff
        # This compensates for the Voss-McCartney algorithm's slight deviation from ideal 1/f
        
        if len(signal) < 2:
            return signal
        
        # First-order high-pass filter coefficients
        # Designed to flatten the low-frequency response
        a = 0.99
        filtered = torch.zeros_like(signal)
        filtered[0] = signal[0]
        
        for i in range(1, len(signal)):
            filtered[i] = a * filtered[i-1] + signal[i] - signal[i-1]
        
        return filtered
    
    def verify_spectral_density(self, signal: torch.Tensor) -> dict:
        """
        Verify that the generated signal has proper 1/f spectral density.
        
        Args:
            signal: Generated pink noise signal
            
        Returns:
            dict: Spectral analysis results
        """
        # Convert to numpy for FFT analysis
        signal_np = signal.cpu().numpy() if self.use_cuda else signal.numpy()
        
        # Compute power spectral density
        freqs = np.fft.fftfreq(len(signal_np), 1/self.sample_rate)
        fft = np.fft.fft(signal_np)
        psd = np.abs(fft)**2
        
        # Focus on positive frequencies, exclude DC
        positive_freqs = freqs[1:len(freqs)//2]
        positive_psd = psd[1:len(psd)//2]
        
        # Theoretical pink noise should have PSD ∝ 1/f
        theoretical_psd = 1.0 / positive_freqs
        
        # Calculate correlation with theoretical 1/f curve
        log_freqs = np.log10(positive_freqs)
        log_psd = np.log10(positive_psd)
        log_theoretical = np.log10(theoretical_psd)
        
        correlation = np.corrcoef(log_psd, log_theoretical)[0, 1]
        
        return {
            "spectral_correlation": correlation,
            "frequency_range": (positive_freqs[0], positive_freqs[-1]),
            "quality": "High" if correlation > 0.9 else "Medium" if correlation > 0.8 else "Low"
        }
    
    def get_spectral_density(self) -> str:
        """
        Get theoretical spectral density for pink noise.
        
        Returns:
            str: Spectral density characteristic
        """
        return "1/f"
    
    def get_algorithm_info(self) -> dict:
        """
        Get information about the algorithm implementation.
        
        Returns:
            dict: Algorithm metadata
        """
        return {
            "name": "Pink Noise",
            "algorithm": "Voss-McCartney",
            "distribution": "Perceptually uniform across octaves",
            "spectral_density": "1/f",
            "quality": "High",
            "num_sources": self.num_sources,
            "cuda_enabled": self.use_cuda
        }