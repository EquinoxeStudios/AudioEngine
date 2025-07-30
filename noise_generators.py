import torch
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class BaseNoiseGenerator:
    """Base class for all noise generators"""
    
    def __init__(self, sample_rate: int, device: Optional[torch.device] = None):
        self.sample_rate = sample_rate
        self.device = device or torch.device('cpu')
    
    def generate(self, num_samples: int) -> torch.Tensor:
        """Generate noise samples. Must be implemented by subclasses."""
        raise NotImplementedError


class WhiteNoiseGenerator(BaseNoiseGenerator):
    """
    White noise generator using Mersenne Twister PRNG with Gaussian distribution.
    Generates truly continuous white noise without segments.
    """
    
    def __init__(self, sample_rate: int, device: Optional[torch.device] = None):
        super().__init__(sample_rate, device)
        # Set random seed for reproducibility if needed
        self.generator = torch.Generator(device=device)
    
    def generate(self, num_samples: int) -> torch.Tensor:
        """
        Generate continuous white noise.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Stereo white noise tensor [2, num_samples]
        """
        # Generate Gaussian white noise directly
        # Using randn for Gaussian distribution (mean=0, std=1)
        noise = torch.randn(2, num_samples, device=self.device, generator=self.generator)
        
        # Normalize to prevent clipping
        noise = noise * 0.5
        
        return noise


class PinkNoiseGenerator(BaseNoiseGenerator):
    """
    Pink noise generator using the Voss-McCartney algorithm.
    Generates 1/f noise with proper spectral characteristics.
    """
    
    def __init__(self, sample_rate: int, device: Optional[torch.device] = None, num_octaves: int = 10):
        super().__init__(sample_rate, device)
        self.num_octaves = num_octaves
        self.generators = [torch.Generator(device=device) for _ in range(num_octaves)]
        
        # Initialize octave values for Voss-McCartney
        self.octave_values = torch.zeros(num_octaves, device=device)
        self.counter = 0
    
    def generate(self, num_samples: int) -> torch.Tensor:
        """
        Generate continuous pink noise using Voss-McCartney algorithm.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Stereo pink noise tensor [2, num_samples]
        """
        # Pre-allocate output tensor
        output = torch.zeros(2, num_samples, device=self.device)
        
        # Generate pink noise sample by sample for continuity
        for i in range(num_samples):
            # Update octave values based on binary representation of counter
            for octave in range(self.num_octaves):
                if (self.counter >> octave) & 1:
                    # Update this octave with new random value
                    self.octave_values[octave] = torch.randn(
                        1, device=self.device, generator=self.generators[octave]
                    ).item()
            
            # Sum all octave values for current sample
            sample_value = self.octave_values.sum() / self.num_octaves
            
            # Apply to both channels with slight variation for stereo
            output[0, i] = sample_value
            output[1, i] = sample_value * 0.98 + torch.randn(1, device=self.device).item() * 0.02
            
            self.counter += 1
        
        # Normalize output
        output = output * 0.5
        
        # Apply additional 1/f filtering for better spectral characteristics
        output = self._apply_pinking_filter(output)
        
        return output
    
    def _apply_pinking_filter(self, white_noise: torch.Tensor) -> torch.Tensor:
        """Apply additional 1/f filtering to improve pink noise characteristics"""
        # Create frequency domain representation
        fft = torch.fft.rfft(white_noise, dim=-1)
        freqs = torch.fft.rfftfreq(white_noise.shape[-1], 1/self.sample_rate, device=self.device)
        
        # Apply 1/f filter (avoiding division by zero)
        freqs[0] = 1e-10  # Avoid division by zero at DC
        pink_filter = 1 / torch.sqrt(freqs)
        pink_filter[0] = 0  # Remove DC component
        
        # Apply filter and convert back to time domain
        fft = fft * pink_filter.unsqueeze(0)
        pink_noise = torch.fft.irfft(fft, n=white_noise.shape[-1], dim=-1)
        
        # Normalize
        rms = torch.sqrt(torch.mean(pink_noise ** 2))
        if rms > 0:
            pink_noise = pink_noise / rms * 0.5
        
        return pink_noise


class BrownNoiseGenerator(BaseNoiseGenerator):
    """
    Brown noise generator using integration method.
    Generates 1/f² noise (Brownian motion).
    """
    
    def __init__(self, sample_rate: int, device: Optional[torch.device] = None):
        super().__init__(sample_rate, device)
        self.generator = torch.Generator(device=device)
        
        # State for continuous generation
        self.last_value = torch.zeros(2, device=device)
    
    def generate(self, num_samples: int) -> torch.Tensor:
        """
        Generate continuous brown noise using integration.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Stereo brown noise tensor [2, num_samples]
        """
        # Generate white noise increments
        increments = torch.randn(2, num_samples, device=self.device, generator=self.generator)
        
        # Scale increments for proper brown noise characteristics
        # Smaller increments for smoother motion
        increments = increments * 0.02
        
        # Integrate (cumulative sum) to get brown noise
        brown_noise = torch.cumsum(increments, dim=1)
        
        # Add last value from previous generation for continuity
        brown_noise = brown_noise + self.last_value.unsqueeze(1)
        
        # Update last value for next generation
        self.last_value = brown_noise[:, -1].clone()
        
        # Apply high-pass filter to remove DC drift
        brown_noise = self._remove_dc_drift(brown_noise)
        
        # Apply proper 1/f² spectral shaping
        brown_noise = self._apply_brown_filter(brown_noise)
        
        # Normalize to prevent clipping
        rms = torch.sqrt(torch.mean(brown_noise ** 2))
        if rms > 0:
            brown_noise = brown_noise / rms * 0.5
        
        return brown_noise
    
    def _remove_dc_drift(self, signal: torch.Tensor) -> torch.Tensor:
        """Remove DC drift using high-pass filter at 1-2Hz"""
        # Simple DC removal using running mean
        window_size = int(self.sample_rate * 0.5)  # 0.5 second window
        
        if signal.shape[1] > window_size:
            # Compute running mean
            kernel = torch.ones(1, 1, window_size, device=self.device) / window_size
            padded = torch.nn.functional.pad(signal.unsqueeze(1), (window_size//2, window_size//2), mode='reflect')
            running_mean = torch.nn.functional.conv1d(padded, kernel).squeeze(1)
            
            # Subtract running mean
            signal = signal - running_mean
        
        return signal
    
    def _apply_brown_filter(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply proper 1/f² spectral shaping"""
        # Transform to frequency domain
        fft = torch.fft.rfft(signal, dim=-1)
        freqs = torch.fft.rfftfreq(signal.shape[-1], 1/self.sample_rate, device=self.device)
        
        # Create 1/f² filter
        freqs[0] = 1e-10  # Avoid division by zero
        brown_filter = 1 / (freqs ** 2)
        brown_filter[0] = 0  # Remove DC
        
        # Limit filter gain to prevent numerical issues
        brown_filter = torch.minimum(brown_filter, torch.tensor(1000.0, device=self.device))
        
        # Apply filter
        fft = fft * brown_filter.unsqueeze(0)
        
        # Convert back to time domain
        filtered = torch.fft.irfft(fft, n=signal.shape[-1], dim=-1)
        
        return filtered