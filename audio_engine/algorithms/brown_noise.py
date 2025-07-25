"""
Brown Noise Algorithm Implementation

Generates brown noise through integration of white noise
with proper 1/f² spectral density for therapeutic applications.
"""

import torch
import numpy as np
from typing import Tuple, Optional
from .white_noise import WhiteNoiseAlgorithm


class BrownNoiseAlgorithm:
    """Brown noise generator through white noise integration."""
    
    def __init__(self, sample_rate: int = 48000, use_cuda: bool = False):
        """
        Initialize brown noise algorithm.
        
        Args:
            sample_rate: Target sample rate in Hz
            use_cuda: Enable CUDA acceleration if available
        """
        self.sample_rate = sample_rate
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        # Initialize white noise generator for integration
        self.white_generator = WhiteNoiseAlgorithm(sample_rate, use_cuda)
        
        # Integration state for continuous generation
        self.integration_state = torch.zeros(2, device=self.device)  # For stereo
        
        # High-pass filter coefficients to prevent DC buildup
        self.hp_cutoff = 1.0  # Hz
        self.hp_alpha = self._calculate_hp_alpha()
        self.hp_state = torch.zeros(2, device=self.device)
        
    def _calculate_hp_alpha(self) -> float:
        """Calculate high-pass filter coefficient."""
        rc = 1.0 / (2.0 * np.pi * self.hp_cutoff)
        dt = 1.0 / self.sample_rate
        return rc / (rc + dt)
    
    def generate(self, duration_samples: int, channels: int = 2) -> torch.Tensor:
        """
        Generate brown noise through integration of white noise.
        
        Args:
            duration_samples: Number of samples to generate
            channels: Number of output channels (1=mono, 2=stereo)
            
        Returns:
            torch.Tensor: Brown noise audio data [channels, samples]
        """
        # Generate white noise as base
        white_noise = self.white_generator.generate(duration_samples, channels)
        
        # Integrate white noise to create brown noise
        brown_noise = self._integrate_with_leakage(white_noise)
        
        # Apply high-pass filter to remove DC buildup
        brown_noise = self._apply_highpass_filter(brown_noise)
        
        # Normalize to prevent clipping while maintaining dynamics
        brown_noise = self._normalize_brown_noise(brown_noise)
        
        return brown_noise
    
    def _integrate_with_leakage(self, white_noise: torch.Tensor) -> torch.Tensor:
        """
        Integrate white noise with leakage to prevent unbounded growth.
        
        Args:
            white_noise: Input white noise [channels, samples]
            
        Returns:
            torch.Tensor: Integrated brown noise
        """
        channels, samples = white_noise.shape
        brown_noise = torch.zeros_like(white_noise)
        
        # Integration with small leakage factor
        leakage = 0.9999  # Very close to 1.0 for proper 1/f² characteristic
        
        # Use existing integration state for continuity
        current_state = self.integration_state[:channels].clone()
        
        for i in range(samples):
            # Leaky integration: y[n] = a*y[n-1] + x[n]
            current_state = leakage * current_state + white_noise[:, i]
            brown_noise[:, i] = current_state
        
        # Update integration state for next generation
        self.integration_state[:channels] = current_state
        
        return brown_noise
    
    def _apply_highpass_filter(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply high-pass filter to remove DC buildup.
        
        Args:
            signal: Input brown noise signal
            
        Returns:
            torch.Tensor: High-pass filtered signal
        """
        channels, samples = signal.shape
        filtered = torch.zeros_like(signal)
        
        # Use existing filter state for continuity
        current_hp_state = self.hp_state[:channels].clone()
        
        for i in range(samples):
            # First-order high-pass filter
            if i == 0:
                filtered[:, i] = signal[:, i] - current_hp_state
            else:
                filtered[:, i] = self.hp_alpha * (filtered[:, i-1] + signal[:, i] - signal[:, i-1])
        
        return filtered
    
    def _normalize_brown_noise(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Normalize brown noise while preserving its natural dynamic range.
        
        Args:
            signal: Input brown noise signal
            
        Returns:
            torch.Tensor: Normalized brown noise
        """
        # Calculate RMS for each channel
        rms = torch.sqrt(torch.mean(signal**2, dim=1, keepdim=True))
        
        # Target RMS level (slightly lower than white/pink for therapeutic use)
        target_rms = 0.15
        
        # Avoid division by zero
        rms = torch.clamp(rms, min=1e-8)
        
        # Normalize to target RMS
        normalized = signal * (target_rms / rms)
        
        # Soft limiting to prevent any clipping
        normalized = torch.tanh(normalized * 0.95) * 0.8
        
        return normalized
    
    def verify_spectral_density(self, signal: torch.Tensor) -> dict:
        """
        Verify that the generated signal has proper 1/f² spectral density.
        
        Args:
            signal: Generated brown noise signal
            
        Returns:
            dict: Spectral analysis results
        """
        # Convert to numpy for FFT analysis
        signal_np = signal[0].cpu().numpy() if self.use_cuda else signal[0].numpy()
        
        # Compute power spectral density
        freqs = np.fft.fftfreq(len(signal_np), 1/self.sample_rate)
        fft = np.fft.fft(signal_np)
        psd = np.abs(fft)**2
        
        # Focus on positive frequencies, exclude DC and very low frequencies
        start_idx = max(1, int(10 * len(freqs) / self.sample_rate))  # Start at ~10Hz
        positive_freqs = freqs[start_idx:len(freqs)//2]
        positive_psd = psd[start_idx:len(psd)//2]
        
        # Theoretical brown noise should have PSD ∝ 1/f²
        theoretical_psd = 1.0 / (positive_freqs**2)
        
        # Calculate correlation with theoretical 1/f² curve
        log_freqs = np.log10(positive_freqs)
        log_psd = np.log10(positive_psd)
        log_theoretical = np.log10(theoretical_psd)
        
        correlation = np.corrcoef(log_psd, log_theoretical)[0, 1]
        
        # Calculate slope in log-log domain (should be close to -2)
        slope = np.polyfit(log_freqs, log_psd, 1)[0]
        
        return {
            "spectral_correlation": correlation,
            "spectral_slope": slope,
            "expected_slope": -2.0,
            "slope_error": abs(slope + 2.0),
            "frequency_range": (positive_freqs[0], positive_freqs[-1]),
            "quality": "High" if correlation > 0.9 and abs(slope + 2.0) < 0.3 else "Medium"
        }
    
    def get_spectral_density(self) -> str:
        """
        Get theoretical spectral density for brown noise.
        
        Returns:
            str: Spectral density characteristic
        """
        return "1/f²"
    
    def reset_state(self) -> None:
        """Reset integration and filter states for fresh generation."""
        self.integration_state.fill_(0.0)
        self.hp_state.fill_(0.0)
    
    def get_algorithm_info(self) -> dict:
        """
        Get information about the algorithm implementation.
        
        Returns:
            dict: Algorithm metadata
        """
        return {
            "name": "Brown Noise",
            "algorithm": "Leaky Integration + High-pass",
            "distribution": "1/f² spectral density",
            "spectral_density": "1/f²",
            "quality": "High",
            "leakage_factor": 0.9999,
            "hp_cutoff": f"{self.hp_cutoff} Hz",
            "cuda_enabled": self.use_cuda
        }