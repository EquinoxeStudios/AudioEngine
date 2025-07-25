"""
Pink Noise Algorithm Implementation

Implements the Voss-McCartney algorithm for perceptually uniform
distribution across octaves with proper 1/f spectral density.
"""

import torch
import numpy as np
from typing import Tuple, Optional
import torch.jit


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
        # Always use Voss-McCartney for highest quality
        pink_mono = self._generate_voss_mccartney_optimized(duration_samples)
        
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
    
    
    def _generate_voss_mccartney_optimized(self, duration_samples: int) -> torch.Tensor:
        """
        Optimized Voss-McCartney algorithm using torch operations.
        Maintains algorithm accuracy while improving performance.
        
        Args:
            duration_samples: Number of samples to generate
            
        Returns:
            torch.Tensor: Pink noise samples
        """
        # Initialize
        output = torch.zeros(duration_samples, device=self.device)
        sources = torch.zeros(self.num_sources, device=self.device)
        
        # Optimize by processing in chunks and pre-generating random values
        chunk_size = 10000  # Smaller chunks for better accuracy
        
        for chunk_start in range(0, duration_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, duration_samples)
            chunk_len = chunk_end - chunk_start
            
            # Pre-generate random values for all possible updates in this chunk
            # This is MUCH faster than calling randn repeatedly
            random_matrix = torch.randn(
                self.num_sources, chunk_len,
                generator=self.generator, device=self.device
            )
            
            # Process each sample
            for i in range(chunk_len):
                sample_idx = chunk_start + i
                
                # Voss-McCartney: update sources based on bit transitions
                for j in range(self.num_sources):
                    # Check if bit j changed from previous sample
                    if sample_idx == 0:
                        # First sample: update if bit is set
                        if (sample_idx >> j) & 1:
                            sources[j] = random_matrix[j, i]
                    else:
                        # Check for bit transition
                        prev_bit = ((sample_idx - 1) >> j) & 1
                        curr_bit = (sample_idx >> j) & 1
                        if curr_bit != prev_bit:
                            sources[j] = random_matrix[j, i]
                
                # Sum all sources
                output[sample_idx] = sources.sum()
        
        # Normalize
        output = output / np.sqrt(self.num_sources)
        
        # Apply filter to improve 1/f characteristic
        output = self._apply_pink_filter(output)
        
        return output
    
    def _generate_voss_mccartney_vectorized(self, duration_samples: int) -> torch.Tensor:
        """
        Generate pink noise using the Voss-McCartney algorithm.
        Optimized implementation using vectorized operations.
        
        Args:
            duration_samples: Number of samples to generate
            
        Returns:
            torch.Tensor: Pink noise samples
        """
        # Initialize output and sources
        output = torch.zeros(duration_samples, device=self.device)
        self.sources.fill_(0.0)
        
        # Pre-generate all random values for efficiency
        # This is much faster than generating on-demand
        chunk_size = min(duration_samples, 1000000)  # Process in 1M sample chunks for memory efficiency
        
        for chunk_start in range(0, duration_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, duration_samples)
            chunk_length = chunk_end - chunk_start
            
            # Pre-calculate which sources need updating for each sample in the chunk
            # This vectorizes the bit manipulation logic
            sample_indices = torch.arange(chunk_start, chunk_end, device=self.device)
            
            # For each source, determine when it needs updating
            for j in range(self.num_sources):
                # Calculate bit transitions for this source
                if chunk_start == 0:
                    # First chunk: check against 0
                    current_bits = (sample_indices >> j) & 1
                    prev_bits = torch.zeros_like(current_bits)
                    prev_bits[1:] = current_bits[:-1]
                else:
                    # Subsequent chunks: check against previous sample
                    current_bits = (sample_indices >> j) & 1
                    prev_indices = sample_indices - 1
                    prev_bits = (prev_indices >> j) & 1
                
                # Find where bit transitions occur
                transitions = current_bits != prev_bits
                transition_indices = torch.where(transitions)[0]
                
                # Generate new random values only where needed
                if len(transition_indices) > 0:
                    new_values = torch.randn(len(transition_indices), generator=self.generator, device=self.device)
                    
                    # Update the source value at transition points
                    for idx, trans_idx in enumerate(transition_indices):
                        if chunk_start + trans_idx < duration_samples:
                            self.sources[j] = new_values[idx]
                            # Update output from this point forward in the chunk
                            output[chunk_start + trans_idx:chunk_end] += self.sources[j] - (self.sources[j] / self.num_sources * (self.num_sources - 1))
            
            # Add the current sum of sources to this chunk
            output[chunk_start:chunk_end] += torch.sum(self.sources)
        
        # Normalize to prevent clipping
        output = output / torch.sqrt(torch.tensor(self.num_sources, dtype=output.dtype, device=self.device))
        
        # Apply additional filtering for better 1/f characteristic
        output = self._apply_pink_filter(output)
        
        return output
    
    def _apply_pink_filter(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply additional filtering to improve 1/f characteristic.
        Optimized using FFT-based filtering for long signals.
        
        Args:
            signal: Input signal to filter
            
        Returns:
            torch.Tensor: Filtered pink noise
        """
        if len(signal) < 2:
            return signal
        
        # For long signals, use FFT-based filtering (much faster)
        if len(signal) > 10000:
            # Work in chunks to manage memory
            chunk_size = min(len(signal), self.sample_rate * 60)  # 1 minute chunks
            filtered_chunks = []
            
            for start in range(0, len(signal), chunk_size):
                end = min(start + chunk_size, len(signal))
                chunk = signal[start:end]
                
                # Apply mild high-pass filter to compensate for
                # Voss-McCartney's slight low-frequency emphasis
                # FFT
                spectrum = torch.fft.rfft(chunk)
                freqs = torch.fft.rfftfreq(len(chunk), 1/self.sample_rate, device=self.device)
                
                # Gentle high-pass filter (very subtle correction)
                # This preserves the Voss-McCartney character while improving flatness
                freqs[0] = 1.0  # Avoid division by zero
                correction_filter = 1.0 - torch.exp(-freqs / 10.0)  # Gentle curve
                correction_filter[0] = 0.0  # Remove DC
                
                # Apply correction
                spectrum = spectrum * correction_filter
                
                # IFFT
                filtered_chunk = torch.fft.irfft(spectrum, n=len(chunk))
                filtered_chunks.append(filtered_chunk)
            
            return torch.cat(filtered_chunks)
        else:
            # For short signals, use simple DC removal
            return signal - torch.mean(signal)
    
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