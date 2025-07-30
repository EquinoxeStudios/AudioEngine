"""
TherapeuticProcessor - Infant-Optimized Audio Processing

Implements evidence-based frequency shaping and processing specifically
designed for infant comfort and therapeutic effectiveness.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any
import torch.nn.functional as F


class TherapeuticProcessor:
    """
    Processor for infant-optimized therapeutic audio enhancement.
    
    Features:
    - Frequency shaping optimized for infant hearing sensitivity
    - Harshness reduction in 2-5kHz range
    - Low-end warmth enhancement below 200Hz
    - Phase coherence preservation
    - Envelope smoothing for comfort
    """
    
    def __init__(
        self,
        sample_rate: int = 48000,
        use_cuda: bool = True,
        enabled: bool = True
    ):
        """
        Initialize therapeutic processor.
        
        Args:
            sample_rate: Audio sample rate
            use_cuda: Enable CUDA acceleration
            enabled: Enable/disable therapeutic processing
        """
        self.sample_rate = sample_rate
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.enabled = enabled
        
        # Therapeutic EQ parameters
        self.low_shelf_freq = 200.0  # Hz - warmth enhancement
        self.low_shelf_gain = 2.0    # dB - gentle boost
        self.harsh_freq_center = 3500.0  # Hz - harshness reduction center
        self.harsh_freq_width = 2000.0   # Hz - reduction bandwidth
        self.harsh_reduction = -4.0      # dB - harshness reduction amount
        
        # Initialize filter coefficients
        self._init_eq_filters()
        
        # Envelope smoothing parameters
        self.envelope_attack = 0.001   # Very fast attack
        self.envelope_release = 0.1    # Slower release for smoothness
        
        # Phase coherence checking parameters
        self.coherence_threshold = 0.85  # Minimum acceptable coherence
        
    def _init_eq_filters(self) -> None:
        """Initialize EQ filter coefficients."""
        nyquist = self.sample_rate / 2
        
        # Low shelf filter (Butterworth 2nd order)
        self.low_shelf_coeffs = self._calculate_low_shelf_coeffs(
            self.low_shelf_freq / nyquist,
            self.low_shelf_gain
        )
        
        # Harsh frequency notch filter (Butterworth bandstop)
        self.harsh_notch_coeffs = self._calculate_bandstop_coeffs(
            self.harsh_freq_center / nyquist,
            self.harsh_freq_width / nyquist,
            self.harsh_reduction
        )
        
        # Initialize filter states
        self.low_shelf_state = torch.zeros(2, 4, device=self.device)  # [channels, states]
        self.harsh_notch_state = torch.zeros(2, 4, device=self.device)
    
    def _calculate_low_shelf_coeffs(self, freq: float, gain_db: float) -> torch.Tensor:
        """Calculate low shelf filter coefficients."""
        # Convert gain to linear
        A = 10**(gain_db / 40)
        omega = 2 * np.pi * freq
        cos_omega = np.cos(omega)
        sin_omega = np.sin(omega)
        alpha = sin_omega / 2 * np.sqrt((A + 1/A) * (1/0.707 - 1) + 2)
        
        # Calculate coefficients
        b0 = A * ((A + 1) - (A - 1) * cos_omega + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_omega)
        b2 = A * ((A + 1) - (A - 1) * cos_omega - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * cos_omega + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_omega)
        a2 = (A + 1) + (A - 1) * cos_omega - 2 * np.sqrt(A) * alpha
        
        # Normalize
        coeffs = torch.tensor([b0/a0, b1/a0, b2/a0, a1/a0, a2/a0], device=self.device)
        return coeffs
    
    def _calculate_bandstop_coeffs(
        self, 
        center_freq: float, 
        bandwidth: float, 
        gain_db: float
    ) -> torch.Tensor:
        """Calculate bandstop filter coefficients for harshness reduction."""
        # Convert gain to linear
        A = 10**(gain_db / 40)
        omega = 2 * np.pi * center_freq
        cos_omega = np.cos(omega)
        sin_omega = np.sin(omega)
        alpha = sin_omega * np.sinh(np.log(2) / 2 * bandwidth * omega / sin_omega)
        
        # Bandstop coefficients
        b0 = 1
        b1 = -2 * cos_omega
        b2 = 1
        a0 = 1 + alpha
        a1 = -2 * cos_omega
        a2 = 1 - alpha
        
        # Apply gain reduction
        b0 *= A
        b1 *= A
        b2 *= A
        
        # Normalize
        coeffs = torch.tensor([b0/a0, b1/a0, b2/a0, a1/a0, a2/a0], device=self.device)
        return coeffs
    
    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply therapeutic processing to audio.
        
        Args:
            audio: Input audio tensor [channels, samples]
            
        Returns:
            torch.Tensor: Therapeutically processed audio
        """
        if not self.enabled:
            return audio
        
        processed = audio.clone()
        
        # 1. Apply therapeutic EQ
        processed = self._apply_therapeutic_eq(processed)
        
        # 2. Apply envelope smoothing
        processed = self._apply_envelope_smoothing(processed)
        
        # 3. Check and maintain phase coherence
        processed = self._maintain_phase_coherence(processed, audio)
        
        # 4. Apply gentle dynamics processing
        processed = self._apply_gentle_dynamics(processed)
        
        # 5. Final therapeutic validation
        self._validate_therapeutic_characteristics(processed)
        
        return processed
    
    def _apply_therapeutic_eq(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply infant-optimized EQ curve."""
        channels, samples = audio.shape
        processed = audio.clone()
        
        # Apply low shelf for warmth
        processed = self._apply_biquad_filter(
            processed, 
            self.low_shelf_coeffs, 
            self.low_shelf_state
        )
        
        # Apply harshness reduction
        processed = self._apply_biquad_filter(
            processed, 
            self.harsh_notch_coeffs, 
            self.harsh_notch_state
        )
        
        return processed
    
    def _apply_biquad_filter(
        self, 
        audio: torch.Tensor, 
        coeffs: torch.Tensor, 
        state: torch.Tensor
    ) -> torch.Tensor:
        """Apply biquad filter with state preservation."""
        channels, samples = audio.shape
        filtered = torch.zeros_like(audio)
        
        # Extract coefficients
        b0, b1, b2, a1, a2 = coeffs
        
        for ch in range(channels):
            # Get current state
            x1, x2, y1, y2 = state[ch]
            
            for i in range(samples):
                x0 = audio[ch, i]
                
                # Biquad difference equation
                y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
                
                filtered[ch, i] = y0
                
                # Update state
                x2, x1 = x1, x0
                y2, y1 = y1, y0
            
            # Store state for next call
            state[ch] = torch.tensor([x1, x2, y1, y2])
        
        return filtered
    
    def _apply_envelope_smoothing(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply envelope smoothing to eliminate micro-transients."""
        channels, samples = audio.shape
        
        # Calculate envelope using RMS with smoothing
        window_size = int(0.01 * self.sample_rate)  # 10ms windows
        
        smoothed = torch.zeros_like(audio)
        
        for ch in range(channels):
            signal = audio[ch]
            
            # Calculate RMS envelope
            envelope = torch.zeros(samples)
            for i in range(samples):
                start = max(0, i - window_size // 2)
                end = min(samples, i + window_size // 2)
                envelope[i] = torch.sqrt(torch.mean(signal[start:end]**2))
            
            # Smooth envelope with attack/release
            smooth_envelope = torch.zeros_like(envelope)
            smooth_envelope[0] = envelope[0]
            
            for i in range(1, samples):
                if envelope[i] > smooth_envelope[i-1]:
                    # Attack
                    alpha = 1 - np.exp(-1 / (self.envelope_attack * self.sample_rate))
                else:
                    # Release
                    alpha = 1 - np.exp(-1 / (self.envelope_release * self.sample_rate))
                
                smooth_envelope[i] = (
                    alpha * envelope[i] + (1 - alpha) * smooth_envelope[i-1]
                )
            
            # Apply envelope-based gain reduction for smoothness
            # Only reduce peaks, never amplify
            gain = torch.where(
                envelope > 0,
                torch.clamp(smooth_envelope / (envelope + 1e-8), 0, 1),
                torch.ones_like(envelope)
            )
            
            smoothed[ch] = signal * gain
        
        return smoothed
    
    def _maintain_phase_coherence(
        self, 
        processed: torch.Tensor, 
        original: torch.Tensor
    ) -> torch.Tensor:
        """Ensure phase coherence is maintained for therapeutic effectiveness."""
        if processed.shape[0] < 2:
            return processed  # Can't check coherence with mono
        
        # Calculate cross-correlation between channels
        left = processed[0].cpu().numpy()
        right = processed[1].cpu().numpy()
        
        # Compute coherence using FFT
        fft_left = np.fft.fft(left)
        fft_right = np.fft.fft(right)
        
        cross_power = fft_left * np.conj(fft_right)
        auto_power_left = np.abs(fft_left)**2
        auto_power_right = np.abs(fft_right)**2
        
        coherence = np.abs(cross_power)**2 / (auto_power_left * auto_power_right + 1e-8)
        mean_coherence = np.mean(coherence)
        
        # If coherence is too low, blend with original
        if mean_coherence < self.coherence_threshold:
            blend_factor = (self.coherence_threshold - mean_coherence) / self.coherence_threshold
            processed = (1 - blend_factor) * processed + blend_factor * original
        
        return processed
    
    def _apply_gentle_dynamics(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply gentle compression to control dynamics."""
        # Very gentle compression with slow attack/release
        threshold = 0.7    # High threshold for gentle action
        ratio = 2.0        # Mild compression ratio
        attack_time = 0.1  # 100ms attack
        release_time = 0.5 # 500ms release
        
        # Calculate gain reduction
        rms = torch.sqrt(torch.mean(audio**2, dim=1, keepdim=True))
        
        # Simple soft-knee compression
        gain_reduction = torch.where(
            rms > threshold,
            threshold + (rms - threshold) / ratio,
            rms
        )
        
        # Apply gain smoothing
        gain = gain_reduction / (rms + 1e-8)
        gain = torch.clamp(gain, 0.1, 1.0)  # Limit gain reduction
        
        return audio * gain
    
    def _validate_therapeutic_characteristics(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Validate that audio maintains therapeutic characteristics."""
        # Calculate frequency response
        audio_np = audio[0].cpu().numpy()
        fft = np.fft.fft(audio_np)
        freqs = np.fft.fftfreq(len(audio_np), 1/self.sample_rate)
        
        # Check therapeutic frequency ranges
        low_freq_idx = np.where((freqs > 20) & (freqs < 200))[0]
        harsh_freq_idx = np.where((freqs > 2000) & (freqs < 5000))[0]
        
        low_freq_power = np.mean(np.abs(fft[low_freq_idx])**2) if len(low_freq_idx) > 0 else 0
        harsh_freq_power = np.mean(np.abs(fft[harsh_freq_idx])**2) if len(harsh_freq_idx) > 0 else 0
        
        # Therapeutic validation metrics
        validation = {
            "low_freq_enhanced": low_freq_power > harsh_freq_power,
            "harshness_reduced": harsh_freq_power < low_freq_power * 0.7,
            "peak_amplitude": torch.max(torch.abs(audio)).item(),
            "rms_level": torch.sqrt(torch.mean(audio**2)).item()
        }
        
        return validation
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get processor configuration and status."""
        return {
            "enabled": self.enabled,
            "sample_rate": self.sample_rate,
            "cuda_enabled": self.use_cuda,
            "low_shelf_freq": self.low_shelf_freq,
            "low_shelf_gain": self.low_shelf_gain,
            "harsh_freq_center": self.harsh_freq_center,
            "harsh_reduction": self.harsh_reduction,
            "coherence_threshold": self.coherence_threshold,
            "envelope_smoothing": True
        }
    
    def reset_state(self) -> None:
        """Reset filter states for fresh processing."""
        self.low_shelf_state.fill_(0.0)
        self.harsh_notch_state.fill_(0.0)