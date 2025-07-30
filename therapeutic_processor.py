import torch
import torchaudio
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TherapeuticProcessor:
    """
    Applies infant-optimized audio processing for therapeutic effectiveness.
    Implements frequency shaping based on infant hearing sensitivity research.
    """
    
    def __init__(self, sample_rate: int, device: Optional[torch.device] = None):
        """
        Initialize therapeutic processor.
        
        Args:
            sample_rate: Audio sample rate
            device: Torch device for processing
        """
        self.sample_rate = sample_rate
        self.device = device or torch.device('cpu')
        
        # Frequency bands for therapeutic EQ
        self.low_shelf_freq = 200  # Hz
        self.low_shelf_gain = 3    # dB - gentle warmth
        
        self.mid_notch_freq = 3500  # Hz (center of 2-5kHz range)
        self.mid_notch_q = 2.0      # Q factor
        self.mid_notch_gain = -3    # dB - reduce harshness
        
        # Initialize filters
        self._init_filters()
        
        logger.info("TherapeuticProcessor initialized")
    
    def _init_filters(self):
        """Initialize therapeutic EQ filters"""
        # Note: Using IIR filters for efficiency
        # In production, might want to use linear phase FIR filters
        
        # Calculate normalized frequencies
        self.low_shelf_norm = self.low_shelf_freq / (self.sample_rate / 2)
        self.mid_notch_norm = self.mid_notch_freq / (self.sample_rate / 2)
    
    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply therapeutic processing to audio.
        
        Args:
            audio: Input audio tensor [channels, samples]
            
        Returns:
            Processed audio tensor
        """
        # Apply low shelf filter for warmth
        audio = self._apply_low_shelf(audio)
        
        # Apply mid-frequency notch to reduce harshness
        audio = self._apply_mid_notch(audio)
        
        # Apply gentle envelope smoothing
        audio = self._smooth_envelope(audio)
        
        # Ensure phase coherence
        audio = self._ensure_phase_coherence(audio)
        
        return audio
    
    def _apply_low_shelf(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply low shelf filter for gentle bass enhancement"""
        # Using butterworth low shelf approximation
        # For production, consider using proper shelf filter design
        
        # Create low-pass and high-pass components
        nyquist = self.sample_rate / 2
        
        # Low shelf implementation using combination of filters
        # Low-pass for boosted low frequencies
        if self.low_shelf_freq < nyquist * 0.9:  # Safety check
            # Apply gentle low frequency boost
            # Using FFT for frequency domain processing
            fft = torch.fft.rfft(audio, dim=-1)
            freqs = torch.fft.rfftfreq(audio.shape[-1], 1/self.sample_rate, device=self.device)
            
            # Create shelf response
            shelf_response = torch.ones_like(freqs)
            low_mask = freqs <= self.low_shelf_freq
            
            # Smooth transition using sigmoid
            transition_width = 50  # Hz
            transition = torch.sigmoid((self.low_shelf_freq - freqs) / transition_width)
            
            # Apply gain in linear scale
            gain_linear = 10 ** (self.low_shelf_gain / 20)
            shelf_response = 1 + (gain_linear - 1) * transition
            
            # Apply to FFT
            fft = fft * shelf_response.unsqueeze(0)
            
            # Convert back to time domain
            audio = torch.fft.irfft(fft, n=audio.shape[-1], dim=-1)
        
        return audio
    
    def _apply_mid_notch(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply notch filter to reduce 2-5kHz harshness"""
        # Parametric EQ notch filter
        fft = torch.fft.rfft(audio, dim=-1)
        freqs = torch.fft.rfftfreq(audio.shape[-1], 1/self.sample_rate, device=self.device)
        
        # Create notch response
        # Using parametric EQ curve
        bandwidth = self.mid_notch_freq / self.mid_notch_q
        
        # Calculate distance from center frequency
        freq_ratio = freqs / self.mid_notch_freq
        freq_ratio[freq_ratio == 0] = 1e-10  # Avoid log(0)
        
        # Parametric EQ response
        gain_linear = 10 ** (self.mid_notch_gain / 20)
        
        # Bell curve centered at notch frequency
        response = torch.ones_like(freqs)
        mask = (freqs > 2000) & (freqs < 5000)  # 2-5kHz range
        
        # Smooth bell curve for the notch
        distance = torch.abs(torch.log2(freq_ratio))
        bell = torch.exp(-distance**2 * self.mid_notch_q)
        
        response[mask] = 1 + (gain_linear - 1) * bell[mask]
        
        # Apply to FFT
        fft = fft * response.unsqueeze(0)
        
        # Convert back to time domain
        audio = torch.fft.irfft(fft, n=audio.shape[-1], dim=-1)
        
        return audio
    
    def _smooth_envelope(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply envelope smoothing to eliminate micro-transients"""
        # Use gentle compression/limiting approach
        
        # Calculate envelope using Hilbert transform
        analytic = torch.view_as_complex(
            torch.stack([
                audio,
                torch.zeros_like(audio)
            ], dim=-1)
        )
        
        # Get magnitude envelope
        envelope = torch.abs(torch.fft.ifft(torch.fft.fft(analytic, dim=-1), dim=-1))
        envelope = envelope[..., 0]  # Take real part
        
        # Smooth the envelope
        smooth_window = int(0.005 * self.sample_rate)  # 5ms window
        if smooth_window > 1:
            kernel = torch.ones(1, 1, smooth_window, device=self.device) / smooth_window
            envelope_padded = torch.nn.functional.pad(
                envelope.unsqueeze(1), 
                (smooth_window//2, smooth_window//2), 
                mode='reflect'
            )
            smooth_envelope = torch.nn.functional.conv1d(envelope_padded, kernel).squeeze(1)
        else:
            smooth_envelope = envelope
        
        # Apply gentle limiting based on smoothed envelope
        threshold = 0.9
        ratio = 4.0  # 4:1 compression ratio
        
        # Calculate gain reduction
        over_threshold = torch.maximum(smooth_envelope - threshold, torch.tensor(0.0, device=self.device))
        gain_reduction = 1 - (over_threshold * (1 - 1/ratio))
        
        # Apply gain with smooth attack/release
        audio = audio * gain_reduction
        
        return audio
    
    def _ensure_phase_coherence(self, audio: torch.Tensor) -> torch.Tensor:
        """Ensure phase coherence between channels for therapeutic effectiveness"""
        if audio.shape[0] == 2:  # Stereo
            # Measure phase coherence
            fft_l = torch.fft.rfft(audio[0])
            fft_r = torch.fft.rfft(audio[1])
            
            # Calculate phase difference
            phase_diff = torch.angle(fft_l) - torch.angle(fft_r)
            
            # Identify frequencies with excessive phase difference
            # Allow some phase difference for stereo width, but limit extremes
            max_phase_diff = np.pi / 3  # 60 degrees
            
            # Smooth phase differences
            problem_freqs = torch.abs(phase_diff) > max_phase_diff
            
            if problem_freqs.any():
                # Blend channels at problematic frequencies
                blend_factor = 0.5
                avg_magnitude = (torch.abs(fft_l) + torch.abs(fft_r)) / 2
                avg_phase = (torch.angle(fft_l) + torch.angle(fft_r)) / 2
                
                # Create blended FFT
                blended = avg_magnitude * torch.exp(1j * avg_phase)
                
                # Apply blending only to problematic frequencies
                fft_l[problem_freqs] = fft_l[problem_freqs] * (1 - blend_factor) + blended[problem_freqs] * blend_factor
                fft_r[problem_freqs] = fft_r[problem_freqs] * (1 - blend_factor) + blended[problem_freqs] * blend_factor
                
                # Convert back to time domain
                audio[0] = torch.fft.irfft(fft_l, n=audio.shape[-1])
                audio[1] = torch.fft.irfft(fft_r, n=audio.shape[-1])
        
        return audio
    
    def get_frequency_response(self, num_points: int = 1024) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the frequency response of the therapeutic EQ.
        
        Args:
            num_points: Number of frequency points to calculate
            
        Returns:
            Tuple of (frequencies, magnitude_db)
        """
        # Generate frequency points
        freqs = torch.linspace(20, self.sample_rate / 2, num_points, device=self.device)
        
        # Calculate combined response
        response_db = torch.zeros_like(freqs)
        
        # Low shelf response
        transition = torch.sigmoid((self.low_shelf_freq - freqs) / 50)
        response_db += self.low_shelf_gain * transition
        
        # Mid notch response
        freq_ratio = freqs / self.mid_notch_freq
        freq_ratio[freq_ratio == 0] = 1e-10
        distance = torch.abs(torch.log2(freq_ratio))
        bell = torch.exp(-distance**2 * self.mid_notch_q)
        mask = (freqs > 2000) & (freqs < 5000)
        response_db[mask] += self.mid_notch_gain * bell[mask]
        
        return freqs, response_db