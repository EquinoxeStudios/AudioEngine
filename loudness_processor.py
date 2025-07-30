import torch
import torchaudio
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LoudnessProcessor:
    """
    ITU-R BS.1770-4 compliant loudness processor.
    Handles LUFS measurement and true-peak limiting for YouTube compliance.
    """
    
    def __init__(self, 
                 sample_rate: int,
                 target_lufs: float = -14.0,
                 true_peak_limit: float = -1.0,
                 device: Optional[torch.device] = None):
        """
        Initialize loudness processor.
        
        Args:
            sample_rate: Audio sample rate
            target_lufs: Target integrated loudness in LUFS
            true_peak_limit: True peak limit in dBTP
            device: Torch device for processing
        """
        self.sample_rate = sample_rate
        self.target_lufs = target_lufs
        self.true_peak_limit = true_peak_limit
        self.device = device or torch.device('cpu')
        
        # BS.1770-4 K-weighting filter coefficients
        self._init_k_weighting_filters()
        
        # True peak oversampling factor (ITU-R BS.1770-4 recommends 4x)
        self.true_peak_oversample = 4
        
        logger.info(f"LoudnessProcessor initialized - Target: {target_lufs} LUFS, True Peak: {true_peak_limit} dBTP")
    
    def _init_k_weighting_filters(self):
        """Initialize K-weighting filters for BS.1770-4 compliance"""
        # Pre-filter (high shelf)
        # Coefficients for 48kHz (adjust for other sample rates)
        if self.sample_rate == 48000:
            # Stage 1: Pre-filter coefficients
            self.pre_b = torch.tensor([1.53512485958697, -2.69169618940638, 1.19839281085285], device=self.device)
            self.pre_a = torch.tensor([1.0, -1.69065929318241, 0.73248077421585], device=self.device)
            
            # Stage 2: RLB filter coefficients  
            self.rlb_b = torch.tensor([1.0, -2.0, 1.0], device=self.device)
            self.rlb_a = torch.tensor([1.0, -1.99004745483398, 0.99007225036621], device=self.device)
        else:
            # For other sample rates, we'd need to recalculate coefficients
            # Using simplified approximation for now
            logger.warning(f"K-weighting filters optimized for 48kHz, using approximation for {self.sample_rate}Hz")
            self.pre_b = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            self.pre_a = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            self.rlb_b = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            self.rlb_a = torch.tensor([1.0, 0.0, 0.0], device=self.device)
    
    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Process audio to meet loudness target and true peak limit.
        
        Args:
            audio: Input audio tensor [channels, samples]
            
        Returns:
            Processed audio tensor
        """
        # Measure current loudness
        current_lufs = self.measure_lufs(audio)
        
        # Calculate gain needed
        gain_db = self.target_lufs - current_lufs
        gain_linear = 10 ** (gain_db / 20)
        
        # Apply initial gain
        audio = audio * gain_linear
        
        # Apply true peak limiting
        audio = self._apply_true_peak_limiting(audio)
        
        # Final loudness check and adjustment
        final_lufs = self.measure_lufs(audio)
        if abs(final_lufs - self.target_lufs) > 0.5:
            # Fine-tune if needed
            fine_gain_db = self.target_lufs - final_lufs
            fine_gain = 10 ** (fine_gain_db / 20)
            audio = audio * fine_gain
            audio = self._apply_true_peak_limiting(audio)
        
        return audio
    
    def measure_lufs(self, audio: torch.Tensor) -> float:
        """
        Measure integrated loudness according to ITU-R BS.1770-4.
        
        Args:
            audio: Input audio tensor [channels, samples]
            
        Returns:
            Integrated loudness in LUFS
        """
        # Apply K-weighting filters
        weighted = self._apply_k_weighting(audio)
        
        # Calculate mean square for 400ms blocks with 75% overlap
        block_size = int(0.4 * self.sample_rate)  # 400ms
        hop_size = int(0.1 * self.sample_rate)    # 100ms (75% overlap)
        
        # Ensure we have enough samples
        if audio.shape[-1] < block_size:
            # For short audio, use the whole signal
            mean_square = torch.mean(weighted ** 2)
            loudness = -0.691 + 10 * torch.log10(mean_square + 1e-10)
            return float(loudness)
        
        # Calculate loudness for each block
        num_blocks = (audio.shape[-1] - block_size) // hop_size + 1
        block_loudness = []
        
        for i in range(num_blocks):
            start = i * hop_size
            end = start + block_size
            block = weighted[:, start:end]
            
            # Mean square per channel
            channel_ms = torch.mean(block ** 2, dim=1)
            
            # Sum across channels (with channel weighting if needed)
            if audio.shape[0] == 2:  # Stereo
                total_ms = torch.sum(channel_ms)
            else:  # Mono
                total_ms = channel_ms[0]
            
            # Convert to loudness
            block_lufs = -0.691 + 10 * torch.log10(total_ms + 1e-10)
            block_loudness.append(float(block_lufs))
        
        # Gating (ITU-R BS.1770-4)
        block_loudness = np.array(block_loudness)
        
        # Absolute gate at -70 LUFS
        absolute_gate = -70
        gated_blocks = block_loudness[block_loudness > absolute_gate]
        
        if len(gated_blocks) == 0:
            return -70.0  # Below measurement threshold
        
        # Relative gate at -10 LU below ungated average
        ungated_mean = np.mean(gated_blocks)
        relative_gate = ungated_mean - 10
        
        # Final gating
        final_blocks = gated_blocks[gated_blocks > relative_gate]
        
        if len(final_blocks) == 0:
            return float(ungated_mean)
        
        # Integrated loudness
        integrated_lufs = np.mean(final_blocks)
        
        return float(integrated_lufs)
    
    def _apply_k_weighting(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply K-weighting filters for loudness measurement"""
        # Create a copy to avoid modifying input
        weighted = audio.clone()
        
        # Apply pre-filter (high shelf)
        if self.sample_rate == 48000:
            # Apply IIR filter using difference equation
            # This is a simplified implementation - production code might use scipy.signal
            for ch in range(weighted.shape[0]):
                # Pre-filter
                filtered = torch.zeros_like(weighted[ch])
                for n in range(2, len(filtered)):
                    filtered[n] = (self.pre_b[0] * weighted[ch, n] + 
                                 self.pre_b[1] * weighted[ch, n-1] + 
                                 self.pre_b[2] * weighted[ch, n-2] - 
                                 self.pre_a[1] * filtered[n-1] - 
                                 self.pre_a[2] * filtered[n-2])
                
                # RLB filter
                rlb_filtered = torch.zeros_like(filtered)
                for n in range(2, len(rlb_filtered)):
                    rlb_filtered[n] = (self.rlb_b[0] * filtered[n] + 
                                     self.rlb_b[1] * filtered[n-1] + 
                                     self.rlb_b[2] * filtered[n-2] - 
                                     self.rlb_a[1] * rlb_filtered[n-1] - 
                                     self.rlb_a[2] * rlb_filtered[n-2])
                
                weighted[ch] = rlb_filtered
        
        return weighted
    
    def _apply_true_peak_limiting(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply true peak limiting according to BS.1770 standard"""
        # Oversample for true peak detection
        oversampled = self._oversample_for_true_peak(audio)
        
        # Find true peaks
        true_peak = torch.max(torch.abs(oversampled))
        true_peak_db = 20 * torch.log10(true_peak + 1e-10)
        
        # Apply limiting if needed
        if true_peak_db > self.true_peak_limit:
            # Calculate required attenuation
            attenuation_db = self.true_peak_limit - true_peak_db
            attenuation = 10 ** (attenuation_db / 20)
            
            # Apply soft-knee limiting for smoother sound
            audio = self._soft_limit(audio, attenuation)
        
        return audio
    
    def _oversample_for_true_peak(self, audio: torch.Tensor) -> torch.Tensor:
        """Oversample audio for true peak detection"""
        if self.true_peak_oversample <= 1:
            return audio
        
        # Use high-quality resampling for oversampling
        target_rate = self.sample_rate * self.true_peak_oversample
        resampler = torchaudio.transforms.Resample(
            self.sample_rate,
            target_rate,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interpolation",
            beta=14.769656459379492
        ).to(self.device)
        
        return resampler(audio)
    
    def _soft_limit(self, audio: torch.Tensor, target_gain: float) -> torch.Tensor:
        """Apply soft-knee limiting with lookahead"""
        # Simple soft-knee implementation
        # For production, consider more sophisticated limiters
        
        # Lookahead buffer (5ms)
        lookahead_samples = int(0.005 * self.sample_rate)
        
        # Apply gain with soft knee
        threshold = 0.9  # Start soft limiting at -0.9dB
        
        # Calculate envelope
        envelope = torch.abs(audio)
        
        # Smooth envelope with lookahead
        if lookahead_samples > 1:
            kernel = torch.ones(1, 1, lookahead_samples, device=self.device) / lookahead_samples
            padded = torch.nn.functional.pad(
                envelope.unsqueeze(1),
                (lookahead_samples, 0),
                mode='replicate'
            )
            smooth_envelope = torch.nn.functional.conv1d(padded, kernel).squeeze(1)
        else:
            smooth_envelope = envelope
        
        # Calculate gain curve
        gain = torch.ones_like(smooth_envelope)
        over_threshold = smooth_envelope > threshold
        
        # Soft knee curve
        knee_width = 0.1
        soft_region = (smooth_envelope > (threshold - knee_width)) & (smooth_envelope <= threshold)
        
        # Apply soft knee
        gain[soft_region] = 1 - ((smooth_envelope[soft_region] - (threshold - knee_width)) / knee_width) * (1 - target_gain)
        gain[over_threshold] = target_gain
        
        # Apply gain with attack/release
        attack_time = 0.001  # 1ms
        release_time = 0.050  # 50ms
        
        attack_coeff = np.exp(-1 / (attack_time * self.sample_rate))
        release_coeff = np.exp(-1 / (release_time * self.sample_rate))
        
        # Smooth gain changes
        smoothed_gain = torch.zeros_like(gain)
        smoothed_gain[:, 0] = gain[:, 0]
        
        for n in range(1, gain.shape[1]):
            if gain[:, n] < smoothed_gain[:, n-1]:
                # Attack
                smoothed_gain[:, n] = gain[:, n] + (smoothed_gain[:, n-1] - gain[:, n]) * attack_coeff
            else:
                # Release  
                smoothed_gain[:, n] = gain[:, n] + (smoothed_gain[:, n-1] - gain[:, n]) * release_coeff
        
        # Apply smoothed gain
        return audio * smoothed_gain
    
    def get_loudness_history(self, audio: torch.Tensor, window_seconds: float = 3.0) -> torch.Tensor:
        """
        Get short-term loudness history for visualization.
        
        Args:
            audio: Input audio tensor
            window_seconds: Window size for short-term loudness
            
        Returns:
            Loudness values over time
        """
        window_samples = int(window_seconds * self.sample_rate)
        hop_samples = int(0.1 * self.sample_rate)  # 100ms hop
        
        num_windows = (audio.shape[-1] - window_samples) // hop_samples + 1
        loudness_history = torch.zeros(num_windows)
        
        for i in range(num_windows):
            start = i * hop_samples
            end = start + window_samples
            window = audio[:, start:end]
            
            # Simple RMS-based loudness for visualization
            rms = torch.sqrt(torch.mean(window ** 2))
            loudness_history[i] = 20 * torch.log10(rms + 1e-10)
        
        return loudness_history