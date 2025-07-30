"""
NoiseGenerator - Core Audio Engine

Main class for generating high-quality therapeutic noise with professional
audio standards and YouTube optimization.
"""

import torch
import torchaudio
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple

from ..algorithms.white_noise import WhiteNoiseAlgorithm
from ..algorithms.pink_noise import PinkNoiseAlgorithm
from ..algorithms.brown_noise import BrownNoiseAlgorithm
from ..processors.therapeutic_processor import TherapeuticProcessor
from ..processors.loudness_processor import LoudnessProcessor
from ..utils.metadata_handler import MetadataHandler
from ..utils.cuda_accelerator import CUDAAccelerator


class NoiseGenerator:
    """
    Professional therapeutic noise generator with YouTube optimization.
    
    Features:
    - Studio-quality noise generation (white, pink, brown)
    - Therapeutic frequency shaping for infant comfort
    - YouTube LUFS compliance (-14 LUFS)
    - CUDA acceleration support
    - Professional audio standards (48kHz/24-bit)
    """
    
    def __init__(
        self,
        sample_rate: int = 48000,
        bit_depth: int = 24,
        target_lufs: float = -14.0,
        use_cuda: bool = True,
        therapeutic_eq: bool = True,
        fade_duration: float = 5.0,
        oversampling_factor: int = 1  # Disabled by default for performance
    ):
        """
        Initialize the NoiseGenerator.
        
        Args:
            sample_rate: Target sample rate in Hz (48000 for YouTube)
            bit_depth: Target bit depth (24 for professional quality)
            target_lufs: Target LUFS level (-14 for YouTube)
            use_cuda: Enable CUDA acceleration if available
            therapeutic_eq: Apply infant-optimized EQ curve
            fade_duration: Fade in/out duration in seconds
            oversampling_factor: Oversampling rate for anti-aliasing
        """
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth
        self.target_lufs = target_lufs
        self.fade_duration = fade_duration
        self.oversampling_factor = oversampling_factor
        
        # Initialize CUDA acceleration
        self.cuda_accelerator = CUDAAccelerator(use_cuda)
        self.device = self.cuda_accelerator.device
        
        # Initialize noise algorithms
        self.white_algo = WhiteNoiseAlgorithm(sample_rate, use_cuda)
        self.pink_algo = PinkNoiseAlgorithm(sample_rate, use_cuda)
        self.brown_algo = BrownNoiseAlgorithm(sample_rate, use_cuda)
        
        # Initialize processors
        self.therapeutic_processor = TherapeuticProcessor(
            sample_rate=sample_rate,
            use_cuda=use_cuda,
            enabled=therapeutic_eq
        )
        
        self.loudness_processor = LoudnessProcessor(
            sample_rate=sample_rate,
            target_lufs=target_lufs,
            use_cuda=use_cuda
        )
        
        # Initialize metadata handler
        self.metadata_handler = MetadataHandler()
        
        # Generation statistics
        self.stats = {
            "total_generated": 0,
            "last_generation_time": None,
            "cuda_enabled": self.cuda_accelerator.is_available()
        }
    
    def generate_white_noise(
        self, 
        duration_minutes: float, 
        channels: int = 2
    ) -> torch.Tensor:
        """
        Generate therapeutic white noise.
        
        Args:
            duration_minutes: Duration in minutes
            channels: Number of output channels (1=mono, 2=stereo)
            
        Returns:
            torch.Tensor: Generated white noise audio [channels, samples]
        """
        return self._generate_noise("white", duration_minutes, channels)
    
    def generate_pink_noise(
        self, 
        duration_minutes: float, 
        channels: int = 2
    ) -> torch.Tensor:
        """
        Generate therapeutic pink noise.
        
        Args:
            duration_minutes: Duration in minutes
            channels: Number of output channels (1=mono, 2=stereo)
            
        Returns:
            torch.Tensor: Generated pink noise audio [channels, samples]
        """
        return self._generate_noise("pink", duration_minutes, channels)
    
    def generate_brown_noise(
        self, 
        duration_minutes: float, 
        channels: int = 2
    ) -> torch.Tensor:
        """
        Generate therapeutic brown noise.
        
        Args:
            duration_minutes: Duration in minutes
            channels: Number of output channels (1=mono, 2=stereo)
            
        Returns:
            torch.Tensor: Generated brown noise audio [channels, samples]
        """
        return self._generate_noise("brown", duration_minutes, channels)
    
    def _generate_noise(
        self, 
        noise_type: str, 
        duration_minutes: float, 
        channels: int
    ) -> torch.Tensor:
        """
        Internal noise generation with full processing pipeline.
        
        Args:
            noise_type: Type of noise ("white", "pink", "brown")
            duration_minutes: Duration in minutes
            channels: Number of channels
            
        Returns:
            torch.Tensor: Processed noise audio
        """
        import time
        start_time = time.time()
        
        # Calculate sample count
        duration_samples = int(duration_minutes * 60 * self.sample_rate)
        
        # Generate raw noise with oversampling for anti-aliasing
        oversampled_samples = duration_samples * self.oversampling_factor
        
        if noise_type == "white":
            raw_audio = self.white_algo.generate(oversampled_samples, channels)
        elif noise_type == "pink":
            raw_audio = self.pink_algo.generate(oversampled_samples, channels)
        elif noise_type == "brown":
            raw_audio = self.brown_algo.generate(oversampled_samples, channels)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Downsample with anti-aliasing
        processed_audio = self._apply_anti_aliasing(raw_audio, duration_samples)
        
        # Apply therapeutic processing
        processed_audio = self.therapeutic_processor.process(processed_audio)
        
        # Apply fade in/out
        processed_audio = self._apply_fades(processed_audio)
        
        # Apply loudness processing for YouTube compliance
        processed_audio = self.loudness_processor.process(processed_audio)
        
        # Remove DC offset
        processed_audio = self._remove_dc_offset(processed_audio)
        
        # Final quality checks
        self._perform_quality_checks(processed_audio, noise_type)
        
        # Update statistics
        generation_time = time.time() - start_time
        self.stats["total_generated"] += 1
        self.stats["last_generation_time"] = generation_time
        
        return processed_audio
    
    def _apply_anti_aliasing(
        self, 
        oversampled_audio: torch.Tensor, 
        target_samples: int
    ) -> torch.Tensor:
        """
        Apply anti-aliasing through proper downsampling.
        
        Args:
            oversampled_audio: Oversampled input audio
            target_samples: Target number of samples
            
        Returns:
            torch.Tensor: Downsampled audio with anti-aliasing
        """
        # Use torchaudio's resampling with anti-aliasing
        if self.oversampling_factor == 1:
            return oversampled_audio
        
        target_rate = self.sample_rate
        source_rate = self.sample_rate * self.oversampling_factor
        
        # Apply anti-aliasing filter and downsample
        resampler = torchaudio.transforms.Resample(
            orig_freq=source_rate,
            new_freq=target_rate,
            resampling_method='sinc_interp_hann'
        ).to(self.device)
        
        downsampled = resampler(oversampled_audio)
        
        # Ensure exact sample count
        if downsampled.shape[1] > target_samples:
            downsampled = downsampled[:, :target_samples]
        elif downsampled.shape[1] < target_samples:
            # Zero-pad if needed
            padding = target_samples - downsampled.shape[1]
            downsampled = torch.nn.functional.pad(downsampled, (0, padding))
        
        return downsampled
    
    def _apply_fades(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply smooth fade-in and fade-out curves.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            torch.Tensor: Audio with fades applied
        """
        channels, samples = audio.shape
        fade_samples = int(self.fade_duration * self.sample_rate)
        
        if fade_samples * 2 >= samples:
            # Audio too short for fades
            return audio
        
        # Create fade curves (cosine curves for smooth transitions)
        fade_in = torch.sin(torch.linspace(0, np.pi/2, fade_samples, device=self.device))**2
        fade_out = torch.cos(torch.linspace(0, np.pi/2, fade_samples, device=self.device))**2
        
        # Apply fades
        audio[:, :fade_samples] *= fade_in
        audio[:, -fade_samples:] *= fade_out
        
        return audio
    
    def _remove_dc_offset(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Remove DC offset using high-pass filter at 1-2Hz.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            torch.Tensor: DC-corrected audio
        """
        # Simple DC removal - subtract mean from each channel
        # This is much faster than the iterative high-pass filter
        # and sufficient for DC offset removal
        for ch in range(audio.shape[0]):
            audio[ch] = audio[ch] - torch.mean(audio[ch])
        
        return audio
    
    def _perform_quality_checks(self, audio: torch.Tensor, noise_type: str) -> None:
        """
        Perform quality checks on generated audio.
        
        Args:
            audio: Generated audio to check
            noise_type: Type of noise for specific checks
        """
        # Check for clipping
        max_amplitude = torch.max(torch.abs(audio)).item()
        if max_amplitude > 0.95:
            print(f"Warning: Possible clipping detected (max: {max_amplitude:.3f})")
        
        # Check for DC offset
        dc_offset = torch.mean(audio).item()
        if abs(dc_offset) > 0.01:
            print(f"Warning: DC offset detected: {dc_offset:.4f}")
        
        # Check dynamic range
        rms = torch.sqrt(torch.mean(audio**2)).item()
        if rms < 0.001:
            print(f"Warning: Very low RMS level: {rms:.6f}")
    
    def export_flac(
        self, 
        filename: Union[str, Path], 
        audio: torch.Tensor,
        noise_type: str = "unknown",
        duration_minutes: Optional[float] = None
    ) -> None:
        """
        Export audio to FLAC format with metadata.
        
        Args:
            filename: Output filename
            audio: Audio tensor to export
            noise_type: Type of noise for metadata
            duration_minutes: Duration for metadata
        """
        # Convert to numpy and ensure proper format
        audio_np = audio.cpu().numpy()
        
        # Convert to target bit depth
        if self.bit_depth == 16:
            audio_np = (audio_np * 32767).astype(np.int16)
        elif self.bit_depth == 24:
            audio_np = (audio_np * 8388607).astype(np.int32)
        else:
            audio_np = audio_np.astype(np.float32)
        
        # Write FLAC file
        sf.write(
            filename,
            audio_np.T,  # soundfile expects [samples, channels]
            self.sample_rate,
            subtype='PCM_24' if self.bit_depth == 24 else 'PCM_16'
        )
        
        # Add metadata
        if duration_minutes:
            metadata = {
                "noise_type": noise_type,
                "duration_minutes": duration_minutes,
                "lufs_target": self.target_lufs,
                "sample_rate": self.sample_rate,
                "bit_depth": self.bit_depth,
                "therapeutic": True,
                "youtube_optimized": True
            }
            
            self.metadata_handler.embed_metadata(filename, metadata)
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get generation statistics and system info.
        
        Returns:
            dict: Generation statistics
        """
        return {
            **self.stats,
            "cuda_device": str(self.device),
            "sample_rate": self.sample_rate,
            "bit_depth": self.bit_depth,
            "target_lufs": self.target_lufs,
            "therapeutic_processing": self.therapeutic_processor.enabled
        }
    
    def cleanup(self) -> None:
        """Clean up resources and reset states."""
        self.brown_algo.reset_state()
        torch.cuda.empty_cache() if self.cuda_accelerator.is_available() else None