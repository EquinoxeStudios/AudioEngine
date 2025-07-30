import torch
import torchaudio
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging
from dataclasses import dataclass
import warnings

from noise_generators import WhiteNoiseGenerator, PinkNoiseGenerator, BrownNoiseGenerator
from therapeutic_processor import TherapeuticProcessor
from loudness_processor import LoudnessProcessor
from metadata_handler import MetadataHandler
from cuda_accelerator import CUDAAccelerator

warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Configuration for audio generation"""
    sample_rate: int = 48000
    bit_depth: int = 24
    target_lufs: float = -14.0
    true_peak_limit: float = -1.0
    oversampling_factor: int = 4
    therapeutic_eq: bool = True
    fade_duration: float = 5.0
    use_cuda: bool = True


class NoiseGenerator:
    """
    Main audio engine for generating therapeutic noise content.
    Generates continuous, uninterrupted audio streams for YouTube.
    """
    
    def __init__(self, 
                 sample_rate: int = 48000,
                 bit_depth: int = 24,
                 target_lufs: float = -14.0,
                 use_cuda: bool = True,
                 therapeutic_eq: bool = True,
                 fade_duration: float = 5.0,
                 oversampling_factor: int = 4):
        """
        Initialize the noise generator with professional audio standards.
        
        Args:
            sample_rate: Sample rate in Hz (48000 for YouTube)
            bit_depth: Bit depth (24-bit for professional quality)
            target_lufs: Target loudness in LUFS (-14 for YouTube)
            use_cuda: Enable GPU acceleration
            therapeutic_eq: Apply infant-optimized frequency shaping
            fade_duration: Fade in/out duration in seconds
            oversampling_factor: Oversampling for anti-aliasing (4x)
        """
        self.config = AudioConfig(
            sample_rate=sample_rate,
            bit_depth=bit_depth,
            target_lufs=target_lufs,
            use_cuda=use_cuda,
            therapeutic_eq=therapeutic_eq,
            fade_duration=fade_duration,
            oversampling_factor=oversampling_factor
        )
        
        # Initialize CUDA acceleration
        self.cuda = CUDAAccelerator(enabled=use_cuda)
        self.device = self.cuda.device
        
        # Initialize processors
        self.therapeutic_processor = TherapeuticProcessor(
            sample_rate=sample_rate,
            device=self.device
        )
        self.loudness_processor = LoudnessProcessor(
            sample_rate=sample_rate,
            target_lufs=target_lufs,
            true_peak_limit=self.config.true_peak_limit,
            device=self.device
        )
        self.metadata_handler = MetadataHandler()
        
        # Initialize noise generators
        self.white_noise_gen = WhiteNoiseGenerator(sample_rate, device=self.device)
        self.pink_noise_gen = PinkNoiseGenerator(sample_rate, device=self.device)
        self.brown_noise_gen = BrownNoiseGenerator(sample_rate, device=self.device)
        
        logger.info(f"NoiseGenerator initialized - SR: {sample_rate}Hz, CUDA: {use_cuda}")
    
    def generate_white_noise(self, duration_minutes: float) -> torch.Tensor:
        """
        Generate continuous white noise.
        
        Args:
            duration_minutes: Duration in minutes
            
        Returns:
            Stereo audio tensor [2, samples]
        """
        logger.info(f"Generating {duration_minutes} minutes of white noise")
        
        # Calculate total samples needed
        duration_seconds = duration_minutes * 60
        total_samples = int(duration_seconds * self.config.sample_rate)
        
        # Generate continuous white noise
        audio = self.white_noise_gen.generate(total_samples)
        
        # Process audio
        audio = self._process_audio(audio, "white")
        
        return audio
    
    def generate_pink_noise(self, duration_minutes: float) -> torch.Tensor:
        """
        Generate continuous pink noise using Voss-McCartney algorithm.
        
        Args:
            duration_minutes: Duration in minutes
            
        Returns:
            Stereo audio tensor [2, samples]
        """
        logger.info(f"Generating {duration_minutes} minutes of pink noise")
        
        duration_seconds = duration_minutes * 60
        total_samples = int(duration_seconds * self.config.sample_rate)
        
        # Generate continuous pink noise
        audio = self.pink_noise_gen.generate(total_samples)
        
        # Process audio
        audio = self._process_audio(audio, "pink")
        
        return audio
    
    def generate_brown_noise(self, duration_minutes: float) -> torch.Tensor:
        """
        Generate continuous brown noise using integration method.
        
        Args:
            duration_minutes: Duration in minutes
            
        Returns:
            Stereo audio tensor [2, samples]
        """
        logger.info(f"Generating {duration_minutes} minutes of brown noise")
        
        duration_seconds = duration_minutes * 60
        total_samples = int(duration_seconds * self.config.sample_rate)
        
        # Generate continuous brown noise
        audio = self.brown_noise_gen.generate(total_samples)
        
        # Process audio
        audio = self._process_audio(audio, "brown")
        
        return audio
    
    def _process_audio(self, audio: torch.Tensor, noise_type: str) -> torch.Tensor:
        """
        Apply full processing chain to audio.
        
        Args:
            audio: Input audio tensor
            noise_type: Type of noise for metadata
            
        Returns:
            Processed audio tensor
        """
        # Oversample for anti-aliasing
        if self.config.oversampling_factor > 1:
            audio = self._oversample(audio)
        
        # Apply therapeutic EQ if enabled
        if self.config.therapeutic_eq:
            audio = self.therapeutic_processor.process(audio)
        
        # Apply loudness processing
        audio = self.loudness_processor.process(audio)
        
        # Downsample back to target rate
        if self.config.oversampling_factor > 1:
            audio = self._downsample(audio)
        
        # Apply fade in/out
        audio = self._apply_fades(audio)
        
        # Apply dithering for bit depth reduction
        audio = self._apply_dither(audio)
        
        return audio
    
    def _oversample(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply oversampling with anti-aliasing filter"""
        target_rate = self.config.sample_rate * self.config.oversampling_factor
        resampler = torchaudio.transforms.Resample(
            self.config.sample_rate, 
            target_rate,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interpolation",
            beta=14.769656459379492
        ).to(self.device)
        return resampler(audio)
    
    def _downsample(self, audio: torch.Tensor) -> torch.Tensor:
        """Downsample back to target sample rate"""
        source_rate = self.config.sample_rate * self.config.oversampling_factor
        resampler = torchaudio.transforms.Resample(
            source_rate,
            self.config.sample_rate,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interpolation",
            beta=14.769656459379492
        ).to(self.device)
        return resampler(audio)
    
    def _apply_fades(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply fade in/out to prevent clicks"""
        fade_samples = int(self.config.fade_duration * self.config.sample_rate)
        
        # Create fade curves
        fade_in = torch.linspace(0, 1, fade_samples, device=self.device)
        fade_out = torch.linspace(1, 0, fade_samples, device=self.device)
        
        # Apply fades
        audio[:, :fade_samples] *= fade_in
        audio[:, -fade_samples:] *= fade_out
        
        return audio
    
    def _apply_dither(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply shaped dither for bit depth reduction"""
        # Calculate dither amplitude based on bit depth
        dither_amplitude = 1.0 / (2 ** (self.config.bit_depth - 1))
        
        # Generate triangular probability density function (TPDF) dither
        dither = (torch.rand_like(audio, device=self.device) + 
                 torch.rand_like(audio, device=self.device) - 1) * dither_amplitude
        
        # Apply noise shaping (simple first-order)
        shaped_dither = torch.zeros_like(dither)
        shaped_dither[:, 1:] = dither[:, 1:] - 0.5 * dither[:, :-1]
        shaped_dither[:, 0] = dither[:, 0]
        
        return audio + shaped_dither
    
    def export_flac(self, 
                    filename: str, 
                    audio: torch.Tensor,
                    noise_type: str = "unknown",
                    duration_minutes: Optional[float] = None) -> None:
        """
        Export audio to FLAC with metadata.
        
        Args:
            filename: Output filename
            audio: Audio tensor to export
            noise_type: Type of noise for metadata
            duration_minutes: Duration in minutes for metadata
        """
        # Ensure audio is on CPU for export
        audio_cpu = audio.cpu()
        
        # Get actual LUFS measurement
        lufs = self.loudness_processor.measure_lufs(audio)
        
        # Prepare metadata
        metadata = {
            'noise_type': noise_type,
            'duration_minutes': duration_minutes or (audio.shape[1] / self.config.sample_rate / 60),
            'sample_rate': self.config.sample_rate,
            'bit_depth': self.config.bit_depth,
            'measured_lufs': float(lufs),
            'target_lufs': self.config.target_lufs
        }
        
        # Export with metadata
        self.metadata_handler.export_flac(
            filename=filename,
            audio=audio_cpu,
            sample_rate=self.config.sample_rate,
            bit_depth=self.config.bit_depth,
            metadata=metadata
        )
        
        logger.info(f"Exported {filename} - LUFS: {lufs:.1f}, Duration: {metadata['duration_minutes']:.1f} min")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the current device configuration"""
        return self.cuda.get_device_info()