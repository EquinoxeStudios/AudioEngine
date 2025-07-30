"""
Audio Engine: Professional Therapeutic Noise Generator

A high-performance Python audio engine for generating studio-quality
therapeutic noise content optimized for YouTube and infant comfort.
"""

from .core.noise_generator import NoiseGenerator
from .processors.therapeutic_processor import TherapeuticProcessor
from .processors.loudness_processor import LoudnessProcessor
from .utils.metadata_handler import MetadataHandler
from .utils.cuda_accelerator import CUDAAccelerator

__version__ = "1.0.0"
__author__ = "Audio Engine Team"
__email__ = "contact@audioengine.com"

__all__ = [
    "NoiseGenerator",
    "TherapeuticProcessor", 
    "LoudnessProcessor",
    "MetadataHandler",
    "CUDAAccelerator"
]