import soundfile as sf
import torch
import numpy as np
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class MetadataHandler:
    """
    Handles FLAC file export with professional metadata tagging.
    Ensures all relevant information is embedded in the audio files.
    """
    
    def __init__(self):
        """Initialize metadata handler"""
        self.standard_tags = {
            'ARTIST': 'AudioEngine Therapeutic Noise Generator',
            'ALBUM': 'Therapeutic Noise Collection',
            'GENRE': 'Therapeutic/Ambient',
            'COPYRIGHT': f'© {datetime.now().year} AudioEngine',
            'SOFTWARE': 'AudioEngine v1.0 - Professional Therapeutic Noise Generator'
        }
        
        logger.info("MetadataHandler initialized")
    
    def export_flac(self,
                    filename: str,
                    audio: torch.Tensor,
                    sample_rate: int,
                    bit_depth: int,
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Export audio to FLAC format with metadata.
        
        Args:
            filename: Output filename
            audio: Audio tensor [channels, samples]
            sample_rate: Sample rate in Hz
            bit_depth: Bit depth (16, 24, or 32)
            metadata: Additional metadata to embed
        """
        try:
            # Convert tensor to numpy array
            audio_np = audio.numpy()
            
            # Ensure audio is in the correct format (samples, channels)
            if audio_np.shape[0] in [1, 2]:  # Mono or stereo
                audio_np = audio_np.T
            
            # Normalize to appropriate range based on bit depth
            if bit_depth == 16:
                # Scale to int16 range
                audio_np = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
                subtype = 'PCM_16'
            elif bit_depth == 24:
                # Scale to int32 range (24-bit stored as 32-bit)
                audio_np = np.clip(audio_np * 8388607, -8388608, 8388607).astype(np.int32)
                subtype = 'PCM_24'
            elif bit_depth == 32:
                # Keep as float32
                audio_np = audio_np.astype(np.float32)
                subtype = 'PCM_32'
            else:
                raise ValueError(f"Unsupported bit depth: {bit_depth}")
            
            # Prepare metadata
            flac_metadata = self._prepare_metadata(metadata, sample_rate, audio.shape[1])
            
            # Write FLAC file
            sf.write(
                filename,
                audio_np,
                sample_rate,
                subtype=subtype,
                format='FLAC'
            )
            
            # Add metadata using soundfile
            # Note: soundfile has limited metadata support
            # For full metadata support, consider using mutagen
            self._write_extended_metadata(filename, flac_metadata)
            
            logger.info(f"Exported FLAC: {filename} - {bit_depth}-bit, {sample_rate}Hz")
            
        except Exception as e:
            logger.error(f"Failed to export FLAC: {str(e)}")
            raise
    
    def _prepare_metadata(self, 
                         custom_metadata: Optional[Dict[str, Any]], 
                         sample_rate: int,
                         num_samples: int) -> Dict[str, str]:
        """
        Prepare metadata for FLAC file.
        
        Args:
            custom_metadata: Custom metadata from user
            sample_rate: Sample rate
            num_samples: Number of audio samples
            
        Returns:
            Complete metadata dictionary
        """
        # Start with standard tags
        metadata = self.standard_tags.copy()
        
        # Add technical information
        duration_seconds = num_samples / sample_rate
        duration_str = self._format_duration(duration_seconds)
        
        metadata.update({
            'DATE': datetime.now().strftime('%Y-%m-%d'),
            'DURATION': duration_str,
            'SAMPLE_RATE': str(sample_rate),
            'ENCODER': 'AudioEngine FLAC Encoder'
        })
        
        # Add custom metadata if provided
        if custom_metadata:
            # Extract specific fields
            if 'noise_type' in custom_metadata:
                noise_type = custom_metadata['noise_type'].capitalize()
                metadata['TITLE'] = f'{noise_type} Noise - {duration_str}'
                metadata['DESCRIPTION'] = f'Therapeutic {noise_type} noise for infant sleep and comfort'
            
            if 'measured_lufs' in custom_metadata:
                metadata['REPLAYGAIN_TRACK_GAIN'] = f"{custom_metadata['measured_lufs']:.1f} LUFS"
            
            if 'target_lufs' in custom_metadata:
                metadata['REPLAYGAIN_REFERENCE_LOUDNESS'] = f"{custom_metadata['target_lufs']:.1f} LUFS"
            
            # Add all custom metadata as comments
            technical_info = {
                'bit_depth': custom_metadata.get('bit_depth', 'unknown'),
                'therapeutic_eq': 'enabled',
                'youtube_optimized': 'true',
                'continuous_generation': 'true',
                'algorithm': self._get_algorithm_name(custom_metadata.get('noise_type', 'unknown'))
            }
            
            metadata['COMMENT'] = json.dumps({**custom_metadata, **technical_info}, indent=2)
        
        return metadata
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def _get_algorithm_name(self, noise_type: str) -> str:
        """Get algorithm name for noise type"""
        algorithms = {
            'white': 'Mersenne Twister PRNG with Gaussian distribution',
            'pink': 'Voss-McCartney algorithm',
            'brown': 'Integration-based with 1/f² spectral density'
        }
        return algorithms.get(noise_type.lower(), 'Custom algorithm')
    
    def _write_extended_metadata(self, filename: str, metadata: Dict[str, str]) -> None:
        """
        Write extended metadata to FLAC file.
        
        Note: This is a placeholder. For full metadata support,
        you would use a library like mutagen.
        """
        try:
            # Try to use mutagen if available
            from mutagen.flac import FLAC
            
            audio_file = FLAC(filename)
            
            # Clear existing tags
            audio_file.clear()
            
            # Add all metadata
            for key, value in metadata.items():
                audio_file[key] = str(value)
            
            # Save metadata
            audio_file.save()
            
            logger.info("Extended metadata written successfully")
            
        except ImportError:
            # Mutagen not available
            logger.warning("mutagen not installed - limited metadata support")
            # soundfile only supports basic metadata
            # In production, you'd want to ensure mutagen is installed
        except Exception as e:
            logger.warning(f"Could not write extended metadata: {str(e)}")
    
    def read_metadata(self, filename: str) -> Dict[str, Any]:
        """
        Read metadata from FLAC file.
        
        Args:
            filename: FLAC file to read
            
        Returns:
            Dictionary of metadata
        """
        try:
            # Try mutagen first
            from mutagen.flac import FLAC
            
            audio_file = FLAC(filename)
            metadata = dict(audio_file)
            
            # Parse technical metadata from COMMENT if present
            if 'COMMENT' in metadata:
                try:
                    technical_data = json.loads(metadata['COMMENT'][0])
                    metadata.update(technical_data)
                except:
                    pass
            
            return metadata
            
        except ImportError:
            # Fall back to soundfile
            info = sf.info(filename)
            return {
                'duration': info.duration,
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'format': info.format,
                'subtype': info.subtype
            }
        except Exception as e:
            logger.error(f"Failed to read metadata: {str(e)}")
            return {}
    
    def create_cue_sheet(self, 
                        filename: str,
                        tracks: list,
                        album_title: str = "Therapeutic Noise Collection") -> None:
        """
        Create a CUE sheet for multiple noise tracks.
        
        Args:
            filename: Output CUE filename
            tracks: List of track info dictionaries
            album_title: Album title for CUE sheet
        """
        cue_content = f"""REM GENERATOR "AudioEngine v1.0"
REM DATE "{datetime.now().strftime('%Y-%m-%d')}"
PERFORMER "AudioEngine"
TITLE "{album_title}"
FILE "compilation.flac" WAVE
"""
        
        for i, track in enumerate(tracks, 1):
            cue_content += f"""  TRACK {i:02d} AUDIO
    TITLE "{track.get('title', f'Track {i}')}"
    PERFORMER "AudioEngine"
    INDEX 01 {track.get('start_time', '00:00:00')}
"""
        
        with open(filename, 'w') as f:
            f.write(cue_content)
        
        logger.info(f"Created CUE sheet: {filename}")