import torch
import logging
from typing import Optional, Dict, Any
import psutil
import numpy as np

logger = logging.getLogger(__name__)


class CUDAAccelerator:
    """
    CUDA acceleration manager for GPU-optimized audio processing.
    Optimized for Google Colab environments.
    """
    
    def __init__(self, enabled: bool = True, memory_fraction: float = 0.9):
        """
        Initialize CUDA accelerator.
        
        Args:
            enabled: Whether to enable CUDA acceleration
            memory_fraction: Fraction of GPU memory to use (0-1)
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.memory_fraction = memory_fraction
        
        if self.enabled:
            # Set up CUDA device
            self.device = torch.device('cuda')
            
            # Configure memory allocation
            self._configure_memory()
            
            # Log GPU information
            self._log_gpu_info()
        else:
            self.device = torch.device('cpu')
            if enabled and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available - using CPU")
            else:
                logger.info("CUDA disabled - using CPU")
    
    def _configure_memory(self):
        """Configure CUDA memory allocation for optimal performance"""
        try:
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
            
            # Enable memory caching for better performance
            torch.cuda.empty_cache()
            
            # Set cudnn benchmarking for optimal convolution algorithms
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
            
            logger.info(f"CUDA memory configured - using {self.memory_fraction*100:.0f}% of available memory")
            
        except Exception as e:
            logger.warning(f"Could not configure CUDA memory: {str(e)}")
    
    def _log_gpu_info(self):
        """Log GPU information for debugging"""
        try:
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            
            # Memory info
            total_memory = torch.cuda.get_device_properties(current_device).total_memory
            allocated_memory = torch.cuda.memory_allocated(current_device)
            cached_memory = torch.cuda.memory_reserved(current_device)
            
            logger.info(f"GPU initialized: {gpu_name}")
            logger.info(f"Total GPU memory: {total_memory / 1e9:.2f} GB")
            logger.info(f"Allocated: {allocated_memory / 1e9:.2f} GB, Cached: {cached_memory / 1e9:.2f} GB")
            
        except Exception as e:
            logger.warning(f"Could not get GPU info: {str(e)}")
    
    def optimize_batch_size(self, 
                          sample_rate: int, 
                          duration_seconds: float,
                          channels: int = 2) -> int:
        """
        Calculate optimal batch size for processing based on available memory.
        
        Args:
            sample_rate: Audio sample rate
            duration_seconds: Duration of audio to process
            channels: Number of audio channels
            
        Returns:
            Optimal batch size in samples
        """
        if not self.enabled:
            # For CPU, use smaller batches
            return min(sample_rate * 10, int(sample_rate * duration_seconds))
        
        try:
            # Get available memory
            free_memory = torch.cuda.mem_get_info()[0]
            
            # Estimate memory per sample (float32)
            bytes_per_sample = 4 * channels  # 4 bytes per float32
            
            # Account for processing overhead (filters, FFTs, etc.)
            overhead_factor = 4  # Conservative estimate
            
            # Calculate maximum samples that fit in memory
            max_samples = int(free_memory / (bytes_per_sample * overhead_factor))
            
            # Limit to reasonable batch sizes
            total_samples = int(sample_rate * duration_seconds)
            
            # Choose batch size (power of 2 for FFT efficiency)
            batch_size = min(max_samples, total_samples)
            batch_size = 2 ** int(np.log2(batch_size))
            
            # Ensure minimum batch size
            batch_size = max(batch_size, sample_rate)  # At least 1 second
            
            logger.info(f"Optimal batch size: {batch_size} samples ({batch_size/sample_rate:.1f} seconds)")
            
            return batch_size
            
        except Exception as e:
            logger.warning(f"Could not optimize batch size: {str(e)}")
            return sample_rate * 10  # Default to 10 seconds
    
    def process_in_batches(self, 
                          audio: torch.Tensor,
                          process_fn: callable,
                          batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Process audio in memory-efficient batches.
        
        Args:
            audio: Input audio tensor
            process_fn: Function to apply to each batch
            batch_size: Batch size in samples (auto-calculated if None)
            
        Returns:
            Processed audio tensor
        """
        if batch_size is None:
            # Auto-calculate based on audio length
            batch_size = self.optimize_batch_size(
                sample_rate=48000,  # Assume 48kHz
                duration_seconds=audio.shape[-1] / 48000,
                channels=audio.shape[0]
            )
        
        # Process in batches
        num_samples = audio.shape[-1]
        
        if num_samples <= batch_size:
            # Process entire audio at once
            return process_fn(audio)
        
        # Process in chunks with overlap for continuity
        overlap = int(batch_size * 0.1)  # 10% overlap
        hop = batch_size - overlap
        
        output = torch.zeros_like(audio)
        
        for start in range(0, num_samples - overlap, hop):
            end = min(start + batch_size, num_samples)
            
            # Process batch
            batch_input = audio[:, start:end]
            batch_output = process_fn(batch_input)
            
            # Handle overlap with crossfade
            if start > 0 and overlap > 0:
                # Crossfade with previous chunk
                fade_in = torch.linspace(0, 1, overlap, device=self.device)
                fade_out = torch.linspace(1, 0, overlap, device=self.device)
                
                output[:, start:start+overlap] = (
                    output[:, start:start+overlap] * fade_out +
                    batch_output[:, :overlap] * fade_in
                )
                output[:, start+overlap:end] = batch_output[:, overlap:end-start]
            else:
                output[:, start:end] = batch_output
            
            # Clear cache periodically
            if self.enabled and start % (hop * 10) == 0:
                torch.cuda.empty_cache()
        
        return output
    
    def optimize_fft(self, fft_size: int) -> int:
        """
        Optimize FFT size for CUDA performance.
        
        Args:
            fft_size: Desired FFT size
            
        Returns:
            Optimized FFT size
        """
        if not self.enabled:
            return fft_size
        
        # CUDA FFT performs best with powers of 2
        # and certain small prime factors
        optimal_size = 2 ** int(np.ceil(np.log2(fft_size)))
        
        # Check if slightly larger size would be more efficient
        good_sizes = [optimal_size]
        for factor in [3, 5, 7]:
            candidate = optimal_size * factor // 4
            if candidate > fft_size:
                good_sizes.append(candidate)
        
        # Choose the smallest good size
        optimal_size = min(good_sizes)
        
        return optimal_size
    
    def pin_memory(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Pin tensor memory for faster GPU transfer.
        
        Args:
            tensor: Tensor to pin
            
        Returns:
            Pinned tensor
        """
        if self.enabled and not tensor.is_cuda:
            return tensor.pin_memory()
        return tensor
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get current device information"""
        info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_enabled': self.enabled,
            'device': str(self.device),
        }
        
        if self.enabled:
            try:
                current_device = torch.cuda.current_device()
                info.update({
                    'gpu_name': torch.cuda.get_device_name(current_device),
                    'gpu_memory_total': torch.cuda.get_device_properties(current_device).total_memory,
                    'gpu_memory_allocated': torch.cuda.memory_allocated(current_device),
                    'gpu_memory_cached': torch.cuda.memory_reserved(current_device),
                })
            except:
                pass
        
        # Add CPU info
        info.update({
            'cpu_count': psutil.cpu_count(),
            'ram_total': psutil.virtual_memory().total,
            'ram_available': psutil.virtual_memory().available,
        })
        
        return info
    
    def synchronize(self):
        """Synchronize CUDA operations"""
        if self.enabled:
            torch.cuda.synchronize()
    
    def clear_cache(self):
        """Clear CUDA cache to free memory"""
        if self.enabled:
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup"""
        self.clear_cache()