"""
CUDAAccelerator - GPU Optimization for Audio Processing

Provides CUDA acceleration capabilities for high-performance audio
generation and processing in Google Colab and other GPU environments.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import psutil
import gc


class CUDAAccelerator:
    """
    CUDA acceleration manager for high-performance audio processing.
    
    Features:
    - Automatic GPU detection and optimization
    - Memory management for large audio files
    - Batch processing capabilities
    - Performance monitoring and profiling
    - Google Colab optimization
    """
    
    def __init__(self, use_cuda: bool = True):
        """
        Initialize CUDA accelerator.
        
        Args:
            use_cuda: Enable CUDA if available
        """
        self.use_cuda = use_cuda
        self.device = self._initialize_device()
        self.memory_stats = {}
        self.performance_stats = {
            "operations_count": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "gpu_utilization": []
        }
        
        # Google Colab specific optimizations
        self.is_colab = self._detect_colab_environment()
        self.colab_optimizations_enabled = False
        
        if self.is_colab and self.is_available():
            self._apply_colab_optimizations()
    
    def _initialize_device(self) -> torch.device:
        """Initialize and configure the compute device."""
        if self.use_cuda and torch.cuda.is_available():
            device = torch.device('cuda')
            
            # Set optimal CUDA settings
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.enabled = True
            
            # Enable mixed precision if supported
            if hasattr(torch.cuda.amp, 'autocast'):
                self.mixed_precision_available = True
            else:
                self.mixed_precision_available = False
                
            # Set memory fraction to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
                
        else:
            device = torch.device('cpu')
            self.mixed_precision_available = False
        
        return device
    
    def _detect_colab_environment(self) -> bool:
        """Detect if running in Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def _apply_colab_optimizations(self) -> None:
        """Apply Google Colab specific optimizations."""
        if not self.is_available():
            return
        
        try:
            # Optimize for T4/V100 GPUs common in Colab
            gpu_name = torch.cuda.get_device_name(0)
            
            if "T4" in gpu_name:
                # T4-specific optimizations
                torch.backends.cudnn.benchmark = True
                self.batch_size_multiplier = 1.2
                self.memory_efficiency_mode = True
                
            elif "V100" in gpu_name:
                # V100-specific optimizations
                torch.backends.cudnn.benchmark = True
                self.batch_size_multiplier = 1.5
                self.memory_efficiency_mode = False
                
            else:
                # General optimizations
                self.batch_size_multiplier = 1.0
                self.memory_efficiency_mode = True
            
            # Enable gradient checkpointing for memory efficiency
            self.gradient_checkpointing = True
            self.colab_optimizations_enabled = True
            
        except Exception as e:
            print(f"Warning: Could not apply Colab optimizations: {e}")
    
    def is_available(self) -> bool:
        """Check if CUDA acceleration is available."""
        return self.device.type == 'cuda'
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information."""
        info = {
            "device_type": self.device.type,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "mixed_precision_available": self.mixed_precision_available,
            "colab_environment": self.is_colab,
            "colab_optimizations": self.colab_optimizations_enabled
        }
        
        if self.is_available():
            current_device = torch.cuda.current_device()
            info.update({
                "device_name": torch.cuda.get_device_name(current_device),
                "device_capability": torch.cuda.get_device_capability(current_device),
                "total_memory_gb": torch.cuda.get_device_properties(current_device).total_memory / 1e9,
                "memory_allocated_gb": torch.cuda.memory_allocated(current_device) / 1e9,
                "memory_reserved_gb": torch.cuda.memory_reserved(current_device) / 1e9,
                "memory_free_gb": (torch.cuda.get_device_properties(current_device).total_memory - 
                                  torch.cuda.memory_allocated(current_device)) / 1e9
            })
        
        return info
    
    def optimize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Optimize tensor for the current device.
        
        Args:
            tensor: Input tensor to optimize
            
        Returns:
            torch.Tensor: Optimized tensor on target device
        """
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        
        # Move to device
        tensor = tensor.to(self.device)
        
        # Optimize memory layout
        if tensor.is_contiguous():
            return tensor
        else:
            return tensor.contiguous()
    
    def process_in_chunks(
        self, 
        audio_data: torch.Tensor, 
        processing_func, 
        chunk_size: Optional[int] = None,
        overlap: int = 0
    ) -> torch.Tensor:
        """
        Process large audio data in memory-efficient chunks.
        
        Args:
            audio_data: Large audio tensor [channels, samples]
            processing_func: Function to apply to each chunk
            chunk_size: Size of each chunk (auto-calculated if None)
            overlap: Overlap between chunks in samples
            
        Returns:
            torch.Tensor: Processed audio data
        """
        channels, total_samples = audio_data.shape
        
        # Auto-calculate chunk size based on available memory
        if chunk_size is None:
            chunk_size = self._calculate_optimal_chunk_size(audio_data)
        
        # Process in chunks with overlap
        processed_chunks = []
        
        for start in range(0, total_samples, chunk_size - overlap):
            end = min(start + chunk_size, total_samples)
            chunk = audio_data[:, start:end]
            
            # Process chunk
            processed_chunk = processing_func(chunk)
            
            # Handle overlap blending
            if overlap > 0 and len(processed_chunks) > 0:
                # Crossfade overlap region
                crossfade_samples = min(overlap, processed_chunk.shape[1])
                fade_out = torch.linspace(1, 0, crossfade_samples, device=self.device)
                fade_in = torch.linspace(0, 1, crossfade_samples, device=self.device)
                
                # Apply crossfade to overlap region
                processed_chunks[-1][:, -crossfade_samples:] *= fade_out
                processed_chunk[:, :crossfade_samples] *= fade_in
                processed_chunks[-1][:, -crossfade_samples:] += processed_chunk[:, :crossfade_samples]
                
                # Append non-overlapping part
                processed_chunks.append(processed_chunk[:, crossfade_samples:])
            else:
                processed_chunks.append(processed_chunk)
            
            # Clear GPU memory periodically
            if self.is_available() and len(processed_chunks) % 10 == 0:
                torch.cuda.empty_cache()
        
        # Concatenate all processed chunks
        result = torch.cat(processed_chunks, dim=1)
        
        # Final memory cleanup
        self.cleanup_memory()
        
        return result
    
    def _calculate_optimal_chunk_size(self, audio_data: torch.Tensor) -> int:
        """Calculate optimal chunk size based on available memory."""
        if not self.is_available():
            # CPU processing - use larger chunks
            return min(audio_data.shape[1], 5 * 48000)  # 5 seconds max
        
        # GPU processing - calculate based on available memory
        device_info = self.get_device_info()
        available_memory_gb = device_info.get("memory_free_gb", 1.0)
        
        # Conservative estimate: use 50% of available memory
        # Each float32 sample takes 4 bytes
        samples_per_gb = 1e9 / (4 * audio_data.shape[0])  # Account for channels
        max_samples = int(available_memory_gb * samples_per_gb * 0.5)
        
        # Ensure minimum chunk size and alignment
        chunk_size = max(48000, min(max_samples, audio_data.shape[1]))  # At least 1 second
        
        # Align to sample rate for clean processing
        chunk_size = (chunk_size // 48000) * 48000
        
        return chunk_size
    
    def batch_process_multiple(
        self, 
        audio_list: List[torch.Tensor], 
        processing_func,
        batch_size: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        Process multiple audio files in batches for efficiency.
        
        Args:
            audio_list: List of audio tensors to process
            processing_func: Function to apply to each audio tensor
            batch_size: Batch size (auto-calculated if None)
            
        Returns:
            List[torch.Tensor]: List of processed audio tensors
        """
        if batch_size is None:
            batch_size = self._calculate_optimal_batch_size(audio_list)
        
        processed_results = []
        
        for i in range(0, len(audio_list), batch_size):
            batch = audio_list[i:i + batch_size]
            
            # Process batch
            batch_results = []
            for audio in batch:
                result = processing_func(audio)
                batch_results.append(result)
            
            processed_results.extend(batch_results)
            
            # Memory cleanup after each batch
            self.cleanup_memory()
        
        return processed_results
    
    def _calculate_optimal_batch_size(self, audio_list: List[torch.Tensor]) -> int:
        """Calculate optimal batch size for multiple audio processing."""
        if not audio_list:
            return 1
        
        # Estimate memory usage per item
        sample_audio = audio_list[0]
        memory_per_item_gb = (sample_audio.numel() * 4) / 1e9  # 4 bytes per float32
        
        if not self.is_available():
            # CPU processing - smaller batches
            return max(1, min(4, len(audio_list)))
        
        # GPU processing
        device_info = self.get_device_info()
        available_memory_gb = device_info.get("memory_free_gb", 1.0)
        
        # Use 60% of available memory for batch processing
        max_batch_size = int((available_memory_gb * 0.6) / memory_per_item_gb)
        
        return max(1, min(max_batch_size, len(audio_list)))
    
    def profile_operation(self, operation_func, *args, **kwargs):
        """
        Profile an operation for performance analysis.
        
        Args:
            operation_func: Function to profile
            *args, **kwargs: Arguments for the function
            
        Returns:
            tuple: (result, profiling_info)
        """
        import time
        
        # Pre-operation memory state
        if self.is_available():
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated()
        
        start_time = time.time()
        
        # Execute operation
        result = operation_func(*args, **kwargs)
        
        # Post-operation measurements
        if self.is_available():
            torch.cuda.synchronize()
            memory_after = torch.cuda.memory_allocated()
            memory_peak = torch.cuda.max_memory_allocated()
        else:
            memory_after = memory_before = memory_peak = 0
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Update performance statistics
        self.performance_stats["operations_count"] += 1
        self.performance_stats["total_processing_time"] += processing_time
        self.performance_stats["average_processing_time"] = (
            self.performance_stats["total_processing_time"] / 
            self.performance_stats["operations_count"]
        )
        
        profiling_info = {
            "processing_time_seconds": processing_time,
            "memory_used_mb": (memory_after - memory_before) / 1e6,
            "memory_peak_mb": memory_peak / 1e6,
            "device_type": self.device.type,
            "cuda_available": self.is_available()
        }
        
        return result, profiling_info
    
    def cleanup_memory(self) -> None:
        """Clean up GPU/CPU memory."""
        # Python garbage collection
        gc.collect()
        
        # GPU memory cleanup
        if self.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = {}
        
        # System memory
        system_memory = psutil.virtual_memory()
        memory_info["system_memory_used_gb"] = system_memory.used / 1e9
        memory_info["system_memory_available_gb"] = system_memory.available / 1e9
        memory_info["system_memory_percent"] = system_memory.percent
        
        # GPU memory (if available)
        if self.is_available():
            memory_info["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            memory_info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
            memory_info["gpu_memory_max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
            
            # Calculate GPU memory usage percentage
            device_props = torch.cuda.get_device_properties(0)
            total_memory = device_props.total_memory
            allocated_memory = torch.cuda.memory_allocated()
            memory_info["gpu_memory_percent"] = (allocated_memory / total_memory) * 100
        
        return memory_info
    
    def optimize_for_youtube_generation(self) -> Dict[str, Any]:
        """
        Apply optimizations specifically for YouTube audio generation.
        
        Returns:
            dict: Applied optimization settings
        """
        optimizations = {
            "batch_processing": True,
            "memory_efficient_mode": True,
            "chunk_processing": True,
            "mixed_precision": self.mixed_precision_available,
            "cuda_optimization": self.is_available()
        }
        
        if self.is_available():
            # Set optimal settings for long-form audio generation
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Optimize memory allocation
            torch.cuda.set_per_process_memory_fraction(0.85)
            
            optimizations["tf32_enabled"] = True
            optimizations["cudnn_benchmark"] = True
        
        return optimizations
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            "device_info": self.get_device_info(),
            "performance_stats": self.performance_stats.copy(),
            "memory_usage": self.get_memory_usage(),
            "optimization_status": {
                "colab_optimized": self.colab_optimizations_enabled,
                "mixed_precision": self.mixed_precision_available,
                "cuda_enabled": self.is_available(),
                "memory_efficient": getattr(self, 'memory_efficiency_mode', False)
            }
        }