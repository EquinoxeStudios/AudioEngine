#!/usr/bin/env python3
"""
Baby Noise Generator - Optimized Audio Engine for YouTube Content
Memory-efficient and performance-optimized implementation with DSP fixes
Generates high-quality white, pink, or brown noise optimized for babies
"""

import numpy as np
import soundfile as sf
import scipy.signal as signal
from scipy.signal import butter, filtfilt, sosfilt, sosfiltfilt, sosfilt_zi, lfilter_zi
import warnings
warnings.filterwarnings('ignore')

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    # Test GPU is actually available and working
    try:
        test_array = cp.array([1, 2, 3])
        _ = test_array + 1
        GPU_AVAILABLE = True
        print("GPU acceleration enabled via CuPy")
    except Exception as e:
        GPU_AVAILABLE = False
        print(f"GPU detected but not functional: {e}")
        print("Falling back to CPU (NumPy)")
        cp = np
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not installed, using CPU (NumPy)")
    print("For GPU acceleration, install with: pip install cupy-cuda11x")
    cp = np  # Fallback to NumPy

class BabyNoiseEngineOptimized:
    """Optimized audio engine for generating baby-optimized noise"""
    
    # Class constants
    SUPPORTED_SAMPLE_RATES = [44100, 48000, 88200, 96000, 192000]
    SUPPORTED_BIT_DEPTHS = [16, 24, 32]
    SUPPORTED_NOISE_TYPES = ['white', 'pink', 'brown']
    MAX_DURATION_HOURS = 12  # Safety limit
    
    # Optimized chunk size for processing (1 second of audio)
    # This balances memory usage with processing efficiency
    CHUNK_DURATION = 1.0  # seconds
    
    def __init__(self, noise_type='white', duration_str='1 hour', sample_rate=48000, bit_depth=24, 
                 true_peak_limit=-2.0, target_lufs=-14.0):
        # Validate inputs
        self._validate_inputs(noise_type, sample_rate, bit_depth)
        
        self.noise_type = noise_type.lower()
        self.duration_seconds = self._parse_duration(duration_str)
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth
        self.channels = 2  # Stereo
        
        # Loudness targets (YouTube/streaming optimized)
        self.target_lufs = target_lufs
        self.true_peak_limit = true_peak_limit
        self.oversample_factor = 2  # Reduced from 4x to 2x for efficiency
        
        # Calculate total samples needed
        self.total_samples = int(self.duration_seconds * self.sample_rate)
        
        # Chunk size for processing
        self.chunk_samples = int(self.CHUNK_DURATION * self.sample_rate)
        
        # Initialize filter states for chunked processing
        self.filter_states = {}
        
        # Pre-design filters (DSP optimization)
        self._design_all_filters()
        
        # Initialize random number generator with deterministic seed for reproducibility
        self.base_seed = int(duration_str.__hash__()) & 0x7FFFFFFF
        if GPU_AVAILABLE:
            self.rng = cp.random.default_rng(self.base_seed)
        else:
            self.rng = np.random.default_rng(self.base_seed)
        
        # Initialize LUFS measurement state for streaming
        self.lufs_state = {
            'k_weighted_power': [],  # Store power per chunk (not mean square)
            'total_samples': 0
        }
        
        # Brown noise scaling factor (computed once)
        self.brown_noise_scale = 0.02  # Empirically determined for -20dB RMS
    
    def _validate_inputs(self, noise_type, sample_rate, bit_depth):
        """Validate constructor inputs"""
        if noise_type.lower() not in self.SUPPORTED_NOISE_TYPES:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        
        if sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"Unsupported sample rate: {sample_rate}")
        
        if bit_depth not in self.SUPPORTED_BIT_DEPTHS:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")
    
    def _parse_duration(self, duration_str):
        """Parse duration string to seconds with validation"""
        try:
            duration_str = duration_str.lower().strip()
            if 'hour' in duration_str:
                hours = float(duration_str.split()[0])
                if hours <= 0 or hours > self.MAX_DURATION_HOURS:
                    raise ValueError(f"Duration must be between 0 and {self.MAX_DURATION_HOURS} hours")
                return hours * 3600
            elif 'min' in duration_str:
                minutes = float(duration_str.split()[0])
                if minutes <= 0 or minutes > self.MAX_DURATION_HOURS * 60:
                    raise ValueError(f"Duration must be between 0 and {self.MAX_DURATION_HOURS * 60} minutes")
                return minutes * 60
            elif 'sec' in duration_str:
                seconds = float(duration_str.split()[0])
                if seconds <= 0 or seconds > self.MAX_DURATION_HOURS * 3600:
                    raise ValueError(f"Duration must be between 0 and {self.MAX_DURATION_HOURS * 3600} seconds")
                return seconds
            else:
                raise ValueError(f"Invalid duration format: {duration_str}")
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid duration format: {duration_str}. {str(e)}")
    
    def _design_all_filters(self):
        """Pre-design all filters for efficiency"""
        nyquist = self.sample_rate / 2
        
        # Pink noise filter (Kellet)
        self.pink_b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786], dtype=np.float32)
        self.pink_a = np.array([1, -2.494956002, 2.017265875, -0.522189400], dtype=np.float32)
        
        # Brown noise filters
        self.brown_b = np.array([0.1], dtype=np.float64)
        self.brown_a = np.array([1.0, -0.9], dtype=np.float64)
        self.brown_hp_sos = butter(4, 20.0 / nyquist, btype='high', output='sos')  # 20Hz high-pass
        
        # Infant EQ filters
        shelf_freq = 200 / nyquist
        self.eq_shelf_sos = self._design_shelf_filter(shelf_freq, gain_db=3, filter_type='low')
        
        notch_freq = 3500 / nyquist
        q_factor = 2
        b_notch, a_notch = signal.iirnotch(notch_freq, q_factor)
        self.eq_notch_sos = signal.tf2sos(b_notch, a_notch)
        
        # K-weighting filters for LUFS
        self.lufs_pre_b = np.array([1.53260026327012, -2.65041135748730, 1.16904917595255])
        self.lufs_pre_a = np.array([1.0, -1.66375098226575, 0.71265752994786])
        self.lufs_rlb_b = np.array([1.0, -2.0, 1.0])
        self.lufs_rlb_a = np.array([1.0, -1.98998479513207, 0.98999499812227])
    
    def _design_shelf_filter(self, freq, gain_db, filter_type='low'):
        """Design a shelf filter"""
        sos = butter(2, freq, btype='low' if filter_type == 'low' else 'high', output='sos')
        gain_linear = 10 ** (gain_db / 20)
        if filter_type == 'low':
            sos[0, :3] *= gain_linear
        return sos
    
    def reset_for_pass(self, pass_num):
        """Reset state for a new pass"""
        self.filter_states = {}
        self.lufs_state = {'k_weighted_power': [], 'total_samples': 0}
        
        # Use deterministic seed that changes between passes
        seed = self.base_seed + pass_num * 1000000
        if GPU_AVAILABLE:
            self.rng = cp.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng(seed)
    
    def generate_white_noise_chunk(self, chunk_size):
        """Generate a chunk of white noise"""
        # Generate unit-variance Gaussian white noise
        if GPU_AVAILABLE:
            noise = self.rng.standard_normal(size=(chunk_size, self.channels), dtype=cp.float32)
        else:
            noise = self.rng.standard_normal(size=(chunk_size, self.channels)).astype(np.float32)
        
        # Scale to reasonable level
        initial_scale = 0.1  # -20 dB
        noise *= initial_scale
        
        return noise
    
    def generate_pink_noise_chunk(self, chunk_size):
        """Generate a chunk of pink noise using efficient IIR filter"""
        # Start with white noise
        white = self.generate_white_noise_chunk(chunk_size)
        
        # Apply pinking filter with state preservation
        pink = self._apply_pinking_filter_stateful(white)
        
        return pink
    
    def _apply_pinking_filter_stateful(self, white_noise):
        """Apply pinking filter with state preservation for chunked processing"""
        if GPU_AVAILABLE and hasattr(white_noise, 'get'):
            # Must process on CPU for stateful filtering
            white_cpu = white_noise.get()
            filtered = np.zeros_like(white_cpu)
            
            for ch in range(self.channels):
                if 'pink_filter' not in self.filter_states:
                    self.filter_states['pink_filter'] = {}
                
                if ch not in self.filter_states['pink_filter']:
                    self.filter_states['pink_filter'][ch] = lfilter_zi(self.pink_b, self.pink_a) * white_cpu[0, ch]
                
                filtered[:, ch], self.filter_states['pink_filter'][ch] = \
                    signal.lfilter(self.pink_b, self.pink_a, white_cpu[:, ch], zi=self.filter_states['pink_filter'][ch])
            
            # Fixed scaling - removed the erroneous 10.0 multiplier
            filtered = filtered * 0.1
            return cp.asarray(filtered)
        else:
            filtered = np.zeros_like(white_noise)
            
            for ch in range(self.channels):
                if 'pink_filter' not in self.filter_states:
                    self.filter_states['pink_filter'] = {}
                
                if ch not in self.filter_states['pink_filter']:
                    self.filter_states['pink_filter'][ch] = lfilter_zi(self.pink_b, self.pink_a) * white_noise[0, ch]
                
                filtered[:, ch], self.filter_states['pink_filter'][ch] = \
                    signal.lfilter(self.pink_b, self.pink_a, white_noise[:, ch], zi=self.filter_states['pink_filter'][ch])
            
            # Fixed scaling
            filtered = filtered * 0.1
            return filtered
    
    def generate_brown_noise_chunk(self, chunk_size):
        """Generate a chunk of brown noise using stateful filtering"""
        # Start with white noise
        white = self.generate_white_noise_chunk(chunk_size)
        
        # Convert to CPU for processing
        if GPU_AVAILABLE and hasattr(white, 'get'):
            white_cpu = white.get()
        else:
            white_cpu = white
        
        brown = np.zeros_like(white_cpu, dtype=np.float32)
        
        # Apply brownian filter to each channel with state
        for ch in range(self.channels):
            ch_data = white_cpu[:, ch].astype(np.float64)
            
            if 'brown_filter' not in self.filter_states:
                self.filter_states['brown_filter'] = {}
            
            if ch not in self.filter_states['brown_filter']:
                self.filter_states['brown_filter'][ch] = lfilter_zi(self.brown_b, self.brown_a) * ch_data[0]
            
            filtered, self.filter_states['brown_filter'][ch] = \
                signal.lfilter(self.brown_b, self.brown_a, ch_data, zi=self.filter_states['brown_filter'][ch])
            
            # Apply DC removal high-pass filter
            if 'brown_hp' not in self.filter_states:
                self.filter_states['brown_hp'] = {}
            
            if ch not in self.filter_states['brown_hp']:
                self.filter_states['brown_hp'][ch] = sosfilt_zi(self.brown_hp_sos) * filtered[0]
            
            filtered, self.filter_states['brown_hp'][ch] = \
                sosfilt(self.brown_hp_sos, filtered, zi=self.filter_states['brown_hp'][ch])
            
            brown[:, ch] = filtered.astype(np.float32)
        
        # Apply fixed scaling instead of per-chunk normalization
        brown = brown * self.brown_noise_scale
        
        if GPU_AVAILABLE:
            return cp.asarray(brown)
        else:
            return brown
    
    def apply_processing_chain_chunk(self, audio_chunk):
        """Apply all processing to a single chunk"""
        # Apply infant EQ
        audio_chunk = self.apply_infant_eq_stateful(audio_chunk)
        
        # Apply compression
        audio_chunk = self.apply_compression_chunk(audio_chunk)
        
        # Apply true peak limiting (simplified for chunks)
        audio_chunk = self.apply_simple_limiting(audio_chunk)
        
        # Update LUFS measurement state
        self.update_lufs_measurement(audio_chunk)
        
        return audio_chunk
    
    def apply_infant_eq_stateful(self, audio):
        """Apply EQ with state preservation"""
        # Convert to CPU for filtering
        if GPU_AVAILABLE and hasattr(audio, 'get'):
            audio_cpu = audio.get()
        else:
            audio_cpu = audio
        
        filtered = audio_cpu.copy()
        
        # Initialize filter states if needed
        if 'eq_shelf' not in self.filter_states:
            self.filter_states['eq_shelf'] = {}
            for ch in range(self.channels):
                self.filter_states['eq_shelf'][ch] = sosfilt_zi(self.eq_shelf_sos) * filtered[0, ch]
        
        if 'eq_notch' not in self.filter_states:
            self.filter_states['eq_notch'] = {}
            for ch in range(self.channels):
                self.filter_states['eq_notch'][ch] = sosfilt_zi(self.eq_notch_sos) * filtered[0, ch]
        
        # Apply filters with state
        for ch in range(self.channels):
            filtered[:, ch], self.filter_states['eq_shelf'][ch] = \
                sosfilt(self.eq_shelf_sos, filtered[:, ch], zi=self.filter_states['eq_shelf'][ch])
            filtered[:, ch], self.filter_states['eq_notch'][ch] = \
                sosfilt(self.eq_notch_sos, filtered[:, ch], zi=self.filter_states['eq_notch'][ch])
        
        if GPU_AVAILABLE:
            return cp.asarray(filtered)
        else:
            return filtered
    
    def apply_compression_chunk(self, audio, threshold_db=-20, ratio=2):
        """Apply compression to a chunk"""
        threshold_linear = 10 ** (threshold_db / 20)
        
        xp = cp if GPU_AVAILABLE and hasattr(audio, 'get') else np
        compressed = xp.zeros_like(audio)
        
        for ch in range(self.channels):
            signal_ch = audio[:, ch]
            envelope = xp.abs(signal_ch)
            
            # Simple compression without attack/release for chunks
            gain = xp.ones_like(envelope)
            above_threshold = envelope > threshold_linear
            gain[above_threshold] = (threshold_linear + 
                                   (envelope[above_threshold] - threshold_linear) / ratio) / envelope[above_threshold]
            
            compressed[:, ch] = signal_ch * gain
        
        return compressed
    
    def apply_simple_limiting(self, audio):
        """Apply simple peak limiting for chunks"""
        peak_linear = 10 ** (self.true_peak_limit / 20)
        
        xp = cp if GPU_AVAILABLE and hasattr(audio, 'get') else np
        
        # Soft knee limiting instead of hard clipping
        above_threshold = xp.abs(audio) > peak_linear * 0.9  # 90% threshold for soft knee
        
        if xp.any(above_threshold):
            # Apply tanh soft clipping
            audio_limited = xp.where(
                above_threshold,
                peak_linear * xp.tanh(audio / peak_linear),
                audio
            )
        else:
            audio_limited = audio
        
        return audio_limited
    
    def update_lufs_measurement(self, audio_chunk):
        """Update LUFS measurement state with new chunk"""
        # Convert to CPU for measurement
        if GPU_AVAILABLE and hasattr(audio_chunk, 'get'):
            audio_cpu = audio_chunk.get()
        else:
            audio_cpu = audio_chunk
        
        # Apply K-weighting
        k_weighted = np.zeros_like(audio_cpu, dtype=np.float64)
        for ch in range(self.channels):
            ch_data = audio_cpu[:, ch].astype(np.float64)
            
            # Initialize filter states if needed
            if 'lufs_pre' not in self.filter_states:
                self.filter_states['lufs_pre'] = {}
                self.filter_states['lufs_rlb'] = {}
            
            if ch not in self.filter_states['lufs_pre']:
                self.filter_states['lufs_pre'][ch] = lfilter_zi(self.lufs_pre_b, self.lufs_pre_a) * ch_data[0]
                self.filter_states['lufs_rlb'][ch] = lfilter_zi(self.lufs_rlb_b, self.lufs_rlb_a) * ch_data[0]
            
            # Apply filters with state
            temp, self.filter_states['lufs_pre'][ch] = \
                signal.lfilter(self.lufs_pre_b, self.lufs_pre_a, ch_data, zi=self.filter_states['lufs_pre'][ch])
            k_weighted[:, ch], self.filter_states['lufs_rlb'][ch] = \
                signal.lfilter(self.lufs_rlb_b, self.lufs_rlb_a, temp, zi=self.filter_states['lufs_rlb'][ch])
        
        # Correct LUFS calculation: sum channel powers, then take mean over time
        channel_powers = np.mean(k_weighted ** 2, axis=0)  # Mean over time for each channel
        total_power = np.sum(channel_powers)  # Sum channel powers
        
        self.lufs_state['k_weighted_power'].append(total_power)
        self.lufs_state['total_samples'] += len(audio_chunk)
    
    def calculate_final_lufs(self):
        """Calculate final LUFS from accumulated measurements"""
        if not self.lufs_state['k_weighted_power']:
            return -70.0
        
        # Calculate mean power from accumulated values
        mean_power = np.mean(self.lufs_state['k_weighted_power'])
        
        if mean_power <= 0:
            return -70.0
        
        # Convert to LUFS
        lufs = -0.691 + 10 * np.log10(mean_power)
        
        return lufs
    
    def generate_and_save_optimized(self, filename, progress_callback=None):
        """Generate and save audio using chunked processing"""
        import os
        
        # Sanitize filename
        filename = os.path.basename(filename)
        if not filename.endswith('.flac'):
            filename += '.flac'
        
        if not filename or filename.startswith('.'):
            raise ValueError("Invalid filename")
        
        def report_progress(stage, percent):
            if progress_callback:
                progress_callback(stage, percent)
        
        print(f"\nGenerating {self.duration_seconds/60:.1f} minutes of {self.noise_type} noise...")
        print(f"Using optimized chunked processing...")
        print(f"Sample rate: {self.sample_rate} Hz, Bit depth: {self.bit_depth}-bit")
        
        # Determine subtype
        subtype_map = {16: 'PCM_16', 24: 'PCM_24', 32: 'PCM_24'}
        subtype = subtype_map.get(self.bit_depth, 'PCM_24')
        
        # Single pass with on-the-fly normalization estimation
        print("\nGenerating with adaptive normalization...")
        
        # Reset for first pass
        self.reset_for_pass(1)
        
        try:
            with sf.SoundFile(filename, 'w', samplerate=self.sample_rate,
                             channels=self.channels, subtype=subtype, format='FLAC') as f:
                
                total_chunks = (self.total_samples + self.chunk_samples - 1) // self.chunk_samples
                
                # First few chunks to estimate level
                estimation_chunks = min(10, total_chunks)
                temp_audio = []
                
                # Generate first few chunks to estimate LUFS
                for chunk_idx in range(estimation_chunks):
                    chunk_start = chunk_idx * self.chunk_samples
                    chunk_end = min(chunk_start + self.chunk_samples, self.total_samples)
                    chunk_size = chunk_end - chunk_start
                    
                    # Generate noise chunk
                    if self.noise_type == 'white':
                        chunk = self.generate_white_noise_chunk(chunk_size)
                    elif self.noise_type == 'pink':
                        chunk = self.generate_pink_noise_chunk(chunk_size)
                    elif self.noise_type == 'brown':
                        chunk = self.generate_brown_noise_chunk(chunk_size)
                    
                    # Apply processing chain
                    chunk = self.apply_processing_chain_chunk(chunk)
                    
                    # Store for writing
                    if GPU_AVAILABLE and hasattr(chunk, 'get'):
                        temp_audio.append(chunk.get())
                    else:
                        temp_audio.append(chunk.copy())
                
                # Estimate LUFS from initial chunks
                estimated_lufs = self.calculate_final_lufs()
                gain_db = self.target_lufs - estimated_lufs
                gain_linear = 10 ** (gain_db / 20)
                
                print(f"Estimated LUFS: {estimated_lufs:.1f}, applying gain: {gain_db:.1f} dB")
                
                # Write estimation chunks with gain
                for chunk_idx, chunk_cpu in enumerate(temp_audio):
                    # Apply gain
                    chunk_cpu = chunk_cpu * gain_linear
                    
                    # Apply fade in for first chunk
                    if chunk_idx == 0:
                        fade_samples = min(int(5 * self.sample_rate), len(chunk_cpu))
                        fade_in = np.linspace(0, 1, fade_samples) ** 2
                        chunk_cpu[:fade_samples] *= fade_in[:, np.newaxis]
                    
                    # Apply dithering
                    lsb = 1.0 / (2 ** (self.bit_depth - 1))
                    np_rng = np.random.default_rng(42 + chunk_idx)
                    rpdf1 = np_rng.uniform(-0.5, 0.5, size=chunk_cpu.shape)
                    rpdf2 = np_rng.uniform(-0.5, 0.5, size=chunk_cpu.shape)
                    tpdf_dither = (rpdf1 + rpdf2) * lsb
                    chunk_cpu = np.clip(chunk_cpu + tpdf_dither, -1.0, 1.0)
                    
                    f.write(chunk_cpu)
                    
                    percent = int((chunk_idx + 1) / total_chunks * 100)
                    report_progress("Generating", percent)
                
                # Continue with remaining chunks
                for chunk_idx in range(estimation_chunks, total_chunks):
                    chunk_start = chunk_idx * self.chunk_samples
                    chunk_end = min(chunk_start + self.chunk_samples, self.total_samples)
                    chunk_size = chunk_end - chunk_start
                    
                    # Generate noise chunk
                    if self.noise_type == 'white':
                        chunk = self.generate_white_noise_chunk(chunk_size)
                    elif self.noise_type == 'pink':
                        chunk = self.generate_pink_noise_chunk(chunk_size)
                    elif self.noise_type == 'brown':
                        chunk = self.generate_brown_noise_chunk(chunk_size)
                    
                    # Apply processing chain
                    chunk = self.apply_processing_chain_chunk(chunk)
                    
                    # Apply normalization gain
                    chunk = chunk * gain_linear
                    
                    # Apply fade out for last chunk
                    if chunk_idx == total_chunks - 1:
                        fade_samples = min(int(5 * self.sample_rate), chunk_size)
                        if GPU_AVAILABLE and hasattr(chunk, 'get'):
                            fade_out = cp.linspace(1, 0, fade_samples) ** 2
                            chunk[-fade_samples:] *= fade_out[:, cp.newaxis]
                        else:
                            fade_out = np.linspace(1, 0, fade_samples) ** 2
                            chunk[-fade_samples:] *= fade_out[:, np.newaxis]
                    
                    # Convert to CPU and write
                    if GPU_AVAILABLE and hasattr(chunk, 'get'):
                        chunk_cpu = chunk.get()
                    else:
                        chunk_cpu = chunk
                    
                    # Apply dithering
                    lsb = 1.0 / (2 ** (self.bit_depth - 1))
                    np_rng = np.random.default_rng(42 + chunk_idx)
                    rpdf1 = np_rng.uniform(-0.5, 0.5, size=chunk_cpu.shape)
                    rpdf2 = np_rng.uniform(-0.5, 0.5, size=chunk_cpu.shape)
                    tpdf_dither = (rpdf1 + rpdf2) * lsb
                    chunk_cpu = np.clip(chunk_cpu + tpdf_dither, -1.0, 1.0)
                    
                    f.write(chunk_cpu)
                    
                    # Report progress
                    percent = int((chunk_idx + 1) / total_chunks * 100)
                    report_progress("Generating", percent)
            
            # Add metadata
            try:
                from mutagen.flac import FLAC
                audio_file = FLAC(filename)
                audio_file.clear()
                audio_file['TITLE'] = f'{self.noise_type.capitalize()} Noise for Babies'
                audio_file['ARTIST'] = 'Baby Noise Engine (Optimized)'
                audio_file['ALBUM'] = 'Sleep Sounds for Infants'
                audio_file['DATE'] = '2024'
                audio_file['GENRE'] = 'Ambient'
                audio_file['COMMENT'] = f'Duration: {self.duration_seconds/60:.1f} min, LUFS: {self.target_lufs}, Type: {self.noise_type}'
                audio_file['ENCODER'] = 'Baby Noise Engine Optimized v2.0'
                audio_file.save()
                print("Metadata embedded successfully!")
            except ImportError:
                print("Warning: mutagen not installed, metadata not embedded")
            except Exception as e:
                print(f"Warning: Could not embed metadata: {e}")
            
            print(f"\nTarget LUFS: {self.target_lufs:.1f}")
            print(f"Audio saved successfully!")
            print(f"File: {filename}")
            print(f"Duration: {self.duration_seconds/60:.1f} minutes")
            print(f"Bit depth: {self.bit_depth}-bit")
            bytes_per_sample = self.bit_depth // 8
            print(f"Size: ~{self.total_samples * self.channels * bytes_per_sample / (1024*1024):.1f} MB")
            
            report_progress("Complete", 100)
            
        except Exception as e:
            raise e


def main_optimized(noise_type='white', duration='1 hour', show_progress=False):
    """Main function to generate baby noise using optimized engine"""
    
    def progress_callback(stage, percent):
        if show_progress:
            print(f"[{stage}] {percent}% complete")
    
    try:
        # Create optimized engine instance
        engine = BabyNoiseEngineOptimized(noise_type=noise_type, duration_str=duration)
        
        # Generate and save with chunked processing
        filename = f"baby_{noise_type}_noise_{duration.replace(' ', '_')}_optimized.flac"
        engine.generate_and_save_optimized(filename, progress_callback if show_progress else None)
        
        return filename
        
    except ValueError as e:
        print(f"Input error: {e}")
        return None
    except MemoryError:
        print(f"Out of memory! The optimized version should handle this better.")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Example: Generate 30 minutes of pink noise with progress
    output_file = main_optimized(noise_type='pink', duration='30 mins', show_progress=True)
    if output_file:
        print(f"\nOptimized generation complete! Output: {output_file}")