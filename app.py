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
from scipy.ndimage import maximum_filter1d
from collections import deque
from dataclasses import dataclass
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


@dataclass
class LimiterConfig:
    """Configuration for lookahead limiter"""
    sample_rate: int = 48000
    channels: int = 2
    threshold_db: float = -0.3  # dB below 0dBFS
    lookahead_ms: float = 5.0  # 5ms lookahead
    attack_ms: float = 0.5  # Fast attack
    release_ms: float = 50.0  # Smooth release
    knee_db: float = 2.0  # Soft knee width
    oversample_factor: int = 4  # For inter-sample peak detection
    
    def __post_init__(self):
        self.threshold_linear = 10 ** (self.threshold_db / 20)
        self.lookahead_samples = int(self.lookahead_ms * self.sample_rate / 1000)
        self.attack_coeff = np.exp(-1 / (self.attack_ms * self.sample_rate / 1000))
        self.release_coeff = np.exp(-1 / (self.release_ms * self.sample_rate / 1000))
        self.knee_start = 10 ** ((self.threshold_db - self.knee_db / 2) / 20)
        self.knee_end = self.threshold_linear


class LookaheadLimiter:
    """Professional lookahead limiter with soft knee and inter-sample peak detection"""
    
    def __init__(self, config: LimiterConfig):
        self.config = config
        
        # Delay buffers for lookahead
        self.delay_buffers = [
            deque(maxlen=config.lookahead_samples)
            for _ in range(config.channels)
        ]
        
        # Initialize delay buffers with zeros
        for buffer in self.delay_buffers:
            buffer.extend([0.0] * config.lookahead_samples)
        
        # Gain reduction envelope
        self.envelope = 0.0
        
        # Upsampling filter for inter-sample peak detection
        if config.oversample_factor > 1:
            # Design a high-quality lowpass filter for upsampling
            cutoff = 0.45  # Slightly below Nyquist to avoid aliasing
            self.upsample_filter = signal.firwin(
                64 * config.oversample_factor,
                cutoff / config.oversample_factor,
                window='blackman'
            )
        else:
            self.upsample_filter = None
    
    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Process audio chunk with vectorized lookahead limiting"""
        num_samples = len(audio_chunk)
        
        # Convert deque buffers to numpy arrays for vectorized operations
        if not hasattr(self, 'lookahead_buffer'):
            # Initialize numpy buffer on first use
            self.lookahead_buffer = np.zeros((self.config.lookahead_samples, self.config.channels))
            self.envelope_state = np.ones(num_samples)  # Initialize envelope
        
        # Combine lookahead buffer with new audio
        combined = np.vstack([self.lookahead_buffer, audio_chunk])
        
        # Extract delayed output (first num_samples from combined buffer)
        delayed_output = combined[:num_samples]
        
        # Update lookahead buffer for next chunk
        self.lookahead_buffer = combined[num_samples:num_samples + self.config.lookahead_samples]
        
        # Vectorized peak detection using sliding window maximum
        peaks = self._find_lookahead_peaks_vectorized(combined, num_samples)
        
        # Vectorized gain reduction calculation
        gain_reductions = self._calculate_gain_reduction_vectorized(peaks)
        
        # Vectorized envelope smoothing
        envelope = self._apply_envelope_vectorized(gain_reductions)
        
        # Apply gain envelope to delayed signal
        output = delayed_output * envelope[:, np.newaxis]
        
        return output
    
    def _find_lookahead_peaks_vectorized(self, combined_buffer: np.ndarray, num_samples: int) -> np.ndarray:
        """Truly vectorized peak detection using scipy.ndimage.maximum_filter1d"""
        # Take the absolute maximum across channels
        abs_signal = np.abs(combined_buffer).max(axis=1) if combined_buffer.ndim > 1 else np.abs(combined_buffer)
        
        # Use scipy's optimized maximum_filter1d for sliding window maximum
        # This is compiled C code, orders of magnitude faster than Python loops
        all_peaks = maximum_filter1d(abs_signal, size=self.config.lookahead_samples, 
                                     mode='constant', cval=0.0)
        
        # Return only the peaks corresponding to the output samples
        return all_peaks[:num_samples]
    
    def _find_lookahead_peak(self) -> float:
        """Find peak level in lookahead buffer, including inter-sample peaks"""
        peak = 0.0
        
        for ch in range(self.config.channels):
            # Get buffer as array
            buffer = np.array(self.delay_buffers[ch])
            
            if self.config.oversample_factor > 1:
                # Upsample to detect inter-sample peaks
                upsampled = self._upsample(buffer)
                ch_peak = np.max(np.abs(upsampled))
            else:
                ch_peak = np.max(np.abs(buffer))
            
            peak = max(peak, ch_peak)
        
        return peak
    
    def _upsample(self, signal_data: np.ndarray) -> np.ndarray:
        """Upsample signal for inter-sample peak detection"""
        # Insert zeros between samples
        upsampled = np.zeros(len(signal_data) * self.config.oversample_factor)
        upsampled[::self.config.oversample_factor] = signal_data
        
        # Apply lowpass filter
        filtered = np.convolve(upsampled, self.upsample_filter, mode='same')
        
        # Compensate for filter gain
        filtered *= self.config.oversample_factor
        
        return filtered
    
    def _calculate_gain_reduction_vectorized(self, peaks: np.ndarray) -> np.ndarray:
        """Vectorized gain reduction with soft knee"""
        gains = np.ones_like(peaks)
        
        # Below knee - no reduction
        below_knee = peaks <= self.config.knee_start
        
        # Above knee - full limiting  
        above_knee = peaks >= self.config.knee_end
        gains[above_knee] = self.config.threshold_linear / peaks[above_knee]
        
        # In knee region - smooth transition
        in_knee = ~below_knee & ~above_knee
        if np.any(in_knee):
            knee_peaks = peaks[in_knee]
            knee_position = (knee_peaks - self.config.knee_start) / (self.config.knee_end - self.config.knee_start)
            knee_factor = knee_position * knee_position
            full_limiting_gain = self.config.threshold_linear / knee_peaks
            gains[in_knee] = 1.0 + (full_limiting_gain - 1.0) * knee_factor
        
        return gains
    
    def _apply_envelope_vectorized(self, gain_reductions: np.ndarray) -> np.ndarray:
        """Truly vectorized envelope smoothing using scipy.signal.lfilter"""
        # Initialize previous envelope if needed
        if not hasattr(self, 'prev_envelope'):
            self.prev_envelope = 1.0
        
        # Prepare gain reductions with previous value prepended
        extended_gains = np.concatenate(([self.prev_envelope], gain_reductions))
        
        # Detect attack vs release (comparing each sample to previous)
        is_attacking = extended_gains[1:] < extended_gains[:-1]
        
        # Find where attack/release state changes
        state_changes = np.where(np.diff(np.concatenate(([False], is_attacking))))[0]
        
        # Add start and end indices
        segment_bounds = np.concatenate(([0], state_changes, [len(gain_reductions)]))
        
        # Process each segment with vectorized filter
        envelope = np.empty_like(gain_reductions)
        prev_val = self.prev_envelope
        
        for i in range(len(segment_bounds) - 1):
            start_idx = segment_bounds[i]
            end_idx = segment_bounds[i + 1]
            
            if start_idx >= end_idx:
                continue
            
            segment = gain_reductions[start_idx:end_idx]
            
            # Determine if this segment is attack or release
            if start_idx < len(is_attacking) and is_attacking[start_idx]:
                coeff = self.config.attack_coeff
            else:
                coeff = self.config.release_coeff
            
            # Vectorized IIR filter: y[n] = (1-coeff)*x[n] + coeff*y[n-1]
            # Using lfilter for true vectorization
            b = np.array([1.0 - coeff])
            a = np.array([1.0, -coeff])
            
            # Set initial conditions based on previous value
            zi = signal.lfilter_zi(b, a) * prev_val
            
            # Apply filter
            segment_envelope, zf = signal.lfilter(b, a, segment, zi=zi)
            envelope[start_idx:end_idx] = segment_envelope
            
            # Update previous value for next segment
            prev_val = segment_envelope[-1] if len(segment_envelope) > 0 else prev_val
        
        self.prev_envelope = envelope[-1] if len(envelope) > 0 else self.prev_envelope
        return envelope
    
    def _calculate_gain_reduction(self, peak: float) -> float:
        """Calculate gain reduction with soft knee"""
        if peak <= self.config.knee_start:
            # Below knee - no reduction
            return 1.0
        elif peak >= self.config.knee_end:
            # Above knee - full limiting
            return self.config.threshold_linear / peak
        else:
            # In knee region - smooth transition
            # Calculate position in knee (0 to 1)
            knee_position = (peak - self.config.knee_start) / (self.config.knee_end - self.config.knee_start)
            
            # Quadratic knee curve
            knee_factor = knee_position * knee_position
            
            # Interpolate between no reduction and full limiting
            no_reduction_gain = 1.0
            full_limiting_gain = self.config.threshold_linear / peak
            
            return no_reduction_gain + (full_limiting_gain - no_reduction_gain) * knee_factor
    
    def reset(self) -> None:
        """Reset limiter state"""
        # Clear delay buffers
        for buffer in self.delay_buffers:
            buffer.clear()
            buffer.extend([0.0] * self.config.lookahead_samples)
        
        # Reset envelope
        self.envelope = 0.0
    
    def get_latency_samples(self) -> int:
        """Get the latency introduced by the lookahead buffer"""
        return self.config.lookahead_samples


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
                 true_peak_limit=-2.0, target_lufs=-14.0, stereo_width=0.5):
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
        
        # Stereo width (0.0 = mono, 1.0 = full width)
        self.stereo_width = max(0.0, min(1.0, stereo_width))
        
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
        
        # Initialize LUFS measurement state for streaming with BS.1770-4 gating
        self.lufs_state = {
            'k_weighted_power': [],  # Store power per chunk (not mean square)
            'momentary_powers': [],  # Store powers for 400ms windows
            'total_samples': 0,
            'sample_buffer': [],  # Buffer for momentary loudness
            'momentary_window_samples': int(0.4 * self.sample_rate),  # 400ms window
            'hop_samples': int(0.1 * self.sample_rate)  # 100ms hop for 75% overlap
        }
        
        # Brown noise scaling factor (computed once)
        self.brown_noise_scale = 0.02  # Empirically determined for -20dB RMS
        
        # Initialize lookahead limiter
        limiter_config = LimiterConfig(
            sample_rate=self.sample_rate,
            channels=self.channels,
            threshold_db=self.true_peak_limit,
            lookahead_ms=5.0,
            attack_ms=0.5,
            release_ms=50.0,
            knee_db=2.0,
            oversample_factor=self.oversample_factor
        )
        self.lookahead_limiter = LookaheadLimiter(limiter_config)
        print(f"Using professional lookahead limiter with {limiter_config.lookahead_ms}ms lookahead")
        print(f"Stereo width enhancement: {int(self.stereo_width * 100)}% {'(mono)' if self.stereo_width == 0 else '(wider)' if self.stereo_width > 0.5 else '(natural)'}")
    
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
        # Brown noise filter - proper integration filter using scipy butter
        # Using a very low cutoff frequency (10 Hz at 48kHz sample rate = 10/24000 = 0.000417)
        # This creates a better 1/f² characteristic for brown noise
        self.brown_sos = butter(1, 10.0 / (self.sample_rate / 2), btype='low', output='sos')
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
        
        # Stereo width enhancement filters (allpass filters for decorrelation)
        # Design complementary allpass filters for L/R channels
        # These create subtle phase differences without affecting frequency response
        self.width_allpass_sos_l = self._design_allpass_filter(800 / nyquist, q=0.7)
        self.width_allpass_sos_r = self._design_allpass_filter(1200 / nyquist, q=0.7)
        
        # Additional allpass for more complex decorrelation
        self.width_allpass2_sos_l = self._design_allpass_filter(2000 / nyquist, q=0.5)
        self.width_allpass2_sos_r = self._design_allpass_filter(3000 / nyquist, q=0.5)
    
    def _design_shelf_filter(self, freq, gain_db, filter_type='low'):
        """Design a shelf filter"""
        sos = butter(2, freq, btype='low' if filter_type == 'low' else 'high', output='sos')
        gain_linear = 10 ** (gain_db / 20)
        if filter_type == 'low':
            sos[0, :3] *= gain_linear
        return sos
    
    def _design_allpass_filter(self, freq, q=0.7):
        """Design an allpass filter for phase decorrelation"""
        # Convert frequency and Q to coefficients
        w0 = 2 * np.pi * freq
        alpha = np.sin(w0) / (2 * q)
        
        # Allpass coefficients
        b = np.array([1 - alpha, -2 * np.cos(w0), 1 + alpha])
        a = np.array([1 + alpha, -2 * np.cos(w0), 1 - alpha])
        
        # Normalize
        b = b / a[0]
        a = a / a[0]
        
        return signal.tf2sos(b, a)
    
    def reset_for_pass(self, pass_num):
        """Reset state for a new pass"""
        self.filter_states = {}
        self.lufs_state = {
            'k_weighted_power': [],
            'momentary_powers': [],
            'total_samples': 0,
            'sample_buffer': [],
            'momentary_window_samples': int(0.4 * self.sample_rate),
            'hop_samples': int(0.1 * self.sample_rate)
        }
        
        # Use deterministic seed that changes between passes
        seed = self.base_seed + pass_num * 1000000
        if GPU_AVAILABLE:
            self.rng = cp.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng(seed)
        
        # Reset lookahead limiter
        self.lookahead_limiter.reset()
    
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
                self.filter_states['brown_filter'][ch] = sosfilt_zi(self.brown_sos) * ch_data[0]
            
            filtered, self.filter_states['brown_filter'][ch] = \
                sosfilt(self.brown_sos, ch_data, zi=self.filter_states['brown_filter'][ch])
            
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
        # Apply stereo width enhancement first (before other processing)
        audio_chunk = self.apply_stereo_width_enhancement(audio_chunk)
        
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
        """Apply peak limiting for chunks with professional lookahead limiter"""
        # Convert to CPU for processing if needed
        if GPU_AVAILABLE and hasattr(audio, 'get'):
            audio_cpu = audio.get()
        else:
            audio_cpu = audio
        
        # Process through lookahead limiter
        limited = self.lookahead_limiter.process(audio_cpu)
        
        # Convert back to GPU if needed
        if GPU_AVAILABLE and hasattr(audio, 'get'):
            return cp.asarray(limited)
        return limited
    
    def apply_stereo_width_enhancement(self, audio):
        """Apply stereo width enhancement using allpass filters for decorrelation"""
        if self.stereo_width == 0.0:
            # Mono - return average of channels
            if GPU_AVAILABLE and hasattr(audio, 'get'):
                xp = cp
                audio_work = audio
            else:
                xp = np
                audio_work = audio
            
            mono = xp.mean(audio_work, axis=1, keepdims=True)
            return xp.tile(mono, (1, 2))
        
        # Convert to CPU for processing (allpass filters need scipy)
        if GPU_AVAILABLE and hasattr(audio, 'get'):
            audio_cpu = audio.get()
        else:
            audio_cpu = audio
        
        # Process left and right channels with different allpass filters
        enhanced = np.zeros_like(audio_cpu)
        
        # Left channel - apply first set of allpass filters
        if 'width_allpass_l' not in self.filter_states:
            self.filter_states['width_allpass_l'] = sosfilt_zi(self.width_allpass_sos_l) * audio_cpu[0, 0]
            self.filter_states['width_allpass2_l'] = sosfilt_zi(self.width_allpass2_sos_l) * audio_cpu[0, 0]
        
        temp_l, self.filter_states['width_allpass_l'] = \
            sosfilt(self.width_allpass_sos_l, audio_cpu[:, 0], zi=self.filter_states['width_allpass_l'])
        enhanced[:, 0], self.filter_states['width_allpass2_l'] = \
            sosfilt(self.width_allpass2_sos_l, temp_l, zi=self.filter_states['width_allpass2_l'])
        
        # Right channel - apply second set of allpass filters
        if 'width_allpass_r' not in self.filter_states:
            self.filter_states['width_allpass_r'] = sosfilt_zi(self.width_allpass_sos_r) * audio_cpu[0, 1]
            self.filter_states['width_allpass2_r'] = sosfilt_zi(self.width_allpass2_sos_r) * audio_cpu[0, 1]
        
        temp_r, self.filter_states['width_allpass_r'] = \
            sosfilt(self.width_allpass_sos_r, audio_cpu[:, 1], zi=self.filter_states['width_allpass_r'])
        enhanced[:, 1], self.filter_states['width_allpass2_r'] = \
            sosfilt(self.width_allpass2_sos_r, temp_r, zi=self.filter_states['width_allpass2_r'])
        
        # Mix original and enhanced based on width parameter
        # Also apply subtle M/S processing for extra width
        if self.stereo_width > 0.5:
            # Extract mid and side
            mid = (enhanced[:, 0] + enhanced[:, 1]) * 0.5
            side = (enhanced[:, 0] - enhanced[:, 1]) * 0.5
            
            # Boost side signal for extra width (carefully to avoid phase issues)
            width_boost = 1.0 + (self.stereo_width - 0.5) * 0.6  # Max 30% boost
            side *= width_boost
            
            # Reconstruct L/R
            enhanced[:, 0] = mid + side
            enhanced[:, 1] = mid - side
        
        # Mix with original signal
        mixed = audio_cpu * (1.0 - self.stereo_width * 0.3) + enhanced * (self.stereo_width * 0.3)
        
        # Convert back to GPU if needed
        if GPU_AVAILABLE and hasattr(audio, 'get'):
            return cp.asarray(mixed)
        return mixed
    
    def update_lufs_measurement(self, audio_chunk):
        """Update LUFS measurement state with new chunk (BS.1770-4 compliant)"""
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
        
        # Add samples to buffer for momentary loudness calculation
        self.lufs_state['sample_buffer'].extend(k_weighted.tolist())
        
        # Process complete 400ms windows with 75% overlap (100ms hop)
        while len(self.lufs_state['sample_buffer']) >= self.lufs_state['momentary_window_samples']:
            # Extract 400ms window
            window = np.array(self.lufs_state['sample_buffer'][:self.lufs_state['momentary_window_samples']])
            # Hop by 100ms for 75% overlap
            self.lufs_state['sample_buffer'] = self.lufs_state['sample_buffer'][self.lufs_state['hop_samples']:]
            
            # Calculate power for this window
            channel_powers = np.mean(window ** 2, axis=0)  # Mean over time for each channel
            total_power = np.sum(channel_powers)  # Sum channel powers
            
            # Calculate momentary loudness
            if total_power > 0:
                momentary_lufs = -0.691 + 10 * np.log10(total_power)
                
                # Apply absolute gating (-70 LUFS threshold)
                if momentary_lufs > -70:
                    self.lufs_state['momentary_powers'].append(total_power)
                    self.lufs_state['k_weighted_power'].append(total_power)
        
        self.lufs_state['total_samples'] += len(audio_chunk)
    
    def calculate_final_lufs(self):
        """Calculate final LUFS from accumulated measurements with BS.1770-4 two-stage gating"""
        if not self.lufs_state['k_weighted_power']:
            return -70.0
        
        # Process any remaining samples in buffer
        if len(self.lufs_state['sample_buffer']) > 0:
            window = np.array(self.lufs_state['sample_buffer'])
            channel_powers = np.mean(window ** 2, axis=0)
            total_power = np.sum(channel_powers)
            
            if total_power > 0:
                momentary_lufs = -0.691 + 10 * np.log10(total_power)
                if momentary_lufs > -70:  # Absolute gating
                    self.lufs_state['k_weighted_power'].append(total_power)
        
        # Stage 1: Absolute gating (already done during collection)
        powers = np.array(self.lufs_state['k_weighted_power'])
        
        if len(powers) == 0:
            return -70.0
        
        # Calculate ungated mean
        ungated_mean = np.mean(powers)
        
        if ungated_mean <= 0:
            return -70.0
        
        # Stage 2: Relative gating (-10 LU below ungated mean)
        relative_threshold = ungated_mean * (10 ** (-10 / 10))  # -10 dB relative
        gated_powers = powers[powers > relative_threshold]
        
        if len(gated_powers) == 0:
            return -70.0
        
        # Final gated mean
        final_mean = np.mean(gated_powers)
        lufs = -0.691 + 10 * np.log10(final_mean)
        
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


def main_optimized(noise_type='white', duration='1 hour', show_progress=False, stereo_width=0.5):
    """Main function to generate baby noise using optimized engine"""
    
    def progress_callback(stage, percent):
        if show_progress:
            print(f"[{stage}] {percent}% complete")
    
    try:
        # Create optimized engine instance
        engine = BabyNoiseEngineOptimized(noise_type=noise_type, duration_str=duration, stereo_width=stereo_width)
        
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


def benchmark_limiter(chunk_samples=48000, num_chunks=10):
    """Benchmark the limiter performance"""
    import time
    
    # Create limiter config
    config = LimiterConfig(
        sample_rate=48000,
        channels=2,
        threshold_db=-0.3,
        lookahead_ms=5.0,
        attack_ms=0.5,
        release_ms=50.0,
        knee_db=2.0,
        oversample_factor=4
    )
    
    # Create limiter instance
    limiter = LookaheadLimiter(config)
    
    print(f"\nBenchmarking with chunk size: {chunk_samples} samples ({chunk_samples/48000:.2f} seconds)")
    print(f"Processing {num_chunks} chunks ({num_chunks * chunk_samples/48000:.1f} seconds of audio)")
    
    # Warm-up run
    test_chunk = np.random.randn(chunk_samples, 2).astype(np.float32) * 0.5
    _ = limiter.process(test_chunk)
    
    # Reset limiter
    limiter.reset()
    
    # Benchmark
    processing_times = []
    for i in range(num_chunks):
        test_chunk = np.random.randn(chunk_samples, 2).astype(np.float32) * 0.5
        
        start_time = time.perf_counter()
        output = limiter.process(test_chunk)
        end_time = time.perf_counter()
        
        processing_times.append(end_time - start_time)
    
    # Calculate statistics
    total_time = sum(processing_times)
    avg_time = np.mean(processing_times)
    audio_duration = num_chunks * chunk_samples / 48000
    rtf = total_time / audio_duration
    
    print(f"\nResults:")
    print(f"  Total processing time: {total_time:.3f} seconds")
    print(f"  Average per chunk: {avg_time*1000:.2f} ms")
    print(f"  Real-time factor: {rtf:.3f}x (lower is better, 1.0 = real-time)")
    print(f"  Throughput: {1/rtf:.1f}x real-time")
    
    return rtf

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        print("=" * 60)
        print("LookaheadLimiter Performance Benchmark")
        print("=" * 60)
        
        # Test different chunk sizes
        chunk_sizes = [
            (480, 100),     # 10ms chunks
            (4800, 100),    # 100ms chunks
            (48000, 60),    # 1 second chunks
        ]
        
        results = []
        for chunk_samples, num_chunks in chunk_sizes:
            rtf = benchmark_limiter(chunk_samples, num_chunks)
            results.append((chunk_samples, rtf))
        
        print("\n" + "=" * 60)
        print("SUMMARY:")
        best_rtf = min(r[1] for r in results)
        
        if best_rtf < 0.1:
            print("[EXCELLENT] Limiter is >10x faster than real-time!")
        elif best_rtf < 0.5:
            print("[GOOD] Limiter is >2x faster than real-time.")
        elif best_rtf < 1.0:
            print("[OK] Limiter is faster than real-time.")
        else:
            print("[POOR] Limiter is slower than real-time!")
            print("  The vectorization still needs improvement.")
    else:
        # Example: Generate 30 minutes of pink noise with progress
        output_file = main_optimized(noise_type='pink', duration='30 mins', show_progress=True)
        if output_file:
            print(f"\nOptimized generation complete! Output: {output_file}")