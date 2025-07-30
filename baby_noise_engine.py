#!/usr/bin/env python3
"""
Baby Noise Generator - Professional Audio Engine for YouTube Content
Generates high-quality white, pink, or brown noise optimized for babies
GPU-accelerated implementation for Google Colab
"""

import numpy as np
import soundfile as sf
import scipy.signal as signal
from scipy.signal import butter, filtfilt, resample_poly, sosfilt, sosfiltfilt
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

class BabyNoiseEngine:
    """Professional audio engine for generating baby-optimized noise"""
    
    # Class constants
    SUPPORTED_SAMPLE_RATES = [44100, 48000, 88200, 96000, 192000]
    SUPPORTED_BIT_DEPTHS = [16, 24, 32]
    SUPPORTED_NOISE_TYPES = ['white', 'pink', 'brown']
    MAX_DURATION_HOURS = 12  # Safety limit
    
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
        # YouTube normalizes to -14 LUFS, but allows down to -2 dBTP after normalization
        # We use -2 dBTP to provide headroom for lossy codec encoding (AAC/Opus)
        # which can cause peak overshoots of 1-2 dB during transcoding
        self.target_lufs = target_lufs
        self.true_peak_limit = true_peak_limit
        self.oversample_factor = 4
        
        # Calculate total samples needed
        self.total_samples = int(self.duration_seconds * self.sample_rate)
        
        # Estimate memory usage and warn if high
        self._check_memory_requirements()
        
        # Initialize random number generator with instance-specific seed
        import time
        seed = int(time.time() * 1000) % 2**32  # Use current time for unique seed
        if GPU_AVAILABLE:
            self.rng = cp.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng(seed)
    
    def _validate_inputs(self, noise_type, sample_rate, bit_depth):
        """Validate constructor inputs"""
        if noise_type.lower() not in self.SUPPORTED_NOISE_TYPES:
            raise ValueError(f"Unsupported noise type: {noise_type}. Supported types: {self.SUPPORTED_NOISE_TYPES}")
        
        if sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"Unsupported sample rate: {sample_rate}. Supported rates: {self.SUPPORTED_SAMPLE_RATES}")
        
        if bit_depth not in self.SUPPORTED_BIT_DEPTHS:
            raise ValueError(f"Unsupported bit depth: {bit_depth}. Supported depths: {self.SUPPORTED_BIT_DEPTHS}")
    
    def _check_memory_requirements(self):
        """Check and warn about memory requirements"""
        # Estimate memory usage (bytes)
        # Main audio array + processing overhead (roughly 2x for temp arrays)
        bytes_per_sample = 4  # float32
        estimated_bytes = self.total_samples * self.channels * bytes_per_sample * 2
        estimated_mb = estimated_bytes / (1024 * 1024)
        estimated_gb = estimated_mb / 1024
        
        # Warn if over 4GB
        if estimated_gb > 4:
            print(f"WARNING: This will require approximately {estimated_gb:.1f} GB of RAM")
            print("Consider using a shorter duration if you experience memory issues")
        elif estimated_gb > 1:
            print(f"Note: This will require approximately {estimated_gb:.1f} GB of RAM")
        
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
                raise ValueError(f"Invalid duration format: {duration_str}. Use format like '1 hour', '30 mins', '45 secs'")
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid duration format: {duration_str}. {str(e)}")
    
    def generate_white_noise(self):
        """Generate white noise with unity variance
        
        Uses standard normal distribution (μ=0, σ=1) for mathematical consistency.
        The LUFS normalization stage will handle scaling to target loudness.
        """
        print("Generating white noise...")
        
        # Generate unit-variance Gaussian white noise
        # This gives us approximately -20 LUFS before processing
        if GPU_AVAILABLE:
            # CuPy's Generator uses standard_normal (μ=0, σ=1)
            noise = self.rng.standard_normal(size=(self.total_samples, self.channels), dtype=cp.float32)
        else:
            # NumPy: generate standard normal
            noise = self.rng.standard_normal(size=(self.total_samples, self.channels)).astype(np.float32)
        
        # Scale to reasonable level to prevent clipping in processing chain
        # Unity variance is very loud (~0 dBFS RMS), so we scale down
        # This gives us headroom for EQ and compression
        initial_scale = 0.1  # -20 dB, approximately -20 LUFS
        noise *= initial_scale
        
        # Verify statistics
        xp = cp if GPU_AVAILABLE else np
        mean = xp.mean(noise)
        std = xp.std(noise)
        
        # Only transfer scalars for printing
        if GPU_AVAILABLE:
            mean = float(mean)
            std = float(std)
            
        print(f"White noise stats - Mean: {mean:.6f}, Std: {std:.6f} (scaled from unity)")
        
        return noise
    
    def generate_pink_noise(self):
        """Generate pink noise using appropriate algorithm based on file size"""
        print("Generating pink noise...")
        
        # For large files (>1 hour at 48kHz), use filtered white noise for efficiency
        large_file_threshold = 48000 * 60 * 60  # 1 hour
        
        if self.total_samples > large_file_threshold:
            # Use filtered white noise approach for large files
            print(f"Using filtered white noise for large file ({self.total_samples/self.sample_rate/60:.1f} minutes)")
            
            # Start with white noise
            white = self.generate_white_noise()
            
            # Apply cascaded first-order IIR filters for 1/f response
            pink_noise = self._apply_efficient_pinking_filter(white)
            
        else:
            # Use Voss-McCartney for smaller files (better quality)
            # Number of octaves (rows in the Voss-McCartney algorithm)
            num_octaves = 16
            
            # Initialize accumulator for the sum instead of storing all octaves
            if GPU_AVAILABLE:
                pink_noise = cp.zeros((self.total_samples, self.channels), dtype=cp.float32)
            else:
                pink_noise = np.zeros((self.total_samples, self.channels), dtype=np.float32)
            
            # Generate and add each octave incrementally
            for i in range(num_octaves):
                # Each octave updates at different rates
                update_rate = 2 ** i
                num_updates = self.total_samples // update_rate + 1
                
                # Generate random values for this octave using standard normal
                if GPU_AVAILABLE:
                    octave_noise = self.rng.standard_normal(size=(num_updates, self.channels), dtype=cp.float32)
                else:
                    octave_noise = self.rng.standard_normal(size=(num_updates, self.channels)).astype(np.float32)
                
                # Vectorized repeat operation for both channels at once
                if GPU_AVAILABLE:
                    repeated = cp.repeat(octave_noise, update_rate, axis=0)[:self.total_samples]
                else:
                    repeated = np.repeat(octave_noise, update_rate, axis=0)[:self.total_samples]
                pink_noise += repeated
            
            # Normalize after summing all octaves
            if GPU_AVAILABLE:
                pink_noise = pink_noise / cp.sqrt(num_octaves)
            else:
                pink_noise = pink_noise / np.sqrt(num_octaves)
            
            # Apply additional 1/f filter for better spectral characteristics
            pink_noise = self._apply_pinking_filter(pink_noise)
            
            # Scale to consistent initial level (same as white noise)
            # Pink noise has different power distribution but similar RMS
            pink_noise *= 0.1
        
        return pink_noise
    
    def _apply_efficient_pinking_filter(self, white_noise):
        """Apply efficient cascaded IIR filters for 1/f response on large files"""
        # Use optimized pinking filter based on Julius O. Smith's design
        # This produces accurate 1/f spectrum using cascaded filters
        
        if GPU_AVAILABLE:
            try:
                # Keep everything on GPU
                import cupyx.scipy.signal as cp_signal
                
                # Apply the standard Kellet pinking filter on GPU
                # This is more accurate than cascaded first-order filters
                b = cp.array([0.049922035, -0.095993537, 0.050612699, -0.004408786], dtype=cp.float32)
                a = cp.array([1, -2.494956002, 2.017265875, -0.522189400], dtype=cp.float32)
                
                filtered = cp.zeros_like(white_noise)
                for ch in range(self.channels):
                    filtered[:, ch] = cp_signal.lfilter(b, a, white_noise[:, ch])
                
                # Scale to match expected output level
                # The filter has approximately 0.1 gain at low frequencies
                filtered = filtered * 10.0 * 0.1  # Scale back to ~0.1 RMS
                
                return filtered
                
            except ImportError:
                # cupyx not available, must use CPU
                print("cupyx.scipy not available, using CPU filtering...")
                noise_cpu = white_noise.get()
                return cp.asarray(self._apply_efficient_pinking_filter_cpu(noise_cpu))
        else:
            return self._apply_efficient_pinking_filter_cpu(white_noise)
    
    def _apply_efficient_pinking_filter_cpu(self, white_noise):
        """CPU version of efficient pinking filter"""
        # Use the same Kellet filter as GPU version for consistency
        filtered = white_noise.copy()
        
        # Kellet pinking filter coefficients
        b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786], dtype=np.float32)
        a = np.array([1, -2.494956002, 2.017265875, -0.522189400], dtype=np.float32)
        
        # Apply filter to each channel
        for ch in range(self.channels):
            filtered[:, ch] = signal.lfilter(b, a, filtered[:, ch])
        
        # Scale to match expected output level
        filtered = filtered * 10.0 * 0.1  # Scale back to ~0.1 RMS
        
        return filtered
    
    def _apply_pinking_filter(self, noise):
        """Apply additional pinking filter for accurate 1/f spectrum"""
        # Pinking filter coefficients (Paul Kellet's method)
        if GPU_AVAILABLE:
            b = cp.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
            a = cp.array([1, -2.494956002, 2.017265875, -0.522189400])
        else:
            b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
            a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
        
        # Apply filter to each channel
        if GPU_AVAILABLE:
            try:
                # Keep data on GPU using cupyx
                import cupyx.scipy.signal as cp_signal
                filtered = cp.zeros_like(noise)
                for ch in range(self.channels):
                    filtered[:, ch] = cp_signal.lfilter(b, a, noise[:, ch])
                return filtered
            except ImportError:
                # cupyx not available, must transfer to CPU
                print("cupyx.scipy not available, using CPU filtering...")
                noise_cpu = noise.get()
                filtered = np.zeros_like(noise_cpu)
                b_cpu = b.get() if hasattr(b, 'get') else b
                a_cpu = a.get() if hasattr(a, 'get') else a
                for ch in range(self.channels):
                    filtered[:, ch] = signal.lfilter(b_cpu, a_cpu, noise_cpu[:, ch])
                return cp.asarray(filtered)
        else:
            filtered = np.zeros_like(noise)
            for ch in range(self.channels):
                filtered[:, ch] = signal.lfilter(b, a, noise[:, ch])
            return filtered
    
    def generate_brown_noise(self):
        """Generate brown noise using stable 1/f² filter"""
        print("Generating brown noise...")
        
        # Start with white noise
        white = self.generate_white_noise()
        
        # Design a stable brownian noise filter (1/f² response)
        # Using a series of first-order low-pass filters
        # H(z) = 0.1 / (1 - 0.9 * z^-1) gives approximately -20dB/decade slope
        
        if GPU_AVAILABLE:
            try:
                # Try to use CuPy's signal processing functions to stay on GPU
                from cupyx.scipy import signal as cp_signal
                
                brown_noise = cp.zeros_like(white, dtype=cp.float32)
                
                # Apply brownian filter to each channel on GPU
                for ch in range(self.channels):
                    # Work in float64 for precision
                    ch_data = white[:, ch].astype(cp.float64)
                    
                    # First-order IIR filter coefficients for 1/f² response
                    b = cp.array([0.1], dtype=cp.float64)
                    a = cp.array([1.0, -0.9], dtype=cp.float64)
                    
                    # Apply filter on GPU
                    filtered = cp_signal.lfilter(b, a, ch_data)
                    
                    # Apply DC removal high-pass filter on GPU
                    sos = cp_signal.butter(4, 2.0 / (self.sample_rate / 2), btype='high', output='sos')
                    filtered = cp_signal.sosfiltfilt(sos, filtered)
                    
                    # Convert back to float32
                    brown_noise[:, ch] = filtered.astype(cp.float32)
                
                # Normalize to prevent clipping (on GPU)
                max_val = cp.max(cp.abs(brown_noise))
                if max_val > 0:
                    brown_noise = brown_noise / max_val * 0.5
                    
            except ImportError:
                # Fallback to CPU if cupyx.scipy.signal not available
                white_cpu = white.get()
                brown_cpu = np.zeros_like(white_cpu, dtype=np.float32)
                
                # Apply brownian filter to each channel
                for ch in range(self.channels):
                    # Convert to float64 for filtering
                    ch_data = white_cpu[:, ch].astype(np.float64)
                    
                    # First-order IIR filter coefficients for 1/f² response
                    b = np.array([0.1], dtype=np.float64)
                    a = np.array([1.0, -0.9], dtype=np.float64)
                    
                    # Apply filter in float64
                    filtered = signal.lfilter(b, a, ch_data)
                    
                    # Apply DC removal high-pass filter
                    sos = butter(4, 2.0 / (self.sample_rate / 2), btype='high', output='sos')
                    filtered = sosfiltfilt(sos, filtered)
                    
                    # Convert back to float32
                    brown_cpu[:, ch] = filtered.astype(np.float32)
                
                # Normalize to prevent clipping
                max_val = np.max(np.abs(brown_cpu))
                if max_val > 0:
                    brown_cpu = brown_cpu / max_val * 0.5
                    
                brown_noise = cp.asarray(brown_cpu)
        else:
            brown_noise = np.zeros_like(white, dtype=np.float32)
            
            # Apply brownian filter to each channel
            for ch in range(self.channels):
                # Convert to float64 for filtering
                ch_data = white[:, ch].astype(np.float64)
                
                # First-order IIR filter coefficients for 1/f² response
                b = np.array([0.1], dtype=np.float64)
                a = np.array([1.0, -0.9], dtype=np.float64)
                
                # Apply filter in float64
                filtered = signal.lfilter(b, a, ch_data)
                
                # Apply DC removal high-pass filter
                sos = butter(4, 2.0 / (self.sample_rate / 2), btype='high', output='sos')
                filtered = sosfiltfilt(sos, filtered)
                
                # Convert back to float32
                brown_noise[:, ch] = filtered.astype(np.float32)
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(brown_noise))
            if max_val > 0:
                brown_noise = brown_noise / max_val * 0.5
        
        # Verify 1/f² spectral density
        if self.total_samples > 10000:  # Only for longer files
            self._verify_spectral_density(brown_noise, expected_slope=-2)
        
        return brown_noise
    
    def _verify_spectral_density(self, noise, expected_slope):
        """Verify the spectral density follows expected power law"""
        # Take a sample for analysis (first channel, middle portion)
        sample_size = min(8192, self.total_samples // 2)
        start_idx = self.total_samples // 4
        
        # Need CPU for spectral analysis
        if GPU_AVAILABLE:
            sample = noise[start_idx:start_idx+sample_size, 0].get()
        else:
            sample = noise[start_idx:start_idx+sample_size, 0]
        
        # Compute power spectral density
        freqs, psd = signal.welch(sample, fs=self.sample_rate, nperseg=1024)
        
        # Fit log-log slope (ignore DC)
        valid_idx = freqs > 10
        log_freqs = np.log10(freqs[valid_idx])
        log_psd = np.log10(psd[valid_idx])
        slope = np.polyfit(log_freqs, log_psd, 1)[0]
        
        print(f"Spectral slope: {slope:.2f} (expected: {expected_slope})")
    
    
    def apply_infant_eq(self, audio):
        """Apply EQ optimized for infant comfort"""
        print("Applying infant-optimized EQ...")
        
        nyquist = self.sample_rate / 2
        
        # Gentle low shelf boost below 200 Hz
        shelf_freq = 200 / nyquist
        sos_shelf = self._design_shelf_filter(shelf_freq, gain_db=3, filter_type='low')
        
        # Notch filter to reduce harshness around 2-5 kHz
        notch_freq = 3500 / nyquist
        q_factor = 2
        b_notch, a_notch = signal.iirnotch(notch_freq, q_factor)
        sos_notch = signal.tf2sos(b_notch, a_notch)
        
        # Apply filters
        if GPU_AVAILABLE:
            try:
                # Try to use GPU-based filtering
                from cupyx.scipy import signal as cp_signal
                filtered = audio.copy()
                for ch in range(self.channels):
                    filtered[:, ch] = cp_signal.sosfilt(sos_shelf, filtered[:, ch])
                    filtered[:, ch] = cp_signal.sosfilt(sos_notch, filtered[:, ch])
                return filtered
            except (ImportError, AttributeError):
                # Fallback to CPU if cupyx not available
                audio_cpu = audio.get()
                filtered = audio_cpu.copy()
                for ch in range(self.channels):
                    filtered[:, ch] = sosfilt(sos_shelf, filtered[:, ch])
                    filtered[:, ch] = sosfilt(sos_notch, filtered[:, ch])
                return cp.asarray(filtered)
        else:
            filtered = audio.copy()
            for ch in range(self.channels):
                filtered[:, ch] = sosfilt(sos_shelf, filtered[:, ch])
                filtered[:, ch] = sosfilt(sos_notch, filtered[:, ch])
            return filtered
    
    def _design_shelf_filter(self, freq, gain_db, filter_type='low'):
        """Design a shelf filter"""
        # Simplified shelf filter using Butterworth approximation
        sos = butter(2, freq, btype='low' if filter_type == 'low' else 'high', output='sos')
        
        # Apply gain adjustment (simplified)
        gain_linear = 10 ** (gain_db / 20)
        if filter_type == 'low':
            sos[0, :3] *= gain_linear  # Boost numerator coefficients
        
        return sos
    
    def apply_compression(self, audio, threshold_db=-20, ratio=2, attack_ms=10, release_ms=100):
        """Apply gentle compression to control dynamics"""
        print("Applying dynamics compression...")
        
        # Convert parameters
        threshold_linear = 10 ** (threshold_db / 20)
        attack_samples = int(attack_ms * self.sample_rate / 1000)
        release_samples = int(release_ms * self.sample_rate / 1000)
        
        # Work on GPU if available, otherwise CPU
        xp = cp if GPU_AVAILABLE else np
        compressed = xp.zeros_like(audio)
        
        for ch in range(self.channels):
            signal_ch = audio[:, ch]
            
            # Compute envelope
            envelope = xp.abs(signal_ch)
            
            # Vectorized envelope smoothing using exponential filters
            alpha_attack = 1 - xp.exp(-1 / attack_samples)
            alpha_release = 1 - xp.exp(-1 / release_samples)
            
            if GPU_AVAILABLE:
                try:
                    # Try GPU-based filtering
                    from cupyx.scipy import signal as cp_signal
                    
                    # First pass: attack envelope (fast rise)
                    b_attack = xp.array([alpha_attack])
                    a_attack = xp.array([1, -(1-alpha_attack)])
                    attack_envelope = cp_signal.lfilter(b_attack, a_attack, envelope)
                    
                    # Second pass: release envelope (slow fall)
                    b_release = xp.array([alpha_release])
                    a_release = xp.array([1, -(1-alpha_release)])
                    release_envelope = cp_signal.lfilter(b_release, a_release, envelope)
                    
                except (ImportError, AttributeError):
                    # Fallback: process on CPU then move back
                    envelope_cpu = envelope.get()
                    
                    b_attack = np.array([alpha_attack.get() if hasattr(alpha_attack, 'get') else alpha_attack])
                    a_attack = np.array([1, -(1-b_attack[0])])
                    attack_envelope_cpu = signal.lfilter(b_attack, a_attack, envelope_cpu)
                    
                    b_release = np.array([alpha_release.get() if hasattr(alpha_release, 'get') else alpha_release])
                    a_release = np.array([1, -(1-b_release[0])])
                    release_envelope_cpu = signal.lfilter(b_release, a_release, envelope_cpu)
                    
                    attack_envelope = cp.asarray(attack_envelope_cpu)
                    release_envelope = cp.asarray(release_envelope_cpu)
            else:
                # CPU path
                b_attack = np.array([alpha_attack])
                a_attack = np.array([1, -(1-alpha_attack)])
                attack_envelope = signal.lfilter(b_attack, a_attack, envelope)
                
                b_release = np.array([alpha_release])
                a_release = np.array([1, -(1-alpha_release)])
                release_envelope = signal.lfilter(b_release, a_release, envelope)
            
            # Combine: use attack when signal is rising, release when falling
            smoothed_envelope = xp.maximum(envelope, release_envelope)
            
            # Apply attack characteristics where signal is rising
            rising = xp.diff(envelope, prepend=envelope[0]) > 0
            smoothed_envelope[rising] = attack_envelope[rising]
            
            # Compute gain reduction
            gain = xp.ones_like(smoothed_envelope)
            above_threshold = smoothed_envelope > threshold_linear
            gain[above_threshold] = (threshold_linear + (smoothed_envelope[above_threshold] - threshold_linear) / ratio) / smoothed_envelope[above_threshold]
            
            # Apply gain
            compressed[:, ch] = signal_ch * gain
        
        return compressed
    
    def apply_true_peak_limiting(self, audio):
        """Apply true peak limiting with lookahead and 4x oversampling"""
        print("Applying true peak limiting with lookahead...")
        
        # Lookahead parameters
        lookahead_ms = 5.0  # 5ms lookahead
        lookahead_samples = int(lookahead_ms * self.sample_rate / 1000)
        
        # Attack and release for the limiter (in samples)
        attack_samples = int(0.5 * self.sample_rate / 1000)  # 0.5ms attack
        release_samples = int(50 * self.sample_rate / 1000)   # 50ms release
        
        # Process on CPU for now (scipy functions required)
        # Single transfer at the beginning instead of multiple transfers
        if GPU_AVAILABLE:
            audio_cpu = audio.get()
        else:
            audio_cpu = audio
            
        limited = audio_cpu.copy()
            
        for ch in range(self.channels):
            # Oversample for true peak detection
            oversampled = resample_poly(audio_cpu[:, ch], self.oversample_factor, 1)
            
            # Detect peaks with lookahead
            peak_envelope = np.zeros(len(oversampled))
            
            # Use maximum filter for lookahead
            from scipy.ndimage import maximum_filter1d
            lookahead_samples_os = lookahead_samples * self.oversample_factor
            peak_envelope = maximum_filter1d(np.abs(oversampled), size=lookahead_samples_os, mode='constant')
            
            # Convert to dB
            peak_db = 20 * np.log10(peak_envelope + 1e-10)
            
            # Calculate required gain reduction
            gain_reduction_db = np.minimum(0, self.true_peak_limit - peak_db)
            target_gain = 10 ** (gain_reduction_db / 20)
            
            # Smooth the gain reduction (attack/release)
            # Fast attack, slow release
            gain_smooth = np.ones_like(target_gain)
            
            # Apply attack (fast decrease in gain)
            alpha_attack = 1 - np.exp(-1 / (attack_samples * self.oversample_factor))
            
            # Apply release (slow increase in gain)
            alpha_release = 1 - np.exp(-1 / (release_samples * self.oversample_factor))
            
            # Vectorized gain smoothing
            # Create separate attack and release filtered versions
            b_attack = np.array([alpha_attack])
            a_attack = np.array([1, -(1 - alpha_attack)])
            gain_attack = signal.lfilter(b_attack, a_attack, target_gain)
            
            b_release = np.array([alpha_release])
            a_release = np.array([1, -(1 - alpha_release)])
            gain_release = signal.lfilter(b_release, a_release, target_gain)
            
            # Use attack when gain is decreasing, release when increasing
            gain_diff = np.diff(target_gain, prepend=target_gain[0])
            gain_smooth = np.where(gain_diff < 0, gain_attack, gain_release)
            
            # Apply gain to oversampled signal
            limited_oversampled = oversampled * gain_smooth
            
            # Apply anti-aliasing filter before downsampling
            # Design anti-aliasing filter (8th order Butterworth)
            nyquist = self.sample_rate / 2
            cutoff = nyquist * 0.9  # 90% of Nyquist for safety margin
            # Cutoff frequency for oversampled rate
            cutoff_normalized = cutoff / (self.sample_rate * self.oversample_factor / 2)
            
            # Design and apply anti-aliasing filter
            sos_aa = butter(8, cutoff_normalized, btype='low', output='sos')
            limited_oversampled = sosfiltfilt(sos_aa, limited_oversampled)
            
            # Downsample back with anti-aliasing applied
            downsampled = resample_poly(limited_oversampled, 1, self.oversample_factor)
            
            # Ensure exact length match
            if len(downsampled) >= len(audio_cpu):
                limited[:, ch] = downsampled[:len(audio_cpu)]
            else:
                limited[:len(downsampled), ch] = downsampled
        
        if GPU_AVAILABLE:
            return cp.asarray(limited)
        else:
            return limited
    
    def apply_fade(self, audio, fade_duration_sec=5):
        """Apply fade in/out to prevent startle response"""
        print("Applying fade in/out...")
        
        fade_samples = int(fade_duration_sec * self.sample_rate)
        audio_length = len(audio)
        
        # Limit fade length to half the audio length at most
        fade_samples = min(fade_samples, audio_length // 2)
        
        if fade_samples < 1:
            return audio  # Audio too short for fades
        
        if GPU_AVAILABLE:
            # Create fade curves
            fade_in = cp.linspace(0, 1, fade_samples) ** 2  # Quadratic fade
            fade_out = cp.linspace(1, 0, fade_samples) ** 2
            
            # Apply fades
            audio[:fade_samples] *= fade_in[:, cp.newaxis]
            audio[-fade_samples:] *= fade_out[:, cp.newaxis]
        else:
            # Create fade curves
            fade_in = np.linspace(0, 1, fade_samples) ** 2
            fade_out = np.linspace(1, 0, fade_samples) ** 2
            
            # Apply fades
            audio[:fade_samples] *= fade_in[:, np.newaxis]
            audio[-fade_samples:] *= fade_out[:, np.newaxis]
        
        return audio
    
    def measure_lufs(self, audio):
        """Measure integrated LUFS according to ITU-R BS.1770-4 with gating"""
        # LUFS measurement requires CPU processing (scipy filters)
        if GPU_AVAILABLE:
            audio_cpu = audio.get()
        else:
            audio_cpu = audio
        
        # K-weighting filter coefficients (BS.1770-4)
        # Stage 1: High shelf filter (f0 = 1681.97 Hz, G = +3.99984 dB, Q = 0.7079)
        # Stage 2: High-pass filter (f0 = 38.13547 Hz, Q = 0.5003, G = 0 dB)
        
        # Pre-filter (high shelf) - updated coefficients for BS.1770-4
        pre_b = np.array([1.53260026327012, -2.65041135748730, 1.16904917595255])
        pre_a = np.array([1.0, -1.66375098226575, 0.71265752994786])
        
        # RLB filter (high-pass)
        rlb_b = np.array([1.0, -2.0, 1.0])
        rlb_a = np.array([1.0, -1.98998479513207, 0.98999499812227])
        
        # Apply K-weighting - use float64 for precision
        k_weighted = np.zeros_like(audio_cpu, dtype=np.float64)
        for ch in range(self.channels):
            # Convert to float64 for filtering
            ch_data = audio_cpu[:, ch].astype(np.float64)
            # Apply pre-filter
            temp = signal.lfilter(pre_b, pre_a, ch_data)
            # Apply RLB filter
            k_weighted[:, ch] = signal.lfilter(rlb_b, rlb_a, temp)
        
        # Measure in 400ms blocks with 75% overlap for gating
        block_size = int(0.4 * self.sample_rate)  # 400ms
        hop_size = int(0.1 * self.sample_rate)    # 100ms (75% overlap)
        
        # Use sliding window for efficient block processing
        if len(k_weighted) < block_size:
            return -70.0  # File too short
        
        # Create sliding window view for efficient computation
        from numpy.lib.stride_tricks import sliding_window_view
        
        # For stereo, we need to handle both channels together
        # Reshape to (samples, channels) if needed
        if k_weighted.ndim == 1:
            k_weighted = k_weighted.reshape(-1, 1)
        
        # Calculate mean square for all blocks at once using sliding window
        try:
            # For NumPy >= 1.20, use sliding_window_view for maximum efficiency
            windows = sliding_window_view(k_weighted, window_shape=(block_size,), axis=0)[::hop_size]
            
            # Calculate mean square power for all blocks at once
            # Shape: (num_blocks, block_size, channels)
            # Mean across block_size and channels axes
            mean_squares = np.mean(windows ** 2, axis=(1, 2) if k_weighted.ndim > 1 else 1)
            
            # Vectorized loudness calculation
            # Avoid log10(0) by using maximum with small epsilon
            block_loudness = -0.691 + 10 * np.log10(np.maximum(mean_squares, 1e-10))
            
        except (ImportError, AttributeError):
            # Fallback for older NumPy versions
            num_blocks = (len(k_weighted) - block_size) // hop_size + 1
            
            # Pre-allocate and vectorize as much as possible
            block_loudness = np.full(num_blocks, -70.0)
            
            # Compute squared k_weighted once
            k_weighted_squared = k_weighted ** 2
            
            # Use cumulative sum for efficient windowed mean calculation
            cumsum = np.insert(np.cumsum(k_weighted_squared, axis=0), 0, 0, axis=0)
            
            for i in range(num_blocks):
                start = i * hop_size
                end = start + block_size
                
                # Calculate mean using cumsum difference
                if k_weighted.ndim > 1:
                    block_sum = cumsum[end] - cumsum[start]
                    mean_square = np.mean(block_sum) / block_size
                else:
                    mean_square = (cumsum[end] - cumsum[start]) / block_size
                
                if mean_square > 0:
                    block_loudness[i] = -0.691 + 10 * np.log10(mean_square)
        
        # Remove invalid blocks
        block_loudness = block_loudness[block_loudness > -70]
        
        if len(block_loudness) == 0:
            return -70.0
        
        # Absolute gate at -70 LUFS
        gated_blocks = block_loudness[block_loudness > -70]
        
        if len(gated_blocks) == 0:
            return -70.0
            
        # Calculate relative gate threshold (10 LU below ungated mean)
        ungated_mean = -0.691 + 10 * np.log10(np.mean(10 ** ((gated_blocks + 0.691) / 10)))
        relative_gate = ungated_mean - 10
        
        # Apply relative gate
        final_gated = gated_blocks[gated_blocks > relative_gate]
        
        if len(final_gated) == 0:
            return -70.0
            
        # Calculate gated loudness
        gated_power = np.mean(10 ** ((final_gated + 0.691) / 10))
        integrated_lufs = -0.691 + 10 * np.log10(gated_power)
        
        return integrated_lufs
    
    def normalize_to_lufs(self, audio, target_lufs=-14):
        """Normalize audio to target LUFS"""
        current_lufs = self.measure_lufs(audio)
        gain_db = target_lufs - current_lufs
        gain_linear = 10 ** (gain_db / 20)
        
        print(f"Current LUFS: {current_lufs:.1f}, applying gain: {gain_db:.1f} dB")
        
        return audio * gain_linear
    
    def apply_dithering(self, audio):
        """Mark audio as needing dither (applied during save)"""
        print("TPDF dither will be applied during save...")
        # Just return the audio as-is; dither will be applied in chunks during save
        return audio
    
    def generate(self, progress_callback=None):
        """Main generation pipeline with progress reporting
        
        Args:
            progress_callback: Optional function(stage, percent) to report progress
        """
        def report_progress(stage, percent):
            if progress_callback:
                progress_callback(stage, percent)
            
        print(f"\nGenerating {self.duration_seconds/60:.1f} minutes of {self.noise_type} noise...")
        print(f"Sample rate: {self.sample_rate} Hz, Bit depth: {self.bit_depth}-bit")
        
        try:
            # Generate base noise
            report_progress("Generating noise", 0)
            if self.noise_type == 'white':
                audio = self.generate_white_noise()
            elif self.noise_type == 'pink':
                audio = self.generate_pink_noise()
            elif self.noise_type == 'brown':
                audio = self.generate_brown_noise()
            else:
                raise ValueError(f"Unknown noise type: {self.noise_type}")
            report_progress("Generating noise", 100)
            
            # Apply processing chain with progress updates
            report_progress("Applying EQ", 0)
            audio = self.apply_infant_eq(audio)
            report_progress("Applying compression", 20)
            audio = self.apply_compression(audio)
            report_progress("Normalizing loudness", 40)
            audio = self.normalize_to_lufs(audio, self.target_lufs)
            report_progress("Limiting peaks", 60)
            audio = self.apply_true_peak_limiting(audio)
            report_progress("Applying fades", 80)
            audio = self.apply_fade(audio)
            report_progress("Preparing output", 90)
            audio = self.apply_dithering(audio)
            
            # Final LUFS check
            final_lufs = self.measure_lufs(audio)
            print(f"\nFinal LUFS: {final_lufs:.1f}")
            
            # Convert to CPU array if on GPU
            if GPU_AVAILABLE:
                audio = audio.get()
            
            report_progress("Complete", 100)
            return audio
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            raise
    
    def save_to_flac(self, audio, filename):
        """Save audio to FLAC with metadata and chunked dithering"""
        import os
        
        # Sanitize filename to prevent path traversal
        filename = os.path.basename(filename)
        if not filename.endswith('.flac'):
            filename += '.flac'
            
        # Validate filename
        if not filename or filename.startswith('.'):
            raise ValueError("Invalid filename")
            
        print(f"\nSaving to {filename} with TPDF dither...")
        
        # Ensure audio is in correct range
        audio = np.clip(audio, -1.0, 1.0)
        
        # Apply TPDF dither in chunks during write
        chunk_size = 1024 * 1024  # 1M samples per chunk
        
        # Correct LSB calculation for dithering
        # For n-bit audio: LSB = 2 / (2^n) = 1 / (2^(n-1))
        # This is because audio range is -1 to +1 (2 units total)
        lsb = 1.0 / (2 ** (self.bit_depth - 1))
        
        # Use NumPy RNG for dithering (always CPU operation during file write)
        np_rng = np.random.default_rng(42)
        
        # Determine the correct subtype based on bit depth
        # Note: FLAC supports up to 24-bit PCM
        subtype_map = {
            16: 'PCM_16',
            24: 'PCM_24',
            32: 'PCM_24'  # FLAC doesn't support 32-bit, use 24-bit
        }
        subtype = subtype_map.get(self.bit_depth, 'PCM_24')
        
        # Warn if downsampling from 32-bit
        if self.bit_depth == 32:
            print("Note: FLAC format limited to 24-bit, will use 24-bit PCM")
        
        # Open file for writing with correct bit depth
        with sf.SoundFile(filename, 'w', samplerate=self.sample_rate, 
                         channels=self.channels, subtype=subtype, format='FLAC') as f:
            
            # Process in chunks to save memory
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                
                # Generate TPDF dither for this chunk
                if GPU_AVAILABLE and isinstance(chunk, cp.ndarray):
                    chunk = chunk.get()  # Convert to numpy for soundfile
                
                # Create TPDF dither by summing two RPDF sources
                rpdf1 = np_rng.uniform(-0.5, 0.5, size=chunk.shape)
                rpdf2 = np_rng.uniform(-0.5, 0.5, size=chunk.shape)
                tpdf_dither = (rpdf1 + rpdf2) * lsb
                
                # Apply dither and write
                dithered_chunk = chunk + tpdf_dither
                dithered_chunk = np.clip(dithered_chunk, -1.0, 1.0)
                f.write(dithered_chunk)
        
        # Now add metadata using mutagen
        try:
            from mutagen.flac import FLAC
            
            # Open the FLAC file
            audio_file = FLAC(filename)
            
            # Clear existing tags
            audio_file.clear()
            
            # Add metadata
            audio_file['TITLE'] = f'{self.noise_type.capitalize()} Noise for Babies'
            audio_file['ARTIST'] = 'Baby Noise Engine'
            audio_file['ALBUM'] = 'Sleep Sounds for Infants'
            audio_file['DATE'] = '2024'
            audio_file['GENRE'] = 'Ambient'
            audio_file['COMMENT'] = f'Duration: {self.duration_seconds/60:.1f} min, LUFS: {self.target_lufs}, Type: {self.noise_type}'
            audio_file['ENCODER'] = 'Baby Noise Engine v1.0'
            
            # Save the metadata
            audio_file.save()
            print("Metadata embedded successfully!")
            
        except ImportError:
            print("Warning: mutagen not installed, metadata not embedded")
            print("Install with: pip install mutagen")
        except Exception as e:
            print(f"Warning: Could not embed metadata: {e}")
        
        print(f"Audio saved successfully!")
        print(f"File: {filename}")
        print(f"Duration: {self.duration_seconds/60:.1f} minutes")
        print(f"Bit depth: {self.bit_depth}-bit")
        # Calculate actual file size based on bit depth
        bytes_per_sample = self.bit_depth // 8
        print(f"Size: ~{len(audio) * self.channels * bytes_per_sample / (1024*1024):.1f} MB")


def main(noise_type='white', duration='1 hour', show_progress=False):
    """Main function to generate baby noise
    
    Args:
        noise_type: Type of noise ('white', 'pink', 'brown')
        duration: Duration string (e.g., '1 hour', '30 mins')
        show_progress: Whether to show progress updates
    """
    
    # Progress callback example
    def progress_callback(stage, percent):
        if show_progress:
            print(f"[{stage}] {percent}% complete")
    
    try:
        # Create engine instance
        engine = BabyNoiseEngine(noise_type=noise_type, duration_str=duration)
        
        # Generate audio with progress callback
        audio = engine.generate(progress_callback if show_progress else None)
        
        # Save to file
        filename = f"baby_{noise_type}_noise_{duration.replace(' ', '_')}.flac"
        engine.save_to_flac(audio, filename)
        
        return filename
        
    except ValueError as e:
        print(f"Input error: {e}")
        return None
    except MemoryError:
        print(f"Out of memory! Try a shorter duration or close other applications.")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Example: Generate 30 minutes of pink noise
    output_file = main(noise_type='pink', duration='30 mins')
    print(f"\nGeneration complete! Output: {output_file}") 