# Anti-Aliasing Implementation Notes

## Problem
The true peak limiter was using 4x oversampling but lacked proper anti-aliasing before downsampling. This could introduce aliasing artifacts (frequencies above Nyquist folding back into the audible range).

## Solution
Added an 8th-order Butterworth low-pass filter before downsampling:

```python
# Anti-aliasing filter design
nyquist = self.sample_rate / 2
cutoff = nyquist * 0.9  # 90% of Nyquist for safety margin
cutoff_normalized = cutoff / (self.sample_rate * self.oversample_factor / 2)

# Apply filter
sos_aa = butter(8, cutoff_normalized, btype='low', output='sos')
limited_oversampled = sosfiltfilt(sos_aa, limited_oversampled)
```

## Technical Details

### Why Anti-Aliasing is Needed
1. **Oversampling**: We upsample to 4x (192 kHz for 48 kHz audio) to accurately detect true peaks
2. **Processing**: Gain reduction can create harmonics above the original Nyquist frequency
3. **Downsampling**: Without filtering, frequencies above 24 kHz would alias into the audible range

### Filter Specifications
- **Type**: Butterworth (maximally flat passband)
- **Order**: 8th order (48 dB/octave rolloff)
- **Cutoff**: 21.6 kHz (90% of Nyquist at 48 kHz)
- **Implementation**: SOS (Second-Order Sections) for numerical stability
- **Phase**: Zero-phase (using sosfiltfilt for forward-backward filtering)

### Performance Impact
- Minimal: The filter is only applied once per channel after limiting
- The 8th-order filter provides excellent stopband attenuation
- SOS implementation is numerically stable and efficient

## Dithering Notes

The TPDF (Triangular Probability Density Function) dithering is correctly implemented:
- Applied during file writing to save memory
- Proper amplitude: ±1 LSB total
- Created by summing two rectangular distributions
- Chunked application for memory efficiency

## Result
The audio pipeline now properly handles:
1. ✅ Oversampling for true peak detection
2. ✅ Anti-aliasing before downsampling
3. ✅ TPDF dithering for bit depth reduction
4. ✅ All processing at appropriate precision levels

This ensures pristine audio quality without aliasing artifacts.