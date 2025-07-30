# Pink Noise Generation Optimization

## Summary of Improvements

### 1. Algorithm Selection Based on File Size
- **Small files (<1 hour)**: Use Voss-McCartney algorithm
  - Produces high-quality 1/f spectrum
  - 16 octaves for accurate low-frequency response
  - Memory-efficient incremental implementation
  
- **Large files (>1 hour)**: Use filtered white noise
  - O(N) complexity vs O(N log N) for Voss-McCartney
  - Single pass through data
  - Maintains GPU residency throughout

### 2. GPU Residency Improvements
- Replaced CPU-based `scipy.signal.lfilter` with `cupyx.scipy.signal.lfilter`
- Eliminates GPU→CPU→GPU transfers in `_apply_pinking_filter()`
- All operations stay on GPU when CuPy is available
- Graceful fallback to CPU when cupyx is not available

### 3. Implementation Details

#### Voss-McCartney (for small files)
```python
# Memory-efficient: accumulate octaves incrementally
for i in range(16):
    update_rate = 2 ** i
    octave_noise = rng.standard_normal(...)
    repeated = cp.repeat(octave_noise, update_rate, axis=0)
    pink_noise += repeated
```

#### Filtered White Noise (for large files)
```python
# Use Kellet filter for accurate 1/f response
b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
a = [1, -2.494956002, 2.017265875, -0.522189400]
filtered = cupyx.scipy.signal.lfilter(b, a, white_noise)
```

### 4. Performance Benefits
- **Large file generation**: Significantly faster for multi-hour files
- **GPU efficiency**: No unnecessary memory transfers
- **Memory usage**: Constant memory for large files (no O(log N) overhead)
- **Spectral accuracy**: Both methods produce true 1/f spectrum

### 5. Threshold Selection
- 1 hour at 48kHz = 172,800,000 samples
- This balances quality vs performance
- Can be adjusted based on available memory

### 6. Testing Recommendations
```bash
# Test both algorithms
python test_pink_noise_efficiency.py

# Verify spectral characteristics
python test_noise_consistency.py
```

## Technical Notes

1. **Unity Variance**: Both methods now start from unity variance white noise
2. **Consistent Scaling**: Final scaling of 0.1 provides ~-20 dB headroom
3. **Filter Design**: Kellet filter provides accurate 1/f response from 20Hz to 20kHz
4. **GPU Compatibility**: Works with all CUDA-capable GPUs via CuPy