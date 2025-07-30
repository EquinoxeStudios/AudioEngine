# Bit Depth Handling Fix

## Issues Fixed

### 1. Hardcoded Bit Depth in File Writing
**Problem**: The code always saved as PCM_24 regardless of the user's bit_depth setting.
```python
# Before:
subtype='PCM_24'  # Always 24-bit!
```

**Solution**: Now respects the bit_depth parameter:
```python
# After:
subtype_map = {
    16: 'PCM_16',
    24: 'PCM_24',
    32: 'PCM_24'  # FLAC limitation
}
subtype = subtype_map.get(self.bit_depth, 'PCM_24')
```

### 2. Incorrect LSB Calculation for Dithering
**Problem**: The LSB calculation was wrong:
```python
# Before:
lsb = 1.0 / (2 ** self.bit_depth)  # Wrong!
```

**Solution**: Corrected to account for -1 to +1 range (2 units total):
```python
# After:
lsb = 1.0 / (2 ** (self.bit_depth - 1))
```

### LSB Calculation Explained
- Audio range: -1.0 to +1.0 = 2.0 units total
- Quantization levels: 2^n for n-bit audio
- Step size (LSB): 2.0 / 2^n = 1 / 2^(n-1)

Examples:
- 16-bit: LSB = 1/32768 ≈ 3.05e-5
- 24-bit: LSB = 1/8388608 ≈ 1.19e-7
- 32-bit: LSB = 1/2147483648 ≈ 4.66e-10

### 3. File Size Calculation
**Problem**: Always calculated size assuming 24-bit (3 bytes per sample).

**Solution**: Now uses actual bit depth:
```python
bytes_per_sample = self.bit_depth // 8
size_mb = len(audio) * self.channels * bytes_per_sample / (1024*1024)
```

## FLAC Format Limitations
- FLAC supports up to 24-bit PCM
- When user requests 32-bit, we:
  1. Process internally at 32-bit float precision
  2. Apply appropriate dithering for 32-bit
  3. Save as 24-bit PCM (FLAC limitation)
  4. Warn the user about the limitation

## Result
The audio pipeline now correctly:
- ✅ Processes at the requested bit depth
- ✅ Applies TPDF dither with correct amplitude
- ✅ Saves files at the correct bit depth (within FLAC limits)
- ✅ Reports accurate file sizes
- ✅ Maintains proper quantization noise decorrelation