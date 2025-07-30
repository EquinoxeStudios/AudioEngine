# True Peak Limiting for Streaming Platforms

## Change Summary
- Changed default true peak limit from **-1.0 dBTP** to **-2.0 dBTP**
- Made true peak limit configurable in constructor
- Added detailed comments explaining the rationale

## Why -2 dBTP?

### 1. Lossy Codec Headroom
When platforms transcode to lossy formats, peaks can increase:
- **AAC (YouTube, Apple)**: +0.5 to +1.5 dB peak overshoot
- **Opus (YouTube)**: +0.5 to +1.0 dB peak overshoot  
- **MP3 (legacy)**: up to +2.0 dB peak overshoot

### 2. Platform Requirements
| Platform | True Peak Limit | LUFS Target |
|----------|----------------|-------------|
| YouTube | -2 dBTP | -14 LUFS |
| Spotify | -2 dBTP (recommended) | -14 LUFS |
| Apple Music | -1 dBTP | -16 LUFS |
| Amazon Music | -2 dBTP | -14 LUFS |

### 3. Benefits for Baby Noise
- **No clipping**: Prevents distortion after platform processing
- **Consistent levels**: Important for maintaining sleep
- **Device compatibility**: Sounds clean on all playback systems
- **Future-proof**: Ready for new codecs and platforms

## Implementation
```python
# Default conservative setting
engine = BabyNoiseEngine(
    noise_type='white',
    duration_str='8 hours',
    true_peak_limit=-2.0,  # Safe for all platforms
    target_lufs=-14.0      # YouTube/Spotify standard
)

# Custom settings if needed
engine = BabyNoiseEngine(
    noise_type='pink',
    duration_str='1 hour',
    true_peak_limit=-1.0,  # Less conservative
    target_lufs=-16.0      # Apple Music target
)
```

## Technical Details
- True peak detection uses 4x oversampling
- Lookahead limiter (5ms) prevents transient overshoot
- Anti-aliasing filter prevents artifacts
- TPDF dithering maintains audio quality

## Recommendations
1. **For YouTube**: Use -2 dBTP (default)
2. **For direct playback**: -1 dBTP is acceptable
3. **For archival**: Use -3 dBTP for maximum headroom
4. **For loudness**: Don't compensate by raising LUFS - let platforms normalize

The -2 dBTP limit ensures your baby noise audio will play cleanly on all platforms without distortion, even after multiple generations of transcoding.