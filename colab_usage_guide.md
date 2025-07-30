# Baby Noise Engine - Google Colab Usage Guide

## Setup Instructions for Google Colab

1. **Open Google Colab**
   - Go to [Google Colab](https://colab.research.google.com)
   - Create a new notebook

2. **Enable GPU Runtime**
   - Go to Runtime → Change runtime type
   - Select GPU as Hardware accelerator
   - Click Save

3. **Install Required Dependencies**
   ```python
   # Run this cell first to install dependencies
   !pip install numpy scipy soundfile cupy-cuda11x
   ```

4. **Upload the Script**
   - Either copy the entire `baby_noise_engine.py` code into a cell
   - Or upload the file using the Files panel on the left

5. **Basic Usage Example**
   ```python
   # Import the engine
   from baby_noise_engine import main
   
   # Generate 30 minutes of white noise
   output_file = main(noise_type='white', duration='30 mins')
   
   # Generate 1 hour of pink noise
   output_file = main(noise_type='pink', duration='1 hour')
   
   # Generate 10 hours of brown noise
   output_file = main(noise_type='brown', duration='10 hours')
   ```

6. **Advanced Usage with Custom Parameters**
   ```python
   from baby_noise_engine import BabyNoiseEngine
   
   # Create custom engine instance
   engine = BabyNoiseEngine(
       noise_type='pink',
       duration_str='2 hours',
       sample_rate=48000,
       bit_depth=24
   )
   
   # Generate audio
   audio = engine.generate()
   
   # Save with custom filename
   engine.save_to_flac(audio, 'custom_baby_noise.flac')
   ```

7. **Download Generated Files**
   ```python
   # After generation, download the file
   from google.colab import files
   files.download('baby_white_noise_30_mins.flac')
   ```

## Supported Parameters

- **noise_type**: 'white', 'pink', or 'brown'
- **duration**: Any format like '30 mins', '1 hour', '6 hours', '10 hours'
- **sample_rate**: Default 48000 Hz (recommended)
- **bit_depth**: Default 24-bit (recommended for FLAC)

## Performance Tips

1. GPU acceleration will significantly speed up generation for long durations
2. For files longer than 6 hours, ensure you have sufficient Colab runtime
3. Generated files are large (~500 MB per hour at 48kHz/24-bit stereo)

## Troubleshooting

- If GPU is not available, the engine will fall back to CPU (slower but functional)
- For very long durations (>6 hours), consider splitting into multiple files
- Ensure stable internet connection for downloading large files