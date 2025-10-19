# WispLive

Real-time voice transcription application using [faster-whisper](https://github.com/SYSTRAN/faster-whisper), a fast reimplementation of OpenAI's Whisper model powered by CTranslate2. Captures audio from your microphone and transcribes it to text with low latency using overlapping chunk processing.

## Features

- **Real-time transcription** from microphone input
- **Audio file transcription** for batch processing
- **Multiple Whisper models selection** (tiny, base, small, medium, large-v3, turbo)
- **GPU acceleration** support (CUDA)
- **VAD (Voice Activity Detection)** using Silero to detect silence and optimize performance
- **Multiple languages** supported
- **Auto-paste mode** to automatically paste transcribed text
- **System tray** integration
- **Live text viewer** with automatic refresh
- **Configurable parameters** (model size, device, compute type)
- **Portable executable** build for Windows

## Architecture

### Core Components

- **AudioCapture**: Thread-safe audio buffer with configurable sample rate and overlap handling
- **TranscriptionEngine**: Wrapper around [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for audio transcription
- **TranscriptionOrchestrator**: Manages the transcription pipeline with chunk processing and overlap resolution
- **OverlapResolver**: Handles word-level deduplication at chunk boundaries
- **DocumentWriter**: Outputs transcribed text with optional auto-paste functionality

### Processing Flow

1. Audio is captured in continuous chunks with configurable overlap (default: 5s chunks, 1s overlap)
2. Each chunk is transcribed independently using faster-whisper with Silero VAD to filter silence
3. Overlapping regions are resolved at word level using timestamp and probability scores
4. Transcribed words are written to output file and optionally pasted to active window

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for GPU acceleration)

### Setup

```bash
pip install -r requirements.txt
```

### Dependencies

- `faster_whisper==1.2.0` - Fast Whisper reimplementation using CTranslate2
- `sounddevice==0.5.1` - Audio capture
- `numpy==2.3.4` - Array processing
- `scipy==1.16.2` - Signal processing
- `pyperclip==1.9.0` - Clipboard integration
- `pyautogui==0.9.54` - Auto-paste functionality
- `pystray==0.19.5` - System tray support
- `Pillow==12.0.0` - Image processing for tray icon
- `PyInstaller==6.16.0` - Executable building
- `pytest==8.4.2` - Testing framework

## Usage

### Running the Application

```bash
python app.py
```

### Configuration

Settings are stored in `config.json`:

```json
{
  "model_size": "turbo",
  "device": "cuda",
  "compute_type": "float32",
  "language": "pt",
  "mic_id": 2,
  "should_paste_content": false
}
```

#### Configuration Options

- **model_size**: `tiny`, `base`, `small`, `medium`, `large-v3`, `turbo`
- **device**: `cpu`, `cuda`
- **compute_type**: `float32`, `float16`, `int8_float16`, `int8`
- **language**: Language code (e.g., `en`, `pt`) or `auto` for detection
- **mic_id**: Audio input device index
- **should_paste_content**: Auto-paste transcribed text when true

### GUI Controls

1. **Activate Model**: Load the selected Whisper model into memory
2. **Start Recording**: Begin real-time transcription from microphone
3. **Select Audio File**: Transcribe a WAV/MP3 file
4. **Copy Text**: Copy transcription to clipboard
5. **Paste Transcription**: Toggle auto-paste mode

### Building Executable

```bash
python build.py
```

Output: `dist/WispLive.exe`

#### Build Requirements

The build process requires copying faster-whisper assets to the executable. The `build.py` script automatically handles this by including the `--add-data` flag to copy Silero VAD model files from the faster_whisper package:

```python
site_packages = site.getsitepackages()[1]
assets_path = os.path.join(site_packages, "faster_whisper", "assets")
--add-data={assets_path};faster_whisper/assets
```

This is necessary because faster-whisper uses Silero VAD (Voice Activity Detection) to detect silence and optimize performance by skipping non-speech segments.

If building manually with PyInstaller:

```bash
pyinstaller --onefile --noconsole app.py --icon=wisp.ico \
  --add-data="C:\Users\YourName\AppData\Roaming\Python\Python312\site-packages\faster_whisper\assets;faster_whisper/assets"
```

Replace the path with your actual Python site-packages location.

## Testing

Run tests:

```bash
pytest
```

Test files:
- `test_audio_capture.py` - Audio buffer and capture tests
- `test_config_manager.py` - Configuration management tests
- `test_transcription_engine.py` - Transcription engine tests

## Project Structure

```
WispLive/
├── app/
│   ├── audio/
│   │   └── audio_capture.py            # Audio capture with buffering
│   ├── transcription/
│   │   ├── orchestrator.py             # Main transcription pipeline
│   │   ├── transcription_engine.py     # Whisper engine wrapper
│   │   ├── transcription_controller.py # High-level transcriber API
│   │   └── overlap_resolver.py         # Chunk overlap handling
│   ├── ui/
│   │   ├── main_window.py              # Main GUI window
│   │   └── live_text_view.py           # Live text display widget
│   └── utils/
│       ├── config_manager.py           # Configuration persistence
│       ├── document_writer.py          # Output file writing
│       ├── file_utils.py               # Temporary file handling
│       └── os_print.py                 # OS-specific printing
├── tests/                         # Test suite
├── app.py                         # Application entry point
├── build.py                       # PyInstaller build script
└── requirements.txt               # Python dependencies
```

## Performance Notes

- **CTranslate2 Optimization**: faster-whisper is up to 4x faster than the original OpenAI implementation while using less memory
- **VAD Optimization**: Silero VAD automatically detects and skips silent segments, reducing processing time and improving efficiency
- **Model Size**: Larger models provide better accuracy but require more VRAM and processing time
  - `tiny`: ~1GB VRAM, fastest
  - `turbo`: ~6GB VRAM, best speed/accuracy balance
  - `large-v3`: ~10GB VRAM, best accuracy
- **Compute Type**: 
  - `float32`: Best quality, slowest
  - `float16`: Good quality, requires CUDA
  - `int8`: Fastest, reduced quality
- **GPU vs CPU**: CUDA provides 5-10x speedup over CPU

## Troubleshooting

### No audio devices detected
- Check microphone permissions
- Verify device is not in use by another application
- Run `AudioCapture.get_input_devices()` to list available devices

### CUDA errors
- Verify CUDA toolkit is installed
- Check GPU compatibility with PyTorch/faster-whisper
- Try `device: "cpu"` as fallback

### Transcription quality issues
- Use larger model size for better accuracy
- Adjust `no_speech_threshold` (default: 0.6)
- Ensure clean audio input with minimal background noise

### Missing Silero VAD assets in built executable
- Ensure faster_whisper assets are copied during build
- Verify the `--add-data` path points to your actual site-packages location
- Use `build.py` which automatically handles asset copying

## License

This project uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT License), which is a reimplementation of OpenAI's Whisper model using CTranslate2. Refer to the respective licenses for usage terms.

