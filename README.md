
# AutoDub - Automatic Video Translation & Dubbing

Automatically transcribe, translate, and dub videos into different languages using AI-powered text-to-speech.

## Features

- 🎙️ **Speech Recognition**: Transcribe audio using OpenAI Whisper
- 🌍 **Translation**: Translate to 100+ languages via Google Translate
- 🗣️ **Three TTS Engines**:
    - **Edge TTS**: High-quality Microsoft voices (recommended)
    - **Silero**: Fast Russian TTS (offline after first download)
    - **XTTS**: Voice cloning from 6-10 second samples
- 🎬 **Video Preservation**: Keeps original video, mixes original audio (20%) with dubbed audio (150%)
- 📝 **Subtitle Generation**: Creates SRT files for translated text

## Requirements

### System Dependencies
```bash
# Fedora/RHEL
sudo dnf install ffmpeg python3.10 python3.10-devel

# Ubuntu/Debian
sudo apt install ffmpeg python3.10 python3.10-devel

# macOS
brew install ffmpeg
```

### Python Dependencies
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install openai-whisper pysrt edge-tts deep-translator soundfile tqdm
pip install TTS  # Only needed for XTTS voice cloning
```

##  Local Translation with Ollama (Optional)

AutoDub now supports fully offline translation using **Ollama**. This is ideal for privacy, avoiding API limits, and achieving more context-aware translations.

### 1. Install Ollama
**For Linux (Fedora/Ubuntu/etc.):**
```bash
curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh
ollama pull llama3
````

## Quick Start

Here is the concise guide on how to get started using your setup.sh script, formatted in Markdown:
🚀 Quick Start Guide

Follow these three steps to set up and start dubbing your videos:
1. Prepare Files

Ensure you have the following files in your project directory:

    setup.sh (The installer)

    autodub_v4_1.py (The main engine)

    install.txt (List of dependencies)

2. Run Installation

Open your terminal in the project folder and execute:
```Bash

chmod +x setup.sh && ./setup.sh
````

### Basic Usage (Edge TTS - Recommended)
```bash
# Dub to Russian (default)
python autodub.py video.mp4

# Dub to English
python autodub.py video.mp4 --target_lang en

# Dub to German
python autodub.py video.mp4 --target_lang de
```

### Silero TTS (Faster, Russian only)
```bash
# Default voice (aidar)
python autodub.py video.mp4 --tts silero

# Female voice
python autodub.py video.mp4 --tts silero --silero_voice xenia

# Available voices: aidar, baya, kseniya, xenia, eugene
```
### XTTS Voice Cloning (Most Natural)
```bash
# Requires 6-10 second clean voice sample
python autodub.py video.mp4 --tts xtts --ref_voice my_voice.wav --target_lang en
```

### Ollama translator

```bash
# Use Ollama with default llama3 model
./run.sh video.mp4 --translator ollama

# Use a specific model (e.g., Mistral)
./run.sh video.mp4 --translator ollama --ollama_model mistral 
```

## Command-Line Options
```
positional arguments:
  video                 Input video file

options:
  -h, --help            Show help message
  --tts {edge,silero,xtts}
                        TTS engine (default: edge)
  --target_lang LANG    Target language code (default: ru)
                        Supports: ru, en, de, fr, es, it, pt, ja, zh, etc.
  --silero_voice {aidar,baya,kseniya,xenia,eugene}
                        Silero voice for Russian (default: aidar)
  --ref_voice FILE      Reference WAV for XTTS voice cloning
  --keep-temp           Keep temporary files after processing
```

## Supported Languages

Edge TTS supports 100+ languages. Common codes:
- `ru` - Russian
- `en` - English
- `de` - German
- `fr` - French
- `es` - Spanish
- `it` - Italian
- `pt` - Portuguese
- `ja` - Japanese
- `zh` - Chinese

Full list: https://speech.microsoft.com/portal/voicegallery

## Output

The script generates:
- `{video}_dubbed.mp4` - Video with dubbed audio
- `{video}_{lang}.srt` - Subtitle file with translations

## Performance

| Engine | Speed | Quality | Languages | Notes |
|--------|-------|---------|-----------|-------|
| Edge TTS | Fast | ⭐⭐⭐⭐⭐ | 100+ | Best quality, requires internet |
| Silero | Very Fast | ⭐⭐⭐ | Russian only | Offline, robotic |
| XTTS | Slow | ⭐⭐⭐⭐⭐ | 16 | Voice cloning, GPU recommended |

## Troubleshooting

### "No module named 'soundfile'"
```bash
pip install soundfile
```

### "TorchCodec is required"
This is already patched in the code. If you still see it, update PyTorch:
```bash
pip install --upgrade torch torchaudio
```

### Silero model download fails
The script will auto-download on first run (~40MB). Check your internet connection.

### XTTS out of memory
Use CPU mode or reduce video length. For long videos, split into segments.

### Poor voice quality with Silero
Use Edge TTS or XTTS instead. Silero is designed for speed, not quality.

### Ollama Integration Features

1. **Privacy**: Your transcripts and translations never leave your local machine.
2. **Custom Context**: LLMs can handle nuances, slang, and technical terms better than basic translators.
3. **Cost**: 100% free with no character limits or subscription fees.
4. **Offline Workflow**: Combined with Silero or XTTS, you can dub videos without an active internet connection.

| Feature   | Google Translate       | Ollama (Local LLM)        |
|-----------|------------------------|---------------------------|
| Speed     | Instant                | Depends on your GPU/RAM   |
| Setup     | Zero setup             | Requires model download  |
| Internet | Required               | Not required              |
| Quality   | Literal / Standard     | Contextual / Natural      |


## Technical Details

### Processing Pipeline
1. **Extract Audio**: FFmpeg extracts mono 16kHz WAV
2. **Transcribe**: Whisper "base" model transcribes with timestamps
3. **Translate**: Google Translate API translates segments
4. **Synthesize**: TTS engine generates speech for each subtitle
5. **Merge**: FFmpeg mixes original (20%) + dubbed (150%) audio with video

### Audio Mixing
- Original audio: 20% volume (background)
- Dubbed audio: 150% volume (foreground)
- Output: AAC 128kbps, video copied without re-encoding

## License

MIT License - see LICENSE file

## Credits

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [Edge-TTS](https://github.com/rany2/edge-tts) - Microsoft TTS
- [Silero Models](https://github.com/snakers4/silero-models) - Russian TTS
- [Coqui TTS](https://github.com/coqui-ai/TTS) - XTTS voice cloning

## Contributing

Issues and pull requests welcome!
