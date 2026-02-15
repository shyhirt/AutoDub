#!/usr/bin/env python3
"""
AutoDub Pro v4.0 - Professional Video Translation and Dubbing Tool
FIXED: Audio distortion and Piper silence issues (Linux/Fedora compatible)
"""

import sys, os, asyncio, subprocess, argparse, requests, json, re, time, logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
import whisper, pysrt, edge_tts
from deep_translator import GoogleTranslator
from datetime import timedelta, datetime
from tqdm import tqdm
import functools
import torch
import torchaudio
import hashlib
import numpy as np

# Language map for Ollama
LANG_MAP = {
    'af': 'Afrikaans', 'ar': 'Arabic', 'az': 'Azerbaijani', 'be': 'Belarusian',
    'bg': 'Bulgarian', 'bn': 'Bengali', 'bs': 'Bosnian', 'ca': 'Catalan',
    'cs': 'Czech', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German',
    'el': 'Greek', 'en': 'English', 'eo': 'Esperanto', 'es': 'Spanish',
    'et': 'Estonian', 'eu': 'Basque', 'fa': 'Persian', 'fi': 'Finnish',
    'fr': 'French', 'gl': 'Galician', 'gu': 'Gujarati', 'he': 'Hebrew',
    'hi': 'Hindi', 'hr': 'Croatian', 'hu': 'Hungarian', 'hy': 'Armenian',
    'id': 'Indonesian', 'is': 'Icelandic', 'it': 'Italian', 'ja': 'Japanese',
    'ka': 'Georgian', 'kk': 'Kazakh', 'km': 'Khmer', 'kn': 'Kannada',
    'ko': 'Korean', 'ky': 'Kyrgyz', 'la': 'Latin', 'lo': 'Lao',
    'lt': 'Lithuanian', 'lv': 'Latvian', 'mk': 'Macedonian', 'ml': 'Malayalam',
    'mn': 'Mongolian', 'mr': 'Marathi', 'ms': 'Malay', 'mt': 'Maltese',
    'my': 'Burmese', 'ne': 'Nepali', 'nl': 'Dutch', 'no': 'Norwegian',
    'pa': 'Punjabi', 'pl': 'Polish', 'pt': 'Portuguese', 'ro': 'Romanian',
    'ru': 'Russian', 'si': 'Sinhala', 'sk': 'Slovak', 'sl': 'Slovenian',
    'sq': 'Albanian', 'sr': 'Serbian', 'sv': 'Swedish', 'sw': 'Swahili',
    'ta': 'Tamil', 'te': 'Telugu', 'th': 'Thai', 'tl': 'Tagalog',
    'tr': 'Turkish', 'uk': 'Ukrainian', 'ur': 'Urdu', 'uz': 'Uzbek',
    'vi': 'Vietnamese', 'zh': 'Chinese'
}

# Piper voice models mapping
PIPER_VOICES = {
    'ru': ('ru_RU', 'ruslan', 'medium'),
    'en': ('en_US', 'lessac', 'medium'),
    'es': ('es_ES', 'davefx', 'medium'),
    'fr': ('fr_FR', 'siwis', 'medium'),
    'de': ('de_DE', 'thorsten', 'medium'),
    'it': ('it_IT', 'riccardo', 'x_low'),
    'pt': ('pt_BR', 'edresson', 'low'),
    'pl': ('pl_PL', 'darkman', 'medium'),
    'uk': ('uk_UA', 'ukrainian_tts', 'medium'),
    'zh': ('zh_CN', 'huayan', 'x_low'),
    'ja': ('ja_JP', 'hikari', 'medium'),
    'ko': ('ko_KR', 'kss', 'medium'),
    'nl': ('nl_NL', 'rdh', 'medium'),
    'tr': ('tr_TR', 'dfki', 'medium'),
    'vi': ('vi_VN', 'vivos', 'x_low'),
}

EMOTION_STYLES = ['angry', 'cheerful', 'excited', 'friendly', 'hopeful', 'sad',
                  'shouting', 'terrified', 'unfriendly', 'whispering']

try:
    import soundfile as sf
    import noisereduce as nr
    from piper import PiperVoice
except ImportError:
    print("📦 Installing required packages...")
    packages = ["soundfile", "noisereduce", "piper-tts"]
    subprocess.run([sys.executable, "-m", "pip", "install"] + packages,
                   check=True, stdout=subprocess.DEVNULL)
    import soundfile as sf
    import noisereduce as nr
    from piper import PiperVoice

torch.load = functools.partial(torch.load, weights_only=False)

def forced_load(uri, **kwargs):
    """Fixed audio loading with proper format handling"""
    data, samplerate = sf.read(uri, dtype='float32')
    tensor = torch.from_numpy(data).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.transpose(0, 1)
    return tensor, samplerate

torchaudio.load = forced_load
os.environ["COQUI_TOS_AGREED"] = "1"

class Logger:
    """Enhanced logging with both file and console output"""
    def __init__(self, work_dir: Path):
        self.log_file = work_dir / "autodub.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def info(self, msg): self.logger.info(msg)
    def warning(self, msg): self.logger.warning(msg)
    def error(self, msg): self.logger.error(msg)
    def debug(self, msg): self.logger.debug(msg)

class StateManager:
    """Manages processing state for smart resume"""
    def __init__(self, work_dir: Path):
        self.state_file = work_dir / "state.json"
        self.state = self.load_state()

    def load_state(self) -> Dict:
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'steps_completed': [],
            'last_update': None,
            'video_hash': None
        }

    def save_state(self):
        self.state['last_update'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def mark_completed(self, step: str):
        if step not in self.state['steps_completed']:
            self.state['steps_completed'].append(step)
        self.save_state()

    def is_completed(self, step: str) -> bool:
        return step in self.state['steps_completed']

def get_file_hash(filepath: str) -> str:
    """Generate hash for file integrity check"""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        hasher.update(f.read(8192))
    return hasher.hexdigest()

def create_silence_wav(duration_seconds: float, output_file: str, sample_rate: int = 22050):
    """Create silent WAV file properly"""
    try:
        num_samples = int(duration_seconds * sample_rate)
        silence = np.zeros(num_samples, dtype=np.float32)
        sf.write(output_file, silence, sample_rate, subtype='PCM_16')
        return True
    except Exception as e:
        logging.error(f"Failed to create silence: {e}")
        return False

def download_piper_model(lang_code: str, models_dir: Path) -> Optional[Path]:
    """
    Hardcoded path to manual download (Most reliable)
    """
    # Hardcoded path to the model downloaded manually
    # Use 'Ruslan' for Russian language
    if 'ru' in lang_code:
        model_name = "ru_RU-ruslan-medium.onnx"
    else:
        # For other languages (default to English)
        model_name = "en_US-lessac-medium.onnx"

    # Search in user's home directory
    manual_path = Path.home() / ".piper_models" / model_name

    if manual_path.exists():
        logging.info(f"✓ Found manual model: {manual_path}")
        return manual_path
    else:
        logging.error(f"❌ Model file missing: {manual_path}")
        logging.error("👉 Please run the wget commands from the instructions to download the model manually!")
        return None

def generate_piper(subs, model_path: Path, concat_list: list, temp_files: list,
                   work_dir: Path, enable_stretch: bool):
    """Generate speech using Piper CLI via subprocess"""

    import shutil
    piper_cmd = shutil.which("piper")

    # If piper not found in PATH, try finding it in venv
    if not piper_cmd:
        possible_path = Path(sys.executable).parent / "piper"
        if possible_path.exists():
            piper_cmd = str(possible_path)

    if not piper_cmd:
        logging.error("❌ Piper command not found!")
        return

    logging.info(f"🎙️ Using Piper binary: {piper_cmd}")
    logging.info(f"📂 Model path: {model_path}")

    # Determine sample_rate (usually 22050 for ruslan medium)
    sample_rate = 22050

    generated_count = 0
    current_time_ms = 0

    for i, sub in enumerate(tqdm(subs, desc="Piper Synthesis", unit="phrase")):
        start_ms = (sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds) * 1000 + sub.start.milliseconds
        end_ms = (sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds) * 1000 + sub.end.milliseconds

        # Clean text from quotes to avoid CLI breakage
        text = sub.text.replace("\n", " ").replace('"', '').replace("'", "").strip()
        if not text:
            continue

        target_duration = (end_ms - start_ms) / 1000.0

        # --- Silence ---
        sil_dur_ms = start_ms - current_time_ms
        if sil_dur_ms > 100:
            sil_file = work_dir / f"sil_{i}.wav"
            try:
                num_samples = int((sil_dur_ms / 1000.0) * sample_rate)
                silence_data = np.zeros(num_samples, dtype=np.float32)
                sf.write(str(sil_file), silence_data, sample_rate)
                concat_list.append(f"file '{sil_file}'")
                temp_files.append(str(sil_file))
                current_time_ms += sil_dur_ms
            except:
                pass

        # --- Generation ---
        f_temp = work_dir / f"p_raw_{i}.wav"
        f_final = work_dir / f"p_fin_{i}.wav"

        if not f_final.exists():
            try:
                # IMPORTANT: pass full path to onnx file
                cmd = [
                    piper_cmd,
                    "--model", str(model_path),
                    "--output_file", str(f_temp)
                ]

                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE
                )
                stdout, stderr = process.communicate(input=text.encode('utf-8'))

                if f_temp.exists() and f_temp.stat().st_size > 1000:
                    # Time stretching
                    if enable_stretch:
                        if not stretch_audio_smart(str(f_temp), str(f_final), target_duration, work_dir, sample_rate):
                            data, sr = sf.read(str(f_temp))
                            sf.write(str(f_final), data, sr)
                    else:
                        data, sr = sf.read(str(f_temp))
                        sf.write(str(f_final), data, sr)

                    concat_list.append(f"file '{f_final}'")
                    temp_files.append(str(f_final))
                    generated_count += 1
                else:
                    logging.warning(f"Piper fail/empty: {text[:20]}... Error: {stderr.decode()[:100]}")

            except Exception as e:
                logging.error(f"Segment {i} error: {e}")
                continue

        current_time_ms = end_ms

    logging.info(f"✓ Generated {generated_count}/{len(subs)} segments")

async def get_edge_voice(lang_code: str, emotion: Optional[str] = None) -> Tuple[str, bool]:
    """Prefer hu-HU-TamasNeural, fallback to any suitable voice"""
    try:
        voices = await edge_tts.VoicesManager.create()
        suitable = voices.find(Locale=lang_code)

        if not suitable:
            suitable = [v for v in voices.voices if v["Locale"].startswith(lang_code[:2])]

        # 1 Prefer Hungarian Tamas explicitly
        if lang_code.lower().startswith("hu"):
            for voice in suitable:
                if voice.get("Name") == "hu-HU-TamasNeural":
                    return voice["Name"], bool(voice.get("StyleList"))

        # 2 Any voice with emotion/styles
        for voice in suitable:
            if voice.get("StyleList"):
                return voice["Name"], True

        # 3️ Any suitable voice
        if suitable:
            return suitable[0]["Name"], False

    except Exception as e:
        print(f"⚠️ Voice search error: {e}")

    return "en-US-ChristopherNeural", False

def format_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp format"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    millis = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def detect_emotion(text: str) -> Optional[str]:
    """Simple emotion detection based on text patterns - IMPROVED"""
    text_lower = text.lower()

    # Angry/Unfriendly
    if any(word in text_lower for word in ['angry', 'hate', 'terrible', 'worst', 'stupid', 'damn', 'hell']):
        return 'angry'

    # Sad
    if any(word in text_lower for word in ['sad', 'unfortunately', 'sorry', 'tragic', 'died', 'death', 'crying']):
        return 'sad'

    # Excited/Cheerful (check for exclamation marks and positive words)
    exclamation_count = text.count('!')
    if exclamation_count >= 2 or any(word in text_lower for word in ['wow', 'amazing', 'awesome', 'fantastic', 'incredible', 'wonderful']):
        return 'excited'
    elif exclamation_count == 1 or any(word in text_lower for word in ['great', 'good', 'nice', 'happy', 'excellent']):
        return 'cheerful'

    # Terrified/Shouting
    if any(word in text_lower for word in ['scared', 'terrified', 'afraid', 'panic', 'scream']):
        return 'terrified'
    if text.isupper() and len(text) > 10:  # ALL CAPS = shouting
        return 'shouting'

    # Whispering
    if any(word in text_lower for word in ['whisper', 'quietly', 'secret', 'shh']):
        return 'whispering'

    # Friendly
    if any(word in text_lower for word in ['hello', 'hi', 'welcome', 'thanks', 'thank you', 'please']):
        return 'friendly'

    # Hopeful
    if any(word in text_lower for word in ['hope', 'maybe', 'perhaps', 'possibly', 'wish']):
        return 'hopeful'

    return None

def translate_with_retry(text: str, target_lang: str, translator_type: str,
                         ollama_model: str, max_retries: int = 3) -> str:
    """Translate with fallback mechanism"""
    for attempt in range(max_retries):
        try:
            if translator_type == "google":
                translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
                if translated and translated != text:
                    return translated
            elif translator_type == "ollama":
                translated = translate_ollama(text, target_lang, ollama_model)
                if translated and translated != text:
                    return translated
        except Exception as e:
            if attempt == max_retries - 1:
                logging.warning(f"Translation failed after {max_retries} attempts: {e}")

    try:
        if translator_type == "google":
            logging.info("Falling back to Ollama translator")
            return translate_ollama(text, target_lang, ollama_model)
        else:
            logging.info("Falling back to Google translator")
            return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except:
        logging.error(f"All translation methods failed for: {text[:50]}...")
        return text

def translate_ollama(text: str, target_lang: str, model_name: str) -> str:
    """Translate using Ollama"""
    url = "http://localhost:11434/api/generate"
    full_lang = LANG_MAP.get(target_lang.lower(), target_lang)

    prompt = (
        f"Translate the following text into {full_lang}. "
        f"Match the tone and style of the original. "
        f"Output ONLY the translation without quotes or explanations.\n\n"
        f"Text: {text}"
    )

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "stop": ["\n\n", "Note:", "Explanation:"]
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            res = response.json().get('response', '').strip()
            return res.strip('"').strip("'").strip()
        return text
    except Exception as e:
        logging.warning(f"Ollama error: {e}")
        return text

def merge_segments_into_sentences(segments: List[Dict], max_duration: float = 10.0) -> List[Dict]:
    """Intelligently merge Whisper segments into complete sentences"""
    sentence_endings = re.compile(r'[.!?;:]\s*$')
    merged = []
    current_group = {'text': '', 'start': None, 'end': None}

    for i, seg in enumerate(segments):
        text = seg['text'].strip()
        if not text:
            continue

        if current_group['start'] is None:
            current_group['start'] = seg['start']

        if current_group['text']:
            current_group['text'] += ' ' + text
        else:
            current_group['text'] = text

        current_group['end'] = seg['end']

        duration = current_group['end'] - current_group['start']
        has_sentence_end = sentence_endings.search(text)

        has_pause = False
        if i + 1 < len(segments):
            next_start = segments[i + 1]['start']
            pause_duration = next_start - seg['end']
            has_pause = pause_duration > 0.5

        if has_sentence_end or duration >= max_duration or has_pause:
            merged.append({
                'text': current_group['text'],
                'start': current_group['start'],
                'end': current_group['end']
            })
            current_group = {'text': '', 'start': None, 'end': None}

    if current_group['text']:
        merged.append({
            'text': current_group['text'],
            'start': current_group['start'],
            'end': current_group['end']
        })

    return merged

def get_audio_duration(audio_file: str) -> Optional[float]:
    """Get audio duration"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_file],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        return float(result.stdout.strip())
    except:
        return None

def calculate_speed_factor(original_duration: float, target_duration: float) -> float:
    """Calculate optimal speed factor"""
    if target_duration == 0:
        return 1.0

    ratio = original_duration / target_duration

    if 0.95 <= ratio <= 1.05:
        return 1.0

    if ratio > 2.5:
        return 2.5
    elif ratio < 0.5:
        return 0.5

    return ratio

def stretch_audio_smart(input_file: str, output_file: str, target_duration: float,
                       work_dir: Path, target_sr: int = 22050) -> bool:
    """Intelligently stretch audio - FIXED VERSION"""
    try:
        # Load audio properly
        data, sr = sf.read(input_file, dtype='float32')
        current_duration = len(data) / sr

        if current_duration == 0:
            return False

        ratio = calculate_speed_factor(current_duration, target_duration)

        # No stretching needed
        if ratio == 1.0:
            # Resample if needed
            if sr != target_sr:
                subprocess.run([
                    "ffmpeg", "-y", "-i", input_file,
                    "-ar", str(target_sr), "-ac", "1",
                    output_file
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            else:
                sf.write(output_file, data, sr, subtype='PCM_16')
            return True

        # Build atempo filter chain
        filter_chain = []
        remaining_ratio = ratio

        while remaining_ratio > 2.0:
            filter_chain.append("atempo=2.0")
            remaining_ratio /= 2.0

        while remaining_ratio < 0.5:
            filter_chain.append("atempo=0.5")
            remaining_ratio /= 0.5

        remaining_ratio = max(0.5, min(2.0, remaining_ratio))
        filter_chain.append(f"atempo={remaining_ratio:.4f}")

        filter_str = ",".join(filter_chain)

        # Apply stretching with target sample rate
        subprocess.run([
            "ffmpeg", "-y", "-i", input_file,
            "-filter:a", filter_str,
            "-ar", str(target_sr), "-ac", "1",
            output_file
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        return True

    except Exception as e:
        logging.warning(f"Audio stretching failed: {e}, using original")
        try:
            # Fallback: just resample
            subprocess.run([
                "ffmpeg", "-y", "-i", input_file,
                "-ar", str(target_sr), "-ac", "1",
                output_file
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except:
            return False

def reduce_noise(input_file: str, output_file: str) -> bool:
    """Apply noise reduction"""
    try:
        data, rate = sf.read(input_file, dtype='float32')
        reduced_noise = nr.reduce_noise(y=data, sr=rate, stationary=True, prop_decrease=0.8)
        sf.write(output_file, reduced_noise, rate, subtype='PCM_16')
        return True
    except Exception as e:
        logging.warning(f"Noise reduction failed: {e}")
        try:
            sf.write(output_file, *sf.read(input_file))
        except:
            subprocess.run(["cp", input_file, output_file], check=True)
        return False

def normalize_audio(input_file: str, output_file: str, target_level: float = -20.0) -> bool:
    """Normalize audio to target loudness"""
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_file,
            "-filter:a", f"loudnorm=I={target_level}:TP=-1.5:LRA=11",
            "-ar", "44100", "-ac", "1",
            output_file
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception as e:
        logging.warning(f"Audio normalization failed: {e}")
        subprocess.run(["cp", input_file, output_file], check=True)
        return False

async def generate_tts_edge(text: str, voice: str, output_file: str,
                           emotion: Optional[str] = None,
                           rate: str = "+0%",
                           has_emotion_support: bool = False) -> bool:
    """Generate TTS with Edge - FIXED emotion support"""
    try:
        # If emotion is requested and voice supports it
        if emotion and has_emotion_support and emotion in EMOTION_STYLES:
            logging.debug(f"Using emotion: {emotion}")
            communicate = edge_tts.Communicate(text, voice, rate=rate, style=emotion)
        else:
            communicate = edge_tts.Communicate(text, voice, rate=rate)

        await communicate.save(output_file)
        return True
    except Exception as e:
        logging.error(f"Edge TTS failed: {e}")
        # Try without emotion as fallback
        if emotion:
            try:
                logging.debug("Retrying without emotion style...")
                communicate = edge_tts.Communicate(text, voice, rate=rate)
                await communicate.save(output_file)
                return True
            except:
                pass
        return False

def parallel_translate(segments: List[Dict], target_lang: str, translator_type: str,
                      ollama_model: str, max_workers: int = 4) -> List[Dict]:
    """Translate multiple segments in parallel"""

    def translate_segment(seg_data):
        idx, seg = seg_data
        text = seg['text'].strip()
        if not text:
            return idx, seg

        translated = translate_with_retry(text, target_lang, translator_type, ollama_model)

        return idx, {
            'text': translated,
            'start': seg['start'],
            'end': seg['end']
        }

    results = [None] * len(segments)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(translate_segment, (i, seg)): i
                  for i, seg in enumerate(segments)}

        for future in tqdm(as_completed(futures), total=len(segments),
                          desc="Translating", unit="segment"):
            try:
                idx, translated_seg = future.result()
                results[idx] = translated_seg
            except Exception as e:
                logging.error(f"Translation failed: {e}")

    return [r for r in results if r is not None]

async def synthesize_speech_batch(subs, voice: str, work_dir: Path,
                                  enable_stretch: bool, emotion_detection: bool,
                                  rate_adjust: bool, has_emotion_support: bool = False) -> Tuple[List[str], List[str]]:
    """Synthesize speech with Edge TTS - FIXED VERSION"""
    concat_list, temp_files = [], []
    current_time_ms = 0
    target_sr = 44100  # Edge TTS default

    emotion_stats = {}  # Track emotion usage

    for i, s in enumerate(tqdm(subs, desc="Edge TTS", unit="sentence")):
        start_ms = (s.start.hours*3600 + s.start.minutes*60 + s.start.seconds)*1000 + s.start.milliseconds
        end_ms = (s.end.hours*3600 + s.end.minutes*60 + s.end.seconds)*1000 + s.end.milliseconds
        txt = s.text.strip()

        if not txt:
            continue

        target_duration = (end_ms - start_ms) / 1000.0

        # Add silence - FIXED
        silence_dur_ms = start_ms - current_time_ms
        if silence_dur_ms > 100:
            silence_file = work_dir / f"silence_{i}.wav"
            if not silence_file.exists():
                create_silence_wav(silence_dur_ms / 1000.0, str(silence_file), target_sr)

            if silence_file.exists():
                concat_list.append(f"file '{silence_file}'")
                temp_files.append(str(silence_file))
                current_time_ms += silence_dur_ms

        # Generate speech
        raw_file = work_dir / f"speech_{i}_raw.mp3"
        wav_file = work_dir / f"speech_{i}_converted.wav"
        processed_file = work_dir / f"speech_{i}_processed.wav"
        final_file = work_dir / f"speech_{i}_final.wav"

        if not final_file.exists():
            # Detect emotion
            emotion = None
            if emotion_detection and has_emotion_support:
                emotion = detect_emotion(txt)
                if emotion:
                    emotion_stats[emotion] = emotion_stats.get(emotion, 0) + 1
                    logging.debug(f"Segment {i}: detected emotion '{emotion}' in '{txt[:30]}...'")

            # Calculate optimal speaking rate
            rate = "+0%"
            if rate_adjust and target_duration > 0:
                words = len(txt.split())
                estimated_duration = (words / 150) * 60  # ~150 words per minute
                if estimated_duration > 0:
                    rate_factor = (estimated_duration / target_duration) - 1
                    rate_factor = max(-0.5, min(0.5, rate_factor))
                    rate = f"{rate_factor*100:+.0f}%"
                    if abs(rate_factor) > 0.1:
                        logging.debug(f"Segment {i}: adjusting rate to {rate}")

            # Generate TTS
            if not raw_file.exists():
                success = await generate_tts_edge(txt, voice, str(raw_file), emotion, rate, has_emotion_support)
                if not success:
                    logging.warning(f"Failed to generate TTS for segment {i}")
                    continue

            # Convert MP3 to WAV immediately
            if not wav_file.exists():
                try:
                    subprocess.run([
                        "ffmpeg", "-y", "-i", str(raw_file),
                        "-ar", str(target_sr), "-ac", "1",
                        str(wav_file)
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                except Exception as e:
                    logging.error(f"Failed to convert MP3 to WAV for segment {i}: {e}")
                    continue

            # Apply noise reduction
            if not processed_file.exists():
                try:
                    reduce_noise(str(wav_file), str(processed_file))
                except:
                    sf.write(str(processed_file), *sf.read(str(wav_file)))

            # Time-stretch to match original duration
            if enable_stretch:
                success = stretch_audio_smart(str(processed_file), str(final_file),
                                             target_duration, work_dir, target_sr)
                if not success:
                    sf.write(str(final_file), *sf.read(str(processed_file)))
            else:
                sf.write(str(final_file), *sf.read(str(processed_file)))

        if final_file.exists() and final_file.stat().st_size > 100:
            concat_list.append(f"file '{final_file}'")
            temp_files.append(str(final_file))
        current_time_ms = end_ms

    # Log emotion statistics
    if emotion_stats:
        logging.info(f"🎭 Emotion usage: {emotion_stats}")

    return concat_list, temp_files

async def process_video(video_path: str, args, logger: Logger):
    """Main video processing pipeline"""

    video_path = Path(video_path)
    video_basename = video_path.stem
    work_dir = Path.cwd() / f"{video_basename}_work"
    work_dir.mkdir(exist_ok=True)

    logger.info(f"📂 Workspace: {work_dir}")

    state = StateManager(work_dir)

    # File paths
    audio_wav = work_dir / "original_audio.wav"
    audio_clean = work_dir / "audio_clean.wav"
    transcript_json = work_dir / "transcript.json"
    merged_json = work_dir / "merged_sentences.json"
    translated_json = work_dir / "translated.json"
    srt_file = work_dir / f"subtitles_{args.target_lang}.srt"
    concat_list_file = work_dir / "concat_list.txt"
    voiceover_wav = work_dir / "voiceover.wav"
    voiceover_norm = work_dir / "voiceover_normalized.wav"
    output_file = Path.cwd() / f"{video_basename}_dubbed_{args.target_lang}.mp4"

    start_time = time.time()

    # Step 1: Extract audio
    if state.is_completed('extract_audio') and audio_wav.exists():
        logger.info("[1/7] ✓ Audio extraction already completed")
    else:
        logger.info("[1/7] Extracting audio...")
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1", str(audio_wav)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        state.mark_completed('extract_audio')

    # Step 2: Audio enhancement
    if state.is_completed('audio_enhancement') and audio_clean.exists():
        logger.info("[2/7] ✓ Audio enhancement already completed")
    else:
        logger.info("[2/7] Enhancing audio (noise reduction)...")
        reduce_noise(str(audio_wav), str(audio_clean))
        state.mark_completed('audio_enhancement')

    # Step 3: Transcription
    segments = []
    if state.is_completed('transcription') and transcript_json.exists():
        logger.info("[3/7] ✓ Transcription already completed")
        with open(transcript_json, 'r', encoding='utf-8') as f:
            segments = json.load(f)
    else:
        logger.info(f"[3/7] Transcribing with Whisper ({args.whisper_model})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"⚙️  Using device: {device.upper()}")

        try:
            model = whisper.load_model(args.whisper_model, device=device)
            result = model.transcribe(str(audio_clean), fp16=(device == "cuda"), verbose=False)
            segments = result['segments']

            detected_lang = result.get('language', 'unknown')
            logger.info(f"🌍 Detected language: {detected_lang}")

            with open(transcript_json, 'w', encoding='utf-8') as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)
            state.mark_completed('transcription')
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return 1

    # Step 4: Merge into sentences
    merged_segments = []
    if state.is_completed('merge_sentences') and merged_json.exists():
        logger.info("[4/7] ✓ Sentence merging already completed")
        with open(merged_json, 'r', encoding='utf-8') as f:
            merged_segments = json.load(f)
    else:
        logger.info("[4/7] Merging segments into sentences...")
        merged_segments = merge_segments_into_sentences(segments, args.max_sentence_duration)
        logger.info(f"✨ Merged {len(segments)} segments → {len(merged_segments)} sentences")

        with open(merged_json, 'w', encoding='utf-8') as f:
            json.dump(merged_segments, f, ensure_ascii=False, indent=2)
        state.mark_completed('merge_sentences')

    # Step 5: Translation
    translated_segments = []
    if state.is_completed('translation') and translated_json.exists():
        logger.info("[5/7] ✓ Translation already completed")
        with open(translated_json, 'r', encoding='utf-8') as f:
            translated_segments = json.load(f)
    else:
        logger.info(f"[5/7] Translating to {args.target_lang} with {args.translator}...")

        if args.parallel and len(merged_segments) > 10:
            logger.info("🚀 Using parallel translation")
            translated_segments = parallel_translate(
                merged_segments, args.target_lang,
                args.translator, args.ollama_model,
                max_workers=args.workers
            )
        else:
            translated_segments = []
            for seg in tqdm(merged_segments, desc="Translating", unit="segment"):
                text = seg['text'].strip()
                if not text:
                    continue

                translated = translate_with_retry(
                    text, args.target_lang,
                    args.translator, args.ollama_model
                )

                translated_segments.append({
                    'text': translated,
                    'start': seg['start'],
                    'end': seg['end']
                })

        with open(translated_json, 'w', encoding='utf-8') as f:
            json.dump(translated_segments, f, ensure_ascii=False, indent=2)
        state.mark_completed('translation')

    # Generate SRT file
    with open(srt_file, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(translated_segments):
            f.write(f"{i+1}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")

    # Step 6: Speech synthesis
    if state.is_completed('synthesis') and voiceover_norm.exists():
        logger.info("[6/7] ✓ Speech synthesis already completed")
    else:
        logger.info("[6/7] Synthesizing speech...")

        subs = pysrt.open(str(srt_file))
        concat_list, temp_files = [], []

        if args.tts == "edge":
            logger.info(f"🔎 Finding voice for: {args.target_lang}")
            voice, has_emotions = await get_edge_voice(args.target_lang)
            logger.info(f"🎙️  Selected: {voice} (Emotions: {'Yes' if has_emotions else 'No'})")

            concat_list, temp_files = await synthesize_speech_batch(
                subs, voice, work_dir,
                enable_stretch=not args.no_stretch,
                emotion_detection=args.detect_emotion,
                rate_adjust=args.auto_rate,
                has_emotion_support=has_emotions  # Pass emotion support flag
            )

        elif args.tts == "piper":
            logger.info(f"🔎 Loading Piper model for: {args.target_lang}")
            models_dir = Path.home() / ".piper_models"
            model_path = download_piper_model(args.target_lang, models_dir)

            if not model_path:
                logger.error("Failed to download Piper model")
                return 1

            logger.info(f"🎙️  Using Piper (offline mode)")
            generate_piper(subs, model_path, concat_list, temp_files, work_dir,
                          enable_stretch=not args.no_stretch)

        if not concat_list:
            logger.error("No audio generated!")
            return 1

        # Save concat list
        with open(concat_list_file, 'w') as f:
            f.write('\n'.join(concat_list))

        # Verify files exist
        missing_files = []
        for line in concat_list:
            if line.startswith("file '"):
                filepath = line[6:-1]
                if not Path(filepath).exists():
                    missing_files.append(filepath)

        if missing_files:
            logger.warning(f"⚠️  {len(missing_files)} audio files missing, removing from list")
            valid_list = [line for line in concat_list
                         if not any(missing in line for missing in missing_files)]
            with open(concat_list_file, 'w') as f:
                f.write('\n'.join(valid_list))

        # Concatenate audio
        logger.info("🔗 Concatenating audio segments...")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list_file),
            "-c", "copy", str(voiceover_wav)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Normalize audio
        logger.info("📊 Normalizing audio levels...")
        normalize_audio(str(voiceover_wav), str(voiceover_norm), target_level=-16.0)

        state.mark_completed('synthesis')

    # Step 7: Final video assembly
    logger.info("[7/7] Assembling final video...")

    bg_volume = args.background_volume
    fg_volume = args.voice_volume

    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(voiceover_norm),
        "-filter_complex",
        f"[0:a]volume={bg_volume}[bg];[1:a]volume={fg_volume}[fg];[bg][fg]amix=inputs=2:duration=first:dropout_transition=2",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-map", "0:v:0",
        str(output_file)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Cleanup
    if not args.keep_temp:
        logger.info("🧹 Cleaning up temporary files...")
        for pattern in ['*_raw.*', 'silence_*.wav', 'speech_*_processed.*']:
            for f in work_dir.glob(pattern):
                f.unlink(missing_ok=True)

    elapsed_time = time.time() - start_time
    logger.info(f"✅ Completed in {elapsed_time/60:.1f} minutes!")
    logger.info(f"📹 Output: {output_file}")
    logger.info(f"📁 Working files: {work_dir}")

    if args.subtitles_only:
        logger.info(f"📝 Subtitles: {srt_file}")

    return 0

async def batch_process(video_files: List[str], args, logger: Logger):
    """Process multiple videos in batch"""
    logger.info(f"🎬 Batch processing {len(video_files)} videos")

    results = []
    for i, video in enumerate(video_files, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing video {i}/{len(video_files)}: {video}")
        logger.info(f"{'='*60}\n")

        try:
            result = await process_video(video, args, logger)
            results.append((video, result == 0))
        except Exception as e:
            logger.error(f"Failed to process {video}: {e}")
            results.append((video, False))

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info(f"{'='*60}")

    success_count = sum(1 for _, success in results if success)
    logger.info(f"✅ Successful: {success_count}/{len(results)}")

    if success_count < len(results):
        logger.info("❌ Failed:")
        for video, success in results:
            if not success:
                logger.info(f"  - {video}")

async def main():
    parser = argparse.ArgumentParser(
        description="AutoDub Pro v4.0 - Professional Video Dubbing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python autodub_pro.py video.mp4 --target_lang ru

  # High quality with all enhancements
  python autodub_pro.py video.mp4 --target_lang es --whisper_model large --detect-emotion --auto-rate

  # Batch processing
  python autodub_pro.py video1.mp4 video2.mp4 video3.mp4 --target_lang fr --parallel

  # Subtitles only
  python autodub_pro.py video.mp4 --target_lang de --subtitles-only
        """
    )

    # Input/Output
    parser.add_argument("videos", nargs="+", help="Input video file(s)")
    parser.add_argument("--target_lang", default="ru", help="Target language code (default: ru)")
    parser.add_argument("--output-dir", help="Output directory (default: current directory)")

    # Transcription
    parser.add_argument("--whisper_model", default="turbo",
                       choices=["tiny", "base", "small", "medium", "large", "turbo"],
                       help="Whisper model size (default: turbo)")

    # Translation
    parser.add_argument("--translator", choices=["google", "ollama"], default="google",
                       help="Translation service (default: google)")
    parser.add_argument("--ollama_model", default="llama3",
                       help="Ollama model for translation (default: llama3)")
    parser.add_argument("--parallel", action="store_true",
                       help="Enable parallel translation (faster)")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers (default: 4)")

    # TTS
    parser.add_argument("--tts", choices=["edge", "piper"], default="edge",
                       help="TTS engine (default: edge, piper=offline)")

    # Audio enhancements
    parser.add_argument("--no-stretch", action="store_true",
                       help="Disable audio time-stretching")
    parser.add_argument("--detect-emotion", action="store_true",
                       help="Enable emotion detection")
    parser.add_argument("--auto-rate", action="store_true",
                       help="Auto adjust TTS rate")
    parser.add_argument("--background-volume", type=float, default=0.15,
                       help="Original audio volume (0.0-1.0, default: 0.15)")
    parser.add_argument("--voice-volume", type=float, default=1.5,
                       help="Dubbed voice volume (0.0-2.0, default: 1.5)")

    # Processing options
    parser.add_argument("--max_sentence_duration", type=float, default=10.0,
                       help="Max sentence duration in seconds (default: 10.0)")
    parser.add_argument("--keep-temp", action="store_true",
                       help="Keep temporary files")
    parser.add_argument("--subtitles-only", action="store_true",
                       help="Generate only subtitles")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--resume", action="store_true", default=True,
                       help="Resume from checkpoint")

    args = parser.parse_args()

    # Setup logging
    first_video = Path(args.videos[0])
    work_dir = Path.cwd() / f"{first_video.stem}_work"
    work_dir.mkdir(exist_ok=True)

    logger = Logger(work_dir)

    # Print configuration
    logger.info("╔════════════════════════════════════════════════════════╗")
    logger.info("║           AutoDub Pro v4.0 - Configuration           ║")
    logger.info("╚════════════════════════════════════════════════════════╝")
    logger.info(f"📹 Videos: {len(args.videos)}")
    logger.info(f"🌍 Target Language: {args.target_lang}")
    logger.info(f"🎙️  TTS Engine: {args.tts}")
    logger.info(f"🔤 Translator: {args.translator}")
    logger.info(f"🧠 Whisper Model: {args.whisper_model}")
    logger.info(f"⚡ Parallel Processing: {'Yes' if args.parallel else 'No'}")
    logger.info(f"🎭 Emotion Detection: {'Yes' if args.detect_emotion else 'No'}")
    logger.info(f"📈 Auto Rate Adjust: {'Yes' if args.auto_rate else 'No'}")
    logger.info(f"🎵 Audio Stretching: {'Yes' if not args.no_stretch else 'No'}")
    logger.info("")

    # Process videos
    try:
        if len(args.videos) > 1:
            await batch_process(args.videos, args, logger)
        else:
            await process_video(args.videos[0], args, logger)
        return 0
    except KeyboardInterrupt:
        logger.info("\n⚠️  Process interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
