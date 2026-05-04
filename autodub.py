#!/usr/bin/env python3
"""
AutoDub Pro v4.0 - Professional Video Translation and Dubbing Tool
FIXED: Audio distortion and Piper silence issues (Linux/Fedora compatible)
Added: XTTS v2 voice cloning support
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

# XTTS supported languages
XTTS_SUPPORTED_LANGS = {
    'en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl',
    'cs', 'ar', 'zh', 'hu', 'ko', 'ja', 'hi'
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


# ─────────────────────────────────────────────
#  XTTS v2 — voice cloning
# ─────────────────────────────────────────────

def load_xtts_model():
    """Load XTTS v2 model (downloaded automatically on first run)"""
    try:
        from TTS.api import TTS
        logging.info("🔄 Loading XTTS v2 model (first run may take a while)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        logging.info(f"✓ XTTS v2 loaded on {device.upper()}")
        return tts
    except ImportError:
        logging.error("❌ TTS package not found. Install with: pip install TTS")
        return None
    except Exception as e:
        logging.error(f"❌ Failed to load XTTS model: {e}")
        return None


def extract_voice_sample(video_path: str, output_wav: str, duration: float = 30.0) -> bool:
    """Extract a clean voice sample from the source video for cloning"""
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "22050", "-ac", "1",
            "-t", str(duration),
            output_wav
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        logging.info(f"✓ Extracted voice sample: {output_wav} ({duration}s)")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to extract voice sample: {e}")
        return False


def generate_xtts(subs, tts_model, speaker_wav: str, lang_code: str,
                  concat_list: list, temp_files: list,
                  work_dir: Path, enable_stretch: bool):
    """
    Generate speech using XTTS v2 with voice cloning.

    Args:
        subs: pysrt subtitles list
        tts_model: loaded TTS model instance
        speaker_wav: path to reference audio for voice cloning
        lang_code: target language code (must be in XTTS_SUPPORTED_LANGS)
        concat_list: ffmpeg concat file list (modified in-place)
        temp_files: list of temp files to clean up (modified in-place)
        work_dir: working directory for temp files
        enable_stretch: whether to time-stretch to match subtitle timing
    """
    # XTTS outputs 24000 Hz
    sample_rate = 24000

    # Normalize lang code — XTTS uses 2-letter codes
    xtts_lang = lang_code[:2].lower()
    if xtts_lang not in XTTS_SUPPORTED_LANGS:
        logging.warning(f"⚠️  XTTS may not support '{xtts_lang}', attempting anyway")

    if not Path(speaker_wav).exists():
        logging.error(f"❌ Speaker WAV not found: {speaker_wav}")
        return

    logging.info(f"🎙️  XTTS voice cloning from: {speaker_wav}")
    logging.info(f"🌍 Target language: {xtts_lang}")

    generated_count = 0
    current_time_ms = 0

    for i, sub in enumerate(tqdm(subs, desc="XTTS Synthesis", unit="phrase")):
        start_ms = (sub.start.hours * 3600 + sub.start.minutes * 60 +
                    sub.start.seconds) * 1000 + sub.start.milliseconds
        end_ms = (sub.end.hours * 3600 + sub.end.minutes * 60 +
                  sub.end.seconds) * 1000 + sub.end.milliseconds

        text = sub.text.replace("\n", " ").strip()
        if not text:
            continue

        target_duration = (end_ms - start_ms) / 1000.0

        # ── Silence before segment ──────────────────────────
        sil_dur_ms = start_ms - current_time_ms
        if sil_dur_ms > 100:
            sil_file = work_dir / f"xtts_sil_{i}.wav"
            try:
                num_samples = int((sil_dur_ms / 1000.0) * sample_rate)
                sf.write(str(sil_file), np.zeros(num_samples, dtype=np.float32),
                         sample_rate, subtype='PCM_16')
                concat_list.append(f"file '{sil_file}'")
                temp_files.append(str(sil_file))
                current_time_ms += sil_dur_ms
            except Exception as e:
                logging.warning(f"Silence generation failed for segment {i}: {e}")

        # ── Speech generation ────────────────────────────────
        raw_file = work_dir / f"xtts_raw_{i}.wav"
        final_file = work_dir / f"xtts_fin_{i}.wav"

        if not final_file.exists():
            try:
                tts_model.tts_to_file(
                    text=text,
                    speaker_wav=speaker_wav,
                    language=xtts_lang,
                    file_path=str(raw_file)
                )

                if raw_file.exists() and raw_file.stat().st_size > 1000:
                    if enable_stretch:
                        if not stretch_audio_smart(str(raw_file), str(final_file),
                                                   target_duration, work_dir, sample_rate):
                            data, sr = sf.read(str(raw_file))
                            sf.write(str(final_file), data, sr, subtype='PCM_16')
                    else:
                        data, sr = sf.read(str(raw_file))
                        sf.write(str(final_file), data, sr, subtype='PCM_16')

                    concat_list.append(f"file '{final_file}'")
                    temp_files.append(str(final_file))
                    generated_count += 1
                else:
                    logging.warning(f"XTTS produced empty output for segment {i}: {text[:40]}")

            except Exception as e:
                logging.error(f"XTTS segment {i} error: {e}")
                continue

        current_time_ms = end_ms

    logging.info(f"✓ XTTS generated {generated_count}/{len(subs)} segments")


# ─────────────────────────────────────────────
#  Existing code below — unchanged
# ─────────────────────────────────────────────

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
        return {'steps_completed': [], 'last_update': None, 'video_hash': None}

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
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        hasher.update(f.read(8192))
    return hasher.hexdigest()


def create_silence_wav(duration_seconds: float, output_file: str, sample_rate: int = 22050):
    try:
        num_samples = int(duration_seconds * sample_rate)
        silence = np.zeros(num_samples, dtype=np.float32)
        sf.write(output_file, silence, sample_rate, subtype='PCM_16')
        return True
    except Exception as e:
        logging.error(f"Failed to create silence: {e}")
        return False


def download_piper_model(lang_code: str, models_dir: Path) -> Optional[Path]:
    if 'ru' in lang_code:
        model_name = "ru_RU-ruslan-medium.onnx"
    else:
        model_name = "en_US-lessac-medium.onnx"
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
    import shutil
    piper_cmd = shutil.which("piper")
    if not piper_cmd:
        possible_path = Path(sys.executable).parent / "piper"
        if possible_path.exists():
            piper_cmd = str(possible_path)
    if not piper_cmd:
        logging.error("❌ Piper command not found!")
        return

    logging.info(f"🎙️ Using Piper binary: {piper_cmd}")
    logging.info(f"📂 Model path: {model_path}")

    sample_rate = 22050
    generated_count = 0
    current_time_ms = 0

    for i, sub in enumerate(tqdm(subs, desc="Piper Synthesis", unit="phrase")):
        start_ms = (sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds) * 1000 + sub.start.milliseconds
        end_ms = (sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds) * 1000 + sub.end.milliseconds

        text = sub.text.replace("\n", " ").replace('"', '').replace("'", "").strip()
        if not text:
            continue

        target_duration = (end_ms - start_ms) / 1000.0

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

        f_temp = work_dir / f"p_raw_{i}.wav"
        f_final = work_dir / f"p_fin_{i}.wav"

        if not f_final.exists():
            try:
                cmd = [piper_cmd, "--model", str(model_path), "--output_file", str(f_temp)]
                process = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate(input=text.encode('utf-8'))

                if f_temp.exists() and f_temp.stat().st_size > 1000:
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
    try:
        voices = await edge_tts.VoicesManager.create()
        suitable = voices.find(Locale=lang_code)
        if not suitable:
            suitable = [v for v in voices.voices if v['Locale'].startswith(lang_code[:2])]
        if suitable:
            for voice in suitable:
                if 'StyleList' in voice and voice['StyleList']:
                    return voice['Name'], True
            return suitable[0]['Name'], False
    except Exception as e:
        print(f"⚠️ Voice search error: {e}")
    return "en-US-ChristopherNeural", False


def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    millis = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def detect_emotion(text: str) -> Optional[str]:
    text_lower = text.lower()
    if any(word in text_lower for word in ['angry', 'hate', 'terrible', 'worst', 'stupid', 'damn', 'hell']):
        return 'angry'
    if any(word in text_lower for word in ['sad', 'unfortunately', 'sorry', 'tragic', 'died', 'death', 'crying']):
        return 'sad'
    exclamation_count = text.count('!')
    if exclamation_count >= 2 or any(word in text_lower for word in ['wow', 'amazing', 'awesome', 'fantastic', 'incredible', 'wonderful']):
        return 'excited'
    elif exclamation_count == 1 or any(word in text_lower for word in ['great', 'good', 'nice', 'happy', 'excellent']):
        return 'cheerful'
    if any(word in text_lower for word in ['scared', 'terrified', 'afraid', 'panic', 'scream']):
        return 'terrified'
    if text.isupper() and len(text) > 10:
        return 'shouting'
    if any(word in text_lower for word in ['whisper', 'quietly', 'secret', 'shh']):
        return 'whispering'
    if any(word in text_lower for word in ['hello', 'hi', 'welcome', 'thanks', 'thank you', 'please']):
        return 'friendly'
    if any(word in text_lower for word in ['hope', 'maybe', 'perhaps', 'possibly', 'wish']):
        return 'hopeful'
    return None


def translate_with_retry(text: str, target_lang: str, translator_type: str,
                         ollama_model: str, max_retries: int = 3) -> str:
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
        "options": {"temperature": 0.2, "top_p": 0.9, "stop": ["\n\n", "Note:", "Explanation:"]}
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
    sentence_endings = re.compile(r'[.!?;:]\s*$')
    merged = []
    current_group = {'text': '', 'start': None, 'end': None}

    for i, seg in enumerate(segments):
        text = seg['text'].strip()
        if not text:
            continue
        if current_group['start'] is None:
            current_group['start'] = seg['start']
        current_group['text'] = (current_group['text'] + ' ' + text).strip()
        current_group['end'] = seg['end']
        duration = current_group['end'] - current_group['start']
        has_sentence_end = sentence_endings.search(text)
        has_pause = False
        if i + 1 < len(segments):
            has_pause = (segments[i + 1]['start'] - seg['end']) > 0.5
        if has_sentence_end or duration >= max_duration or has_pause:
            merged.append({
                'text': current_group['text'],
                'start': current_group['start'],
                'end': current_group['end']
            })
            current_group = {'text': '', 'start': None, 'end': None}

    if current_group['text']:
        merged.append(current_group)
    return merged


def get_audio_duration(audio_file: str) -> Optional[float]:
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
    if target_duration == 0:
        return 1.0
    ratio = original_duration / target_duration
    if 0.95 <= ratio <= 1.05:
        return 1.0
    return max(0.5, min(2.5, ratio))


def stretch_audio_smart(input_file: str, output_file: str, target_duration: float,
                        work_dir: Path, target_sr: int = 22050) -> bool:
    try:
        data, sr = sf.read(input_file, dtype='float32')
        current_duration = len(data) / sr
        if current_duration == 0:
            return False
        ratio = calculate_speed_factor(current_duration, target_duration)
        if ratio == 1.0:
            if sr != target_sr:
                subprocess.run([
                    "ffmpeg", "-y", "-i", input_file,
                    "-ar", str(target_sr), "-ac", "1", output_file
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            else:
                sf.write(output_file, data, sr, subtype='PCM_16')
            return True
        filter_chain = []
        remaining_ratio = ratio
        while remaining_ratio > 2.0:
            filter_chain.append("atempo=2.0")
            remaining_ratio /= 2.0
        while remaining_ratio < 0.5:
            filter_chain.append("atempo=0.5")
            remaining_ratio /= 0.5
        filter_chain.append(f"atempo={max(0.5, min(2.0, remaining_ratio)):.4f}")
        subprocess.run([
            "ffmpeg", "-y", "-i", input_file,
            "-filter:a", ",".join(filter_chain),
            "-ar", str(target_sr), "-ac", "1", output_file
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception as e:
        logging.warning(f"Audio stretching failed: {e}, using original")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", input_file,
                "-ar", str(target_sr), "-ac", "1", output_file
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except:
            return False


def reduce_noise(input_file: str, output_file: str) -> bool:
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
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_file,
            "-filter:a", f"loudnorm=I={target_level}:TP=-1.5:LRA=11",
            "-ar", "44100", "-ac", "1", output_file
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
    try:
        if emotion and has_emotion_support and emotion in EMOTION_STYLES:
            communicate = edge_tts.Communicate(text, voice, rate=rate, style=emotion)
        else:
            communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(output_file)
        return True
    except Exception as e:
        logging.error(f"Edge TTS failed: {e}")
        if emotion:
            try:
                communicate = edge_tts.Communicate(text, voice, rate=rate)
                await communicate.save(output_file)
                return True
            except:
                pass
        return False


def parallel_translate(segments: List[Dict], target_lang: str, translator_type: str,
                       ollama_model: str, max_workers: int = 4) -> List[Dict]:
    def translate_segment(seg_data):
        idx, seg = seg_data
        text = seg['text'].strip()
        if not text:
            return idx, seg
        translated = translate_with_retry(text, target_lang, translator_type, ollama_model)
        return idx, {'text': translated, 'start': seg['start'], 'end': seg['end']}

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
    concat_list, temp_files = [], []
    current_time_ms = 0
    target_sr = 44100
    emotion_stats = {}

    for i, s in enumerate(tqdm(subs, desc="Edge TTS", unit="sentence")):
        start_ms = (s.start.hours*3600 + s.start.minutes*60 + s.start.seconds)*1000 + s.start.milliseconds
        end_ms = (s.end.hours*3600 + s.end.minutes*60 + s.end.seconds)*1000 + s.end.milliseconds
        txt = s.text.strip()
        if not txt:
            continue
        target_duration = (end_ms - start_ms) / 1000.0

        silence_dur_ms = start_ms - current_time_ms
        if silence_dur_ms > 100:
            silence_file = work_dir / f"silence_{i}.wav"
            if not silence_file.exists():
                create_silence_wav(silence_dur_ms / 1000.0, str(silence_file), target_sr)
            if silence_file.exists():
                concat_list.append(f"file '{silence_file}'")
                temp_files.append(str(silence_file))
                current_time_ms += silence_dur_ms

        raw_file = work_dir / f"speech_{i}_raw.mp3"
        wav_file = work_dir / f"speech_{i}_converted.wav"
        processed_file = work_dir / f"speech_{i}_processed.wav"
        final_file = work_dir / f"speech_{i}_final.wav"

        if not final_file.exists():
            emotion = None
            if emotion_detection and has_emotion_support:
                emotion = detect_emotion(txt)
                if emotion:
                    emotion_stats[emotion] = emotion_stats.get(emotion, 0) + 1

            rate = "+0%"
            if rate_adjust and target_duration > 0:
                words = len(txt.split())
                estimated_duration = (words / 150) * 60
                if estimated_duration > 0:
                    rate_factor = max(-0.5, min(0.5, (estimated_duration / target_duration) - 1))
                    rate = f"{rate_factor*100:+.0f}%"

            if not raw_file.exists():
                success = await generate_tts_edge(txt, voice, str(raw_file), emotion, rate, has_emotion_support)
                if not success:
                    continue

            if not wav_file.exists():
                try:
                    subprocess.run([
                        "ffmpeg", "-y", "-i", str(raw_file),
                        "-ar", str(target_sr), "-ac", "1", str(wav_file)
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                except Exception as e:
                    logging.error(f"Failed to convert MP3 to WAV for segment {i}: {e}")
                    continue

            if not processed_file.exists():
                try:
                    reduce_noise(str(wav_file), str(processed_file))
                except:
                    sf.write(str(processed_file), *sf.read(str(wav_file)))

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

    audio_wav       = work_dir / "original_audio.wav"
    audio_clean     = work_dir / "audio_clean.wav"
    transcript_json = work_dir / "transcript.json"
    merged_json     = work_dir / "merged_sentences.json"
    translated_json = work_dir / "translated.json"
    srt_file        = work_dir / f"subtitles_{args.target_lang}.srt"
    concat_list_file= work_dir / "concat_list.txt"
    voiceover_wav   = work_dir / "voiceover.wav"
    voiceover_norm  = work_dir / "voiceover_normalized.wav"
    output_file     = Path.cwd() / f"{video_basename}_dubbed_{args.target_lang}.mp4"

    start_time = time.time()

    # Step 1: Extract audio
    if state.is_completed('extract_audio') and audio_wav.exists():
        logger.info("[1/7] ✓ Audio extraction already completed")
    else:
        logger.info("[1/7] Extracting audio...")
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_wav)
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
            logger.info(f"🌍 Detected language: {result.get('language', 'unknown')}")
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
                merged_segments, args.target_lang, args.translator,
                args.ollama_model, max_workers=args.workers
            )
        else:
            translated_segments = []
            for seg in tqdm(merged_segments, desc="Translating", unit="segment"):
                text = seg['text'].strip()
                if not text:
                    continue
                translated = translate_with_retry(text, args.target_lang, args.translator, args.ollama_model)
                translated_segments.append({'text': translated, 'start': seg['start'], 'end': seg['end']})
        with open(translated_json, 'w', encoding='utf-8') as f:
            json.dump(translated_segments, f, ensure_ascii=False, indent=2)
        state.mark_completed('translation')

    # Generate SRT
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
                has_emotion_support=has_emotions
            )

        elif args.tts == "piper":
            logger.info(f"🔎 Loading Piper model for: {args.target_lang}")
            model_path = download_piper_model(args.target_lang, Path.home() / ".piper_models")
            if not model_path:
                logger.error("Failed to download Piper model")
                return 1
            logger.info("🎙️  Using Piper (offline mode)")
            generate_piper(subs, model_path, concat_list, temp_files, work_dir,
                           enable_stretch=not args.no_stretch)

        elif args.tts == "xtts":
            # ── XTTS voice cloning ───────────────────────────
            lang_code = args.target_lang[:2].lower()
            if lang_code not in XTTS_SUPPORTED_LANGS:
                logger.error(f"❌ XTTS does not support language '{lang_code}'. "
                             f"Supported: {sorted(XTTS_SUPPORTED_LANGS)}")
                return 1

            # Determine speaker reference wav
            speaker_wav = args.voice_sample
            if not speaker_wav:
                # Auto-extract from source video
                auto_sample = work_dir / "auto_voice_sample.wav"
                if not auto_sample.exists():
                    logger.info("🎤 No --voice-sample provided, extracting from source video...")
                    if not extract_voice_sample(str(video_path), str(auto_sample)):
                        logger.error("Failed to extract voice sample")
                        return 1
                speaker_wav = str(auto_sample)
            else:
                if not Path(speaker_wav).exists():
                    logger.error(f"❌ Voice sample not found: {speaker_wav}")
                    return 1
                logger.info(f"🎤 Using provided voice sample: {speaker_wav}")

            # Load model
            tts_model = load_xtts_model()
            if tts_model is None:
                return 1

            generate_xtts(
                subs, tts_model, speaker_wav, lang_code,
                concat_list, temp_files, work_dir,
                enable_stretch=not args.no_stretch
            )

        if not concat_list:
            logger.error("No audio generated!")
            return 1

        with open(concat_list_file, 'w') as f:
            f.write('\n'.join(concat_list))

        # Verify files
        missing_files = [
            line[6:-1] for line in concat_list
            if line.startswith("file '") and not Path(line[6:-1]).exists()
        ]
        if missing_files:
            logger.warning(f"⚠️  {len(missing_files)} audio files missing, removing from list")
            valid_list = [line for line in concat_list
                         if not any(m in line for m in missing_files)]
            with open(concat_list_file, 'w') as f:
                f.write('\n'.join(valid_list))

        logger.info("🔗 Concatenating audio segments...")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list_file), "-c", "copy", str(voiceover_wav)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        logger.info("📊 Normalizing audio levels...")
        normalize_audio(str(voiceover_wav), str(voiceover_norm), target_level=-16.0)
        state.mark_completed('synthesis')

    # Step 7: Final video assembly
    logger.info("[7/7] Assembling final video...")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(voiceover_norm),
        "-filter_complex",
        f"[0:a]volume={args.background_volume}[bg];[1:a]volume={args.voice_volume}[fg];"
        f"[bg][fg]amix=inputs=2:duration=first:dropout_transition=2",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-map", "0:v:0",
        str(output_file)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    if not args.keep_temp:
        logger.info("🧹 Cleaning up temporary files...")
        for pattern in ['*_raw.*', 'silence_*.wav', 'xtts_sil_*.wav', 'speech_*_processed.*']:
            for f in work_dir.glob(pattern):
                f.unlink(missing_ok=True)

    elapsed_time = time.time() - start_time
    logger.info(f"✅ Completed in {elapsed_time/60:.1f} minutes!")
    logger.info(f"📹 Output: {output_file}")
    logger.info(f"📁 Working files: {work_dir}")
    return 0


async def batch_process(video_files: List[str], args, logger: Logger):
    logger.info(f"🎬 Batch processing {len(video_files)} videos")
    results = []
    for i, video in enumerate(video_files, 1):
        logger.info(f"\n{'='*60}\nProcessing video {i}/{len(video_files)}: {video}\n{'='*60}\n")
        try:
            result = await process_video(video, args, logger)
            results.append((video, result == 0))
        except Exception as e:
            logger.error(f"Failed to process {video}: {e}")
            results.append((video, False))

    success_count = sum(1 for _, s in results if s)
    logger.info(f"\n{'='*60}\nBATCH SUMMARY\n{'='*60}")
    logger.info(f"✅ Successful: {success_count}/{len(results)}")
    if success_count < len(results):
        for video, success in results:
            if not success:
                logger.info(f"  ❌ {video}")


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

  # XTTS voice cloning (clone voice from source video automatically)
  python autodub_pro.py video.mp4 --target_lang ru --tts xtts

  # XTTS with custom voice reference
  python autodub_pro.py video.mp4 --target_lang ru --tts xtts --voice-sample my_voice.wav

  # Batch processing
  python autodub_pro.py video1.mp4 video2.mp4 --target_lang fr --parallel

  # Subtitles only
  python autodub_pro.py video.mp4 --target_lang de --subtitles-only
        """
    )

    parser.add_argument("videos", nargs="+", help="Input video file(s)")
    parser.add_argument("--target_lang", default="ru", help="Target language code (default: ru)")
    parser.add_argument("--output-dir", help="Output directory (default: current directory)")

    parser.add_argument("--whisper_model", default="turbo",
                        choices=["tiny", "base", "small", "medium", "large", "turbo"],
                        help="Whisper model size (default: turbo)")

    parser.add_argument("--translator", choices=["google", "ollama"], default="google",
                        help="Translation service (default: google)")
    parser.add_argument("--ollama_model", default="llama3",
                        help="Ollama model for translation (default: llama3)")
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel translation (faster)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")

    parser.add_argument("--tts", choices=["edge", "piper", "xtts"], default="edge",
                        help="TTS engine: edge (online), piper (offline), xtts (voice cloning)")
    parser.add_argument("--voice-sample", type=str, default=None,
                        help="Path to reference WAV for XTTS voice cloning (optional, "
                             "auto-extracted from source video if not provided)")

    parser.add_argument("--no-stretch", action="store_true",
                        help="Disable audio time-stretching")
    parser.add_argument("--detect-emotion", action="store_true",
                        help="Enable emotion detection (edge TTS only)")
    parser.add_argument("--auto-rate", action="store_true",
                        help="Auto adjust TTS rate")
    parser.add_argument("--background-volume", type=float, default=0.15,
                        help="Original audio volume (0.0-1.0, default: 0.15)")
    parser.add_argument("--voice-volume", type=float, default=1.5,
                        help="Dubbed voice volume (0.0-2.0, default: 1.5)")

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

    first_video = Path(args.videos[0])
    work_dir = Path.cwd() / f"{first_video.stem}_work"
    work_dir.mkdir(exist_ok=True)
    logger = Logger(work_dir)

    logger.info("╔════════════════════════════════════════════════════════╗")
    logger.info("║           AutoDub Pro v4.0 - Configuration           ║")
    logger.info("╚════════════════════════════════════════════════════════╝")
    logger.info(f"📹 Videos: {len(args.videos)}")
    logger.info(f"🌍 Target Language: {args.target_lang}")
    logger.info(f"🎙️  TTS Engine: {args.tts}")
    if args.tts == "xtts":
        logger.info(f"🎤 Voice Sample: {args.voice_sample or 'auto-extract from video'}")
    logger.info(f"🔤 Translator: {args.translator}")
    logger.info(f"🧠 Whisper Model: {args.whisper_model}")
    logger.info(f"⚡ Parallel Processing: {'Yes' if args.parallel else 'No'}")
    logger.info(f"🎭 Emotion Detection: {'Yes' if args.detect_emotion else 'No'}")
    logger.info(f"📈 Auto Rate Adjust: {'Yes' if args.auto_rate else 'No'}")
    logger.info(f"🎵 Audio Stretching: {'Yes' if not args.no_stretch else 'No'}")
    logger.info("")

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