#!/usr/bin/env python3
"""
AutoDub - Automatic Video Translation and Dubbing Tool
Transcribes, translates, and dubs videos using AI
"""

import sys, os, asyncio, subprocess, argparse
import whisper, pysrt, edge_tts
from deep_translator import GoogleTranslator
from datetime import timedelta
from tqdm import tqdm
import functools
import torch
import torchaudio

try:
    import soundfile as sf
except ImportError:
    print("Installing soundfile...")
    subprocess.run([sys.executable, "-m", "pip", "install", "soundfile"], check=True)
    import soundfile as sf

# PyTorch 2.6+ compatibility patches
torch.load = functools.partial(torch.load, weights_only=False)

def forced_load(uri, **kwargs):
    """Replacement for torchaudio.load to avoid TorchCodec dependency"""
    data, samplerate = sf.read(uri)
    tensor = torch.from_numpy(data).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.transpose(0, 1)
    return tensor, samplerate

torchaudio.load = forced_load
os.environ["COQUI_TOS_AGREED"] = "1"

# Voice configurations
EDGE_VOICES = {
    'ru': 'ru-RU-DmitryNeural',
    'en': 'en-US-ChristopherNeural', 
    'de': 'de-DE-ConradNeural',
    'fr': 'fr-FR-HenriNeural',
    'es': 'es-ES-AlvaroNeural',
    'it': 'it-IT-DiegoNeural',
    'pt': 'pt-BR-AntonioNeural',
    'ja': 'ja-JP-KeitaNeural',
    'zh': 'zh-CN-YunxiNeural'
}

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format"""
    td = timedelta(seconds=seconds)
    return f"{td.seconds//3600:02}:{(td.seconds//60)%60:02}:{td.seconds%60:02},{td.microseconds//1000:03}"

def generate_xtts(subs, ref_wav, lang, concat_list, temp_files):
    """Generate speech using XTTS voice cloning"""
    from TTS.api import TTS
    print("[XTTS] Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    
    ref_short = "temp_ref_short.wav"
    print("[XTTS] Preparing voice sample...")
    subprocess.run([
        "ffmpeg", "-y", "-i", ref_wav, "-t", "10", 
        "-ar", "22050", "-ac", "1", ref_short
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    temp_files.append(ref_short)
    
    current_time_ms = 0
    for i, sub in enumerate(tqdm(subs, desc="XTTS Synthesis")):
        start_ms = (sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds) * 1000 + sub.start.milliseconds
        text = sub.text.replace("\n", " ").strip()
        if not text: continue
        
        sil_dur = start_ms - current_time_ms
        if sil_dur > 100:
            f = f"temp_sil_x_{i}.wav"
            subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", str(sil_dur/1000), f], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            concat_list.append(f"file '{f}'"); temp_files.append(f); current_time_ms += sil_dur
        
        f_out = f"temp_phrase_x_{i}.wav"
        tts_model.tts_to_file(text=text, speaker_wav=ref_short, language=lang, file_path=f_out)
        concat_list.append(f"file '{f_out}'"); temp_files.append(f_out)
        current_time_ms = (sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds) * 1000 + sub.end.milliseconds

def generate_silero(subs, voice, concat_list, temp_files, model):
    """Generate speech using Silero TTS"""
    current_time_ms = 0
    for i, sub in enumerate(tqdm(subs, desc="Silero Synthesis")):
        start_ms = (sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds) * 1000 + sub.start.milliseconds
        text = sub.text.replace("\n", " ").strip()
        if not text: continue
        
        sil_dur = start_ms - current_time_ms
        if sil_dur > 100:
            f = f"temp_sil_s_{i}.wav"
            subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", str(sil_dur/1000), f], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            concat_list.append(f"file '{f}'"); temp_files.append(f); current_time_ms += sil_dur
        
        audio = model.apply_tts(text=text, speaker=voice, sample_rate=48000)
        f_out = f"temp_phrase_s_{i}.wav"
        audio_np = audio.cpu().numpy()
        sf.write(f_out, audio_np, 48000)
        concat_list.append(f"file '{f_out}'"); temp_files.append(f_out)
        current_time_ms = (sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds) * 1000 + sub.end.milliseconds

async def main():
    parser = argparse.ArgumentParser(
        description="AutoDub - Automatic video translation and dubbing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4                                    # Dub to Russian with Edge TTS
  %(prog)s video.mp4 --target_lang en                   # Dub to English
  %(prog)s video.mp4 --tts silero --silero_voice baya   # Use Silero TTS
  %(prog)s video.mp4 --tts xtts --ref_voice voice.wav   # Clone voice with XTTS
        """
    )
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--tts", choices=["edge", "silero", "xtts"], default="edge",
                       help="TTS engine (default: edge)")
    parser.add_argument("--target_lang", default="ru",
                       help="Target language code (default: ru)")
    parser.add_argument("--silero_voice", default="aidar",
                       choices=["aidar", "baya", "kseniya", "xenia", "eugene"],
                       help="Silero voice for Russian (default: aidar)")
    parser.add_argument("--ref_voice", help="Reference WAV for XTTS (6-10 sec clean speech)")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    args = parser.parse_args()
    
    if args.tts == "xtts" and not args.ref_voice:
        print("❌ Error: --ref_voice required for XTTS")
        return 1
    
    base = os.path.splitext(args.video)[0]
    audio_wav = "temp_audio.wav"
    srt_file = f"{base}_{args.target_lang}.srt"
    voiceover_wav = "temp_voiceover.wav"
    output = f"{base}_dubbed.mp4"
    
    # 1. Extract audio
    print("[1/5] Extracting audio...")
    subprocess.run(["ffmpeg", "-y", "-i", args.video, "-vn", "-acodec", "pcm_s16le", 
                   "-ar", "16000", "-ac", "1", audio_wav], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    # 2. Transcribe
    print("[2/5] Transcribing with Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_wav, fp16=False)
    
    # 3. Translate
    print("[3/5] Translating...")
    with open(srt_file, "w", encoding="utf-8") as f:
        for i, seg in enumerate(tqdm(result["segments"])):
            translated = GoogleTranslator(source='auto', target=args.target_lang).translate(seg['text'])
            f.write(f"{i+1}\n{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n{translated}\n\n")
    
    # 4. Synthesize speech
    print("[4/5] Synthesizing speech...")
    subs = pysrt.open(srt_file)
    concat_list, temp_files = [], []
    
    if args.tts == "edge":
        voice = EDGE_VOICES.get(args.target_lang, "en-US-ChristopherNeural")
        async def run_edge():
            curr = 0
            for i, s in enumerate(tqdm(subs, desc="Edge TTS")):
                start = (s.start.hours*3600+s.start.minutes*60+s.start.seconds)*1000 + s.start.milliseconds
                txt = s.text.strip()
                if not txt: continue
                
                sdur = start - curr
                if sdur > 100:
                    fn = f"tsil_{i}.wav"
                    subprocess.run(["ffmpeg","-y","-f","lavfi","-i","anullsrc=r=24000:cl=mono","-t",str(sdur/1000),fn], 
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    concat_list.append(f"file '{fn}'"); temp_files.append(fn); curr += sdur
                
                fn = f"tph_{i}.mp3"
                await edge_tts.Communicate(txt, voice).save(fn)
                concat_list.append(f"file '{fn}'"); temp_files.append(fn)
                curr = (s.end.hours*3600+s.end.minutes*60+s.end.seconds)*1000 + s.end.milliseconds
        await run_edge()
    
    elif args.tts == "silero":
        print("[Silero] Loading model...")
        device = torch.device('cpu')
        result = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='ru',
            speaker='v4_ru',
            trust_repo=True
        )
        silero_model = result[0] if isinstance(result, tuple) else result
        silero_model.to(device)
        generate_silero(subs, args.silero_voice, concat_list, temp_files, silero_model)
    
    elif args.tts == "xtts":
        generate_xtts(subs, args.ref_voice, args.target_lang, concat_list, temp_files)
    
    # Concatenate audio
    with open("list.txt", "w") as f: 
        f.write("\n".join(concat_list))
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt", 
                   "-c", "copy", voiceover_wav], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    # 5. Merge with video
    print("[5/5] Merging audio with video...")
    subprocess.run([
        "ffmpeg", "-y", "-i", args.video, "-i", voiceover_wav, 
        "-filter_complex", "[0:a]volume=0.2[bg];[1:a]volume=1.5[fg];[bg][fg]amix=inputs=2:duration=first", 
        "-c:v", "copy", "-map", "0:v:0", output
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    # Cleanup
    if not args.keep_temp:
        for f in [audio_wav, voiceover_wav, "list.txt"] + temp_files:
            if os.path.exists(f): 
                os.remove(f)
    
    print(f"✅ Done! Output: {output}")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
