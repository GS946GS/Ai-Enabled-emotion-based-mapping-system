"""
input_pipeline_fasterwhisper.py

- Captures microphone audio using PyAudio + webrtcvad
- Runs offline ASR using faster-whisper
- Translates recognized text to English using Seamless M4T
- Falls back to Silero TTS for English audio if Seamless can't produce speech
- Returns a dict with recognized_text, english_text, english_audio
"""

import os
import gc
import time
import numpy as np
import pyaudio
import webrtcvad
import torch
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model
from aksharamukha import transliterate
from faster_whisper import WhisperModel

# ---------------------------
# CONFIG
# ---------------------------
MODEL_DIR = r"C:\Users\sinth\seamless-m4t-v2-large"   # your Seamless model path
SAMPLE_RATE = 16000
OUTPUT_DIR = "translated_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_LANG = "eng"   # Always target English

# Choose faster-whisper model size: "small", "base", "medium", "large-v2" etc.
FW_MODEL_SIZE = "small"   # change to "medium" or "base" for better accuracy if you want

# ---------------------------
# Languages table
# ---------------------------
indian_langs = {
    "hin": "Hindi", "tel": "Telugu", "tam": "Tamil", "ben": "Bengali",
    "mar": "Marathi", "pan": "Punjabi", "guj": "Gujarati", "kan": "Kannada",
    "mal": "Malayalam", "ory": "Odia", "urd": "Urdu", "asm": "Assamese",
    "npi": "Nepali", "mai": "Maithili", "bho": "Bhojpuri", "san": "Sanskrit",
    "mni": "Meitei", "sat": "Santali", "raj": "Rajasthani"
}

speech_supported = {
    "arb","ben","cat","ces","cmn","cym","dan","deu","eng","est",
    "fin","fra","hin","ind","ita","jpn","kor","mlt","nld","pes","pol",
    "por","ron","rus","slk","spa","swe","swh","tel","tgl","tha",
    "tur","ukr","urd","uzn","vie"
}

romanize_map = {
    "hin": ("Devanagari", "ISO"), "tam": ("Tamil", "ISO"), "tel": ("Telugu", "ISO"),
    "mal": ("Malayalam", "ISO"), "kan": ("Kannada", "ISO"), "guj": ("Gujarati", "ISO"),
    "ben": ("Bengali", "ISO"), "ory": ("Odia", "ISO"), "mni": ("Bengali", "ISO"),
    "raj": ("Devanagari", "ISO")
}

silero_speakers = {
    "hin": "hindi_male", "tam": "tamil_male", "tel": "telugu_male",
    "mal": "malayalam_male", "kan": "kannada_male", "guj": "gujarati_male",
    "ben": "bengali_male", "ory": "oriya_male", "mni": "manipuri_female"
}

# ---------------------------
# LANGUAGE SELECTOR
# ---------------------------
def select_source_language():
    print("\n==========================")
    print("   SELECT INPUT LANGUAGE")
    print("==========================")
    for code, name in indian_langs.items():
        print(f"{code} → {name}")
    print("---------------------------")

    while True:
        lang = input("Enter language code: ").strip().lower()
        if lang in indian_langs:
            print(f"✔ Selected: {indian_langs[lang]} ({lang})")
            return lang
        print("❌ Invalid code. Try again.\n")

# ---------------------------
# LOAD Seamless model (safe)
# ---------------------------
def load_seamless(model_dir):
    gc.collect()
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Seamless processor & model on {device} ...")
    processor = AutoProcessor.from_pretrained(model_dir)
    try:
        model = SeamlessM4Tv2Model.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto")
    except Exception:
        model = SeamlessM4Tv2Model.from_pretrained(model_dir, torch_dtype=torch.float32, device_map="cpu")
    model.eval()
    return processor, model, device

print("🔁 Loading Seamless model (may take a while)...")
processor, seamless_model, seamless_device = load_seamless(MODEL_DIR)

# ---------------------------
# LOAD faster-whisper
# ---------------------------
# Device selection for faster-whisper: "cuda" if available else "cpu"
fw_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔁 Loading faster-whisper model '{FW_MODEL_SIZE}' on {fw_device} ...")
fw_model = WhisperModel(FW_MODEL_SIZE, device=fw_device, compute_type="float16" if fw_device=="cuda" else "int8")

# ---------------------------
# VOICE CAPTURE: PyAudio + webrtcvad
# ---------------------------
def capture_speech(sample_rate=SAMPLE_RATE, vad_mode=2, chunk_ms=30, max_silence=1.0, max_duration=30.0):
    """
    Record from microphone until silence is detected. Returns float32 numpy array (mono, SAMPLE_RATE).
    """
    vad = webrtcvad.Vad(vad_mode)
    pa = pyaudio.PyAudio()

    frame_size = int(sample_rate * chunk_ms / 1000)
    stream = pa.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=sample_rate,
                     input=True,
                     frames_per_buffer=frame_size)

    print("\n🎤 Speak now (recording). Press Ctrl+C to cancel.")

    frames = []
    speaking = False
    silence_time = 0.0
    start_time = time.time()

    try:
        while True:
            data = stream.read(frame_size, exception_on_overflow=False)
            frames.append(data)

            is_speech = vad.is_speech(data, sample_rate)

            if is_speech:
                speaking = True
                silence_time = 0.0
            else:
                if speaking:
                    silence_time += chunk_ms / 1000.0

            if speaking and silence_time >= max_silence:
                break

            if time.time() - start_time > max_duration:
                print("⚠️ Max duration reached, stopping capture.")
                break

    except KeyboardInterrupt:
        print("Recording cancelled by user.")

    stream.stop_stream()
    stream.close()
    pa.terminate()

    # bytes -> int16 -> float32 normalized [-1,1]
    audio_bytes = b"".join(frames)
    if len(audio_bytes) == 0:
        return np.zeros(0, dtype=np.float32)
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    # If capture sample rate differs from model expectation, resample:
    if SAMPLE_RATE != 16000:
        audio_float32 = torchaudio.functional.resample(torch.from_numpy(audio_float32), orig_freq=SAMPLE_RATE, new_freq=16000).numpy()

    return audio_float32

# ---------------------------
# Silero TTS fallback (English)
# ---------------------------
def silero_tts_english(text):
    # simple English Silero TTS via torch.hub
    model, _ = torch.hub.load("snakers4/silero-models", "silero_tts", language="en", speaker="v3_en")
    audio = model.apply_tts(text=text, speaker="en_0", sample_rate=24000)
    out_path = os.path.join(OUTPUT_DIR, "english_silero.wav")
    torchaudio.save(out_path, torch.tensor(audio).unsqueeze(0), 24000)
    return out_path

# ---------------------------
# Seamless translation wrapper
# ---------------------------
def seamless_translate_text(src_lang, text):
    """
    Input: text in src_lang
    Output: dict with english_text (if available), english_audio (path) if Seamless produced speech
    """
    print("\n🌍 Translating to English using Seamless...")
    inputs = processor(text=text, src_lang=src_lang, return_tensors="pt").to(seamless_device)

    if OUTPUT_LANG in speech_supported:
        # Seamless can generate English speech directly
        out = seamless_model.generate(**inputs, tgt_lang=OUTPUT_LANG)[0].cpu().numpy().squeeze()
        out = out.astype("float32")
        path = os.path.join(OUTPUT_DIR, "english_seamless.wav")
        torchaudio.save(path, torch.tensor(out).unsqueeze(0), SAMPLE_RATE)
        return {"english_text": None, "english_audio": path}

    # Fallback: generate English text
    tokens = seamless_model.generate(**inputs, tgt_lang=OUTPUT_LANG, generate_speech=False)
    english_text = processor.decode(tokens[0].tolist()[0], skip_special_tokens=True)
    return {"english_text": english_text, "english_audio": None}

# ---------------------------
# ASR with faster-whisper
# ---------------------------
def asr_with_faster_whisper(audio_np, initial_language=None):
    """
    audio_np: 1D numpy float32 array (sample rate SAMPLE_RATE)
    returns recognized_text (string)
    """
    if audio_np.size == 0:
        return ""

    # faster-whisper expects either file path or numpy array + sample rate
    # We pass numpy array and sample_rate param
    segments, info = fw_model.transcribe(audio_np, beam_size=15, language=None)  # let model detect language
    texts = []
    for segment in segments:
        texts.append(segment.text)
    recognized_text = " ".join(texts).strip()
    return recognized_text

# ---------------------------
# Main process_input wrapper
# ---------------------------
def process_input(src_lang):
    """
    Captures audio, runs ASR (faster-whisper), then translates with Seamless,
    and ensures english audio exists (via Seamless or Silero fallback).

    Returns:
        {
            "recognized_text": "...",    # original language text (from ASR)
            "english_text": "...",       # english translation (if available)
            "english_audio": "path.wav"  # path to english tts
        }
    """
    audio = capture_speech()

    print("\n🔎 Running ASR (faster-whisper)...")
    recognized_text = asr_with_faster_whisper(audio)
    print("📝 Recognized Text:", recognized_text)

    # send recognized_text to Seamless for translation
    trans_result = seamless_translate_text(src_lang, recognized_text)

    # if Seamless produced only audio, we might want to also get text — for now return audio and None text
    if trans_result["english_audio"] is None and trans_result["english_text"] is not None:
        # produce english audio via Silero fallback
        print("\n🔊 Producing English audio via Silero fallback...")
        audio_path = silero_tts_english(trans_result["english_text"])
        trans_result["english_audio"] = audio_path

    return {
        "recognized_text": recognized_text,
        "english_text": trans_result["english_text"],
        "english_audio": trans_result["english_audio"]
    }

# ---------------------------
# Run as script
# ---------------------------
if __name__ == "__main__":
    print("=== Input Pipeline (faster-whisper + Seamless) ===")

    # Select input language only once
    src_lang = select_source_language()

    while True:
        print("\n🎤 Ready for new speech input (or type 'q' to quit).")
        ch = input("Press Enter to speak or 'q' to quit: ").strip().lower()
        if ch == "q":
            print("👋 Exiting pipeline...")
            break

        output = process_input(src_lang)

        print("\n=============================")
        print(" Source Text    :", output["recognized_text"])
        print(" English Text   :", output["english_text"])
        print(" English Audio  :", output["english_audio"])
        print("=============================\n")

