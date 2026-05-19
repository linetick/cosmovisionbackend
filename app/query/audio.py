import os
import subprocess
import tempfile
import time

from ..config import WHISPER_BEAM_SIZE, WHISPER_VAD_FILTER
from ..runtime import get_whisper_model


def wav_to_16k_mono_wav_bytes(wav_bytes: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fin:
        fin.write(wav_bytes)
        fin.flush()
        in_path = fin.name

    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(out_fd)

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", "16000", "-f", "wav", out_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        with open(out_path, "rb") as file:
            return file.read()
    finally:
        for path in (in_path, out_path):
            try:
                os.remove(path)
            except OSError:
                pass


def transcribe_audio_bytes_detailed(data: bytes, language: str = "ru") -> tuple[str, dict]:
    stats = {
        "audio_prepare": 0.0,
        "transcribe": 0.0,
        "total": 0.0,
    }

    t_prepare = time.perf_counter()
    data = wav_to_16k_mono_wav_bytes(data)
    stats["audio_prepare"] = time.perf_counter() - t_prepare

    t_transcribe = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        segments, _ = get_whisper_model().transcribe(
            tmp.name,
            language=language,
            vad_filter=WHISPER_VAD_FILTER,
            beam_size=WHISPER_BEAM_SIZE,
            condition_on_previous_text=False,
        )
        transcript = " ".join(seg.text for seg in segments).strip()
    stats["transcribe"] = time.perf_counter() - t_transcribe
    stats["total"] = stats["audio_prepare"] + stats["transcribe"]
    return transcript, stats


def transcribe_audio_bytes(data: bytes, language: str = "ru") -> str:
    transcript, _ = transcribe_audio_bytes_detailed(data, language=language)
    return transcript
