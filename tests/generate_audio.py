"""
Генерация тестовых WAV-файлов из текстовых запросов через gTTS.
Запуск: python tests/generate_audio.py
"""
import os
import sys
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio_perf")
os.makedirs(AUDIO_DIR, exist_ok=True)

QUERIES = [
    "Расскажи про Спутник один",
    "Что такое Метеор М?",
    "Как устроена антенна спутника?",
    "Для чего нужны солнечные панели?",
    "Какова масса Спутника один?",
    "Расскажи об орбите Метеор М",
    "Что такое космический аппарат?",
    "Как работает система связи спутника?",
    "Когда был запущен Спутник один?",
    "Какие подсистемы есть у Метеор М?",
]


def generate():
    try:
        from gtts import gTTS
    except ImportError:
        print("Установи gtts: pip install gtts")
        sys.exit(1)

    try:
        import subprocess
    except ImportError:
        pass

    print(f"Генерирую {len(QUERIES)} аудиофайлов в {AUDIO_DIR}/")

    for i, text in enumerate(QUERIES):
        out_path = os.path.join(AUDIO_DIR, f"query_{i:02d}.wav")
        if os.path.exists(out_path):
            print(f"  [{i+1}/{len(QUERIES)}] уже существует: query_{i:02d}.wav")
            continue

        # Генерируем MP3 через gTTS
        mp3_path = out_path.replace(".wav", ".mp3")
        tts = gTTS(text=text, lang="ru", slow=False)
        tts.save(mp3_path)

        # Конвертируем в WAV через ffmpeg
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ac", "1", "-ar", "16000", out_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.remove(mp3_path)

        if result.returncode == 0:
            print(f"  [{i+1}/{len(QUERIES)}] ✅ query_{i:02d}.wav — «{text[:40]}»")
        else:
            print(f"  [{i+1}/{len(QUERIES)}] ❌ ошибка ffmpeg для «{text[:40]}»")

    print(f"\nГотово. Файлы в: {AUDIO_DIR}")


if __name__ == "__main__":
    generate()
