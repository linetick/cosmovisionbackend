"""
Генерация тестовых WAV-файлов для ASR-теста (tests/test_asr.py).
Запуск: python tests/generate_asr_audio.py
"""
import os
import sys
import subprocess

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# Имя файла → текст для TTS → тип запроса
ASR_CASES = [
    # Команды управления (10 файлов)
    ("rotate.wav",        "покрути спутник",                   "action"),
    ("stop.wav",          "останови вращение",                  "action"),
    ("increase.wav",      "увеличь модель",                     "action"),
    ("decrease.wav",      "уменьши спутник",                    "action"),
    ("reset.wav",         "верни как было",                     "action"),
    ("animation.wav",     "запусти анимацию",                   "action"),
    ("rotate2.wav",       "можешь повернуть его",               "action"),
    ("stop2.wav",         "перестань вращать",                  "action"),
    ("increase2.wav",     "сделай модель больше",               "action"),
    ("animation2.wav",    "включи анимацию спутника",           "action"),
    # Информационные вопросы (10 файлов)
    ("info.wav",          "расскажи про спутник один",          "info"),
    ("antenna.wav",       "что такое антенна",                  "info"),
    ("meteor.wav",        "расскажи о метеор м",                "info"),
    ("mass.wav",          "какова масса спутника один",         "info"),
    ("orbit.wav",         "расскажи об орбите",                 "info"),
    ("panels.wav",        "для чего нужны солнечные панели",    "info"),
    ("launch.wav",        "когда был запущен спутник один",     "info"),
    ("purpose.wav",       "для чего предназначен метеор м",     "info"),
    ("structure.wav",     "как устроен спутник",                "info"),
    ("transmitter.wav",   "что такое передатчик спутника",      "info"),
]


def generate():
    try:
        from gtts import gTTS
    except ImportError:
        print("Установи gtts: pip install gtts")
        sys.exit(1)

    print(f"Генерирую {len(ASR_CASES)} аудиофайлов в {AUDIO_DIR}/\n")
    action_count = sum(1 for _, _, t in ASR_CASES if t == "action")
    info_count   = sum(1 for _, _, t in ASR_CASES if t == "info")
    print(f"  Команды управления:    {action_count} файлов")
    print(f"  Информационные вопросы: {info_count} файлов\n")

    ok = 0
    for filename, text, query_type in ASR_CASES:
        out_path = os.path.join(AUDIO_DIR, filename)
        if os.path.exists(out_path):
            print(f"  ✅ уже существует: {filename}")
            ok += 1
            continue

        mp3_path = out_path.replace(".wav", ".mp3")
        try:
            tts = gTTS(text=text, lang="ru", slow=False)
            tts.save(mp3_path)

            result = subprocess.run(
                ["ffmpeg", "-y", "-i", mp3_path,
                 "-ac", "1", "-ar", "16000", out_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if os.path.exists(mp3_path):
                os.remove(mp3_path)

            if result.returncode == 0:
                print(f"  ✅ [{query_type}] {filename} — «{text}»")
                ok += 1
            else:
                print(f"  ❌ ffmpeg ошибка: {filename}")
        except Exception as e:
            print(f"  ❌ {filename}: {e}")

    print(f"\nГотово: {ok}/{len(ASR_CASES)} файлов в {AUDIO_DIR}/")
    print("\nЗапусти тест:")
    print("  pytest tests/test_asr.py::test_asr_report -v -s")


if __name__ == "__main__":
    generate()
