"""
4.5 Тестирование распознавания речи
Запуск: pytest tests/test_asr.py -v -s

Для запуска нужны аудиофайлы в tests/audio/.
Если файлов нет — тесты пропускаются.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio")

# (имя файла, ожидаемая транскрипция, тип запроса)
ASR_TEST_CASES = [
    # Команды управления
    ("rotate.wav",      "покрути спутник",                  "action"),
    ("stop.wav",        "останови вращение",                 "action"),
    ("increase.wav",    "увеличь модель",                    "action"),
    ("decrease.wav",    "уменьши спутник",                   "action"),
    ("reset.wav",       "верни как было",                    "action"),
    ("animation.wav",   "запусти анимацию",                  "action"),
    ("rotate2.wav",     "можешь повернуть его",              "action"),
    ("stop2.wav",       "перестань вращать",                 "action"),
    ("increase2.wav",   "сделай модель больше",              "action"),
    ("animation2.wav",  "включи анимацию спутника",          "action"),
    # Информационные вопросы
    ("info.wav",        "расскажи про спутник один",         "info"),
    ("antenna.wav",     "что такое антенна",                 "info"),
    ("meteor.wav",      "расскажи о метеор м",               "info"),
    ("mass.wav",        "какова масса спутника один",        "info"),
    ("orbit.wav",       "расскажи об орбите",                "info"),
    ("panels.wav",      "для чего нужны солнечные панели",   "info"),
    ("launch.wav",      "когда был запущен спутник один",    "info"),
    ("purpose.wav",     "для чего предназначен метеор м",    "info"),
    ("structure.wav",   "как устроен спутник",               "info"),
    ("transmitter.wav", "что такое передатчик спутника",     "info"),
]


def normalize(text: str) -> str:
    import re
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def wer(reference: str, hypothesis: str) -> float:
    ref_words = normalize(reference).split()
    hyp_words = normalize(hypothesis).split()
    if not ref_words:
        return 0.0
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


@pytest.mark.skipif(not os.path.isdir(AUDIO_DIR), reason="Папка tests/audio/ не найдена")
@pytest.mark.parametrize("filename,expected,query_type", ASR_TEST_CASES)
def test_asr_transcription(filename, expected, query_type):
    from app.query.audio import transcribe_audio_bytes

    audio_path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(audio_path):
        pytest.skip(f"Файл {filename} не найден")

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    transcript = transcribe_audio_bytes(audio_bytes, language="ru").lower().strip()
    error_rate = wer(expected, transcript)

    assert error_rate < 0.5, (
        f"WER={error_rate:.2f} слишком высокий\n"
        f"Тип: {query_type}\n"
        f"Ожидалось: «{expected}»\n"
        f"Получено:  «{transcript}»"
    )


@pytest.mark.skipif(not os.path.isdir(AUDIO_DIR), reason="Папка tests/audio/ не найдена. Запусти: python tests/generate_asr_audio.py")
def test_asr_report():
    """Генерирует отчёт WER для таблицы главы 4."""
    from app.query.audio import transcribe_audio_bytes

    print("\n\n=== ОТЧЁТ: Распознавание речи ===\n")
    print(f"{'Файл':<22} {'Тип':<8} {'Ожидалось':<35} {'Получено':<35} {'WER':>5} {'OK':>4}")
    print("─" * 115)

    by_type: dict[str, list[float]] = {"action": [], "info": []}

    for filename, expected, query_type in ASR_TEST_CASES:
        audio_path = os.path.join(AUDIO_DIR, filename)
        if not os.path.exists(audio_path):
            print(f"{filename:<22} — файл не найден —")
            continue

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        transcript = transcribe_audio_bytes(audio_bytes, language="ru").lower().strip()
        error_rate = wer(expected, transcript)
        by_type[query_type].append(error_rate)
        ok_mark = "✅" if error_rate < 0.3 else "❌"
        print(f"{filename:<22} {query_type:<8} {expected:<35} {transcript:<35} {error_rate:>5.2f} {ok_mark:>4}")

    print("─" * 115)

    # Сводная таблица по типам
    print("\n=== СВОДНАЯ ТАБЛИЦА (для диплома) ===\n")
    print(f"{'Тип запроса':<25} {'Число записей':>15} {'Среднее WER':>13} {'Доля верно (WER<0.3)':>22}")
    print("─" * 80)

    all_scores = []
    for label, key in [("Команды управления", "action"), ("Информационные вопросы", "info")]:
        scores = by_type[key]
        if not scores:
            continue
        avg_wer = sum(scores) / len(scores)
        correct = sum(1 for s in scores if s < 0.3)
        all_scores.extend(scores)
        print(f"{label:<25} {len(scores):>15} {avg_wer:>13.2f} {correct}/{len(scores):>20}")

    if all_scores:
        avg_total = sum(all_scores) / len(all_scores)
        correct_total = sum(1 for s in all_scores if s < 0.3)
        print("─" * 80)
        print(f"{'Итого':<25} {len(all_scores):>15} {avg_total:>13.2f} {correct_total}/{len(all_scores):>20}")
