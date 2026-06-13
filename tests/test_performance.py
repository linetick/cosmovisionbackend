"""
4.4.1 Оценка производительности под нагрузкой
Запуск: pytest tests/test_performance.py -v -s
"""
import sys
import os
import time
import statistics
import threading
import requests
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from tests.conftest import BASE_URL

CONCURRENT_LEVELS = [1, 10, 30, 100]
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio_perf")

LOAD_QUERIES = [
    "Расскажи про Спутник-1",
    "Что такое Метеор-М?",
    "Как устроена антенна спутника?",
    "Для чего нужны солнечные панели?",
    "Какова масса Спутника-1?",
    "Расскажи об орбите Метеор-М",
    "Что такое космический аппарат?",
    "Как работает система связи спутника?",
    "Когда был запущен Спутник-1?",
    "Какие подсистемы есть у Метеор-М?",
    "Расскажи о передатчике Спутника-1",
    "Что такое солнечно-синхронная орбита?",
    "Расскажи о корпусе Спутника-1",
    "Как устроено энергопитание спутника?",
    "Назначение Спутника-1",
    "Опиши антенны Спутника-1",
    "Расскажи о гидрометеорологических спутниках",
    "Что передаёт Метеор-М на землю?",
    "Какой диаметр у Спутника-1?",
    "Расскажи о запуске первого спутника",
    "Как спутник удерживается на орбите?",
    "Что такое телеметрия спутника?",
    "Расскажи о батарее Спутника-1",
    "Какова скорость Спутника-1 на орбите?",
    "Для чего нужен передатчик на спутнике?",
    "Расскажи о вращении Земли и спутниках",
    "Что такое искусственный спутник?",
    "Какой период обращения у Спутника-1?",
    "Расскажи о Советской космической программе",
    "Чем Метеор-М отличается от Спутника-1?",
]


# ── Параллельная нагрузка ─────────────────────────────────────────────────────

def _send_query(headers: dict, results: list, idx: int) -> None:
    query = LOAD_QUERIES[idx % len(LOAD_QUERIES)]
    t0 = time.time()
    try:
        r = requests.post(
            f"{BASE_URL}/query",
            json={"text": query},
            headers=headers,
            timeout=60,
        )
        elapsed = time.time() - t0
        status = r.status_code
        ok = status == 200
        rate_limited = status == 429
        timing = r.json().get("timing", {}) if ok else {}
        results[idx] = {
            "elapsed": elapsed,
            "ok": ok,
            "status": status,
            "rate_limited": rate_limited,
            "timing": timing,
        }
    except requests.exceptions.Timeout:
        results[idx] = {"elapsed": time.time() - t0, "ok": False, "status": 0, "rate_limited": False, "timeout": True, "timing": {}}
    except Exception:
        results[idx] = {"elapsed": time.time() - t0, "ok": False, "status": -1, "rate_limited": False, "timing": {}}


def run_concurrent(n: int, headers: dict) -> list:
    """Отправляет n параллельных запросов и возвращает список результатов."""
    results = [None] * n
    threads = [
        threading.Thread(target=_send_query, args=(headers, results, i))
        for i in range(n)
    ]
    # Сбрасываем кэш перед каждым прогоном
    requests.post(f"{BASE_URL}/debug/cache/clear", headers=headers)

    t_start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_time = time.time() - t_start

    return results, wall_time


@pytest.mark.parametrize("n_users", [1, 10])
def test_concurrent_latency(n_users, headers):
    """Базовые уровни нагрузки — запросы должны завершиться без таймаутов."""
    results, _ = run_concurrent(n_users, headers)
    timeouts = [r for r in results if r.get("timeout")]
    assert not timeouts, f"Из {n_users} запросов зависло {len(timeouts)}"


def _avg(values: list) -> str:
    return f"{statistics.mean(values):.3f}" if values else "—"


def test_concurrent_report(headers):
    """Таблица latency с разбивкой по этапам при разном числе пользователей."""
    print("\n\n=== ОТЧЁТ: Производительность под нагрузкой ===\n")

    # Заголовок сводной таблицы
    print(
        f"{'Польз.':>7} {'Класс.(с)':>11} {'Поиск (с)':>11} {'Генер.(с)':>11} "
        f"{'Итого (с)':>11} {'Стена (с)':>11} {'OK':>8} {'429':>5} {'Err':>5}"
    )
    print("─" * 90)

    for n in CONCURRENT_LEVELS:
        results, wall = run_concurrent(n, headers)
        ok_results  = [r for r in results if r["ok"]]
        rate_limited = sum(1 for r in results if r.get("rate_limited"))
        errors       = sum(1 for r in results if not r["ok"] and not r.get("rate_limited"))

        # Собираем поэтапные времена из timing
        def stage_times(key):
            return [r["timing"][key] for r in ok_results if key in r.get("timing", {})]

        retrieve  = stage_times("retrieve")
        generate  = [v for v in stage_times("generate") if v > 0.01]  # исключаем кэш
        totals    = stage_times("total")

        # Классификация = total - normalize - retrieve - generate (неявный этап роутера)
        classify_times = []
        for r in ok_results:
            t = r.get("timing", {})
            if "total" in t and "normalize" in t and "retrieve" in t:
                gen = t.get("generate", 0)
                cls_time = t["total"] - t["normalize"] - t["retrieve"] - gen
                if cls_time > 0:
                    classify_times.append(cls_time)

        print(
            f"{n:>7} {_avg(classify_times):>11} {_avg(retrieve):>11} {_avg(generate):>11} "
            f"{_avg(totals):>11} {wall:>11.3f} {len(ok_results):>5}/{n} {rate_limited:>5} {errors:>5}"
        )

    print("\n─── Расшифровка ───────────────────────────────────")
    print("  Класс.  = классификация интента (LLM-роутер)")
    print("  Поиск   = семантический поиск в ChromaDB")
    print("  Генер.  = генерация ответа (Yandex Cloud LLM)")
    print("  Итого   = среднее полное время одного запроса")
    print("  Стена   = реальное время от старта до конца всех запросов")
    print("  429     = rate limit Yandex Cloud")


# ── Аудио нагрузка ───────────────────────────────────────────────────────────

def _send_audio_query(headers: dict, results: list, idx: int) -> None:
    audio_files = sorted([
        f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")
    ]) if os.path.isdir(AUDIO_DIR) else []

    if not audio_files:
        results[idx] = {"elapsed": 0, "ok": False, "timing": {}, "no_files": True}
        return

    audio_path = os.path.join(AUDIO_DIR, audio_files[idx % len(audio_files)])
    t0 = time.time()
    try:
        with open(audio_path, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/query_audio",
                files={"file": ("audio.wav", f, "audio/wav")},
                headers=headers,
                timeout=60,
            )
        elapsed = time.time() - t0
        ok = r.status_code == 200
        timing = r.json().get("timing", {}) if ok else {}
        results[idx] = {"elapsed": elapsed, "ok": ok, "status": r.status_code, "timing": timing}
    except Exception:
        results[idx] = {"elapsed": time.time() - t0, "ok": False, "timing": {}}


def run_concurrent_audio(n: int, headers: dict):
    results = [None] * n
    threads = [
        threading.Thread(target=_send_audio_query, args=(headers, results, i))
        for i in range(n)
    ]
    requests.post(f"{BASE_URL}/debug/cache/clear", headers=headers)
    t_start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, time.time() - t_start


@pytest.mark.skipif(not os.path.isdir(AUDIO_DIR), reason="Папка audio_perf/ не найдена. Запусти python tests/generate_audio.py")
def test_audio_concurrent_report(headers):
    """Нагрузочный тест аудио endpoint с разбивкой по этапам."""
    print("\n\n=== ОТЧЁТ: Производительность аудио endpoint ===\n")
    print(
        f"{'Польз.':>7} {'ASR (с)':>9} {'Класс.(с)':>11} {'Поиск (с)':>11} "
        f"{'Генер.(с)':>11} {'Итого (с)':>11} {'Стена (с)':>11} {'OK':>8}"
    )
    print("─" * 90)

    for n in CONCURRENT_LEVELS:
        results, wall = run_concurrent_audio(n, headers)
        ok_results = [r for r in results if r["ok"]]

        def stage(key):
            return [r["timing"][key] for r in ok_results if key in r.get("timing", {})]

        asr      = stage("asr")
        retrieve = stage("retrieve")
        generate = [v for v in stage("generate") if v > 0.01]
        totals   = stage("total")

        classify_times = []
        for r in ok_results:
            t = r.get("timing", {})
            if all(k in t for k in ("total", "asr", "normalize", "retrieve")):
                gen = t.get("generate", 0)
                cls = t["total"] - t["asr"] - t["normalize"] - t["retrieve"] - gen
                if cls > 0:
                    classify_times.append(cls)

        print(
            f"{n:>7} {_avg(asr):>9} {_avg(classify_times):>11} {_avg(retrieve):>11} "
            f"{_avg(generate):>11} {_avg(totals):>11} {wall:>11.3f} {len(ok_results):>5}/{n}"
        )

    print("\nASR = время распознавания речи (Whisper)")


# ── ChromaDB: поиск по текущей коллекции ─────────────────────────────────────

def test_chromadb_report():
    """Замеряет время поиска и выводит статистику ChromaDB."""
    from app.query.retrieval import retrieve_hits
    from app.runtime import collection

    doc_count = collection.count()

    queries = [
        "Спутник-1 запуск",
        "антенны спутника",
        "Метеор-М назначение",
        "солнечные панели",
        "орбита спутника",
    ]

    times = []
    for q in queries:
        t0 = time.perf_counter()
        retrieve_hits(q, initial_n=3, max_n=5)
        times.append(time.perf_counter() - t0)

    print("\n\n=== ОТЧЁТ: Производительность ChromaDB ===\n")
    print(f"Документов в коллекции: {doc_count}")
    print(f"\n{'Запрос':<30} {'Время (мс)':>12}")
    print("─" * 45)
    for q, t in zip(queries, times):
        print(f"{q:<30} {t*1000:>12.1f}")
    print("─" * 45)
    print(f"{'Среднее':.<30} {statistics.mean(times)*1000:>12.1f}")
    print(f"{'Минимум':.<30} {min(times)*1000:>12.1f}")
    print(f"{'Максимум':.<30} {max(times)*1000:>12.1f}")

    print("\n--- Таблица для диплома ---")
    print(f"{'Документов в БЗ':>20} {'Среднее время поиска (мс)':>28} {'Hit@3':>8}")
    print("─" * 60)
    print(f"{doc_count:>20} {statistics.mean(times)*1000:>28.1f} {'см. 4.3':>8}")
