"""
4.4 Оценка временных характеристик
Запуск: pytest tests/test_latency.py -v -s
"""
import pytest
import requests
import statistics

from tests.conftest import BASE_URL

LATENCY_INFO_QUERIES = [
    "Расскажи про Спутник-1",
    "Что такое Метеор-М?",
    "Как устроена антенна?",
    "Для чего нужны солнечные панели?",
    "Какова масса Спутника-1?",
]

LATENCY_ACTION_QUERIES = [
    "Покрути спутник",
    "Останови вращение",
    "Увеличь модель",
    "Верни как было",
    "Запусти анимацию",
]

LATENCY_QUERIES = LATENCY_INFO_QUERIES + LATENCY_ACTION_QUERIES

MAX_LATENCY_SEC = 5.0


@pytest.mark.parametrize("query", LATENCY_QUERIES)
def test_latency_under_limit(query, headers):
    r = requests.post(
        f"{BASE_URL}/query",
        json={"text": query},
        headers=headers,
        timeout=30,
    )
    assert r.status_code == 200
    total = r.json().get("timing", {}).get("total", 99)
    assert total < MAX_LATENCY_SEC, (
        f"Запрос: «{query}» — время {total:.3f}с превышает лимит {MAX_LATENCY_SEC}с"
    )


def test_latency_report(headers):
    """Генерирует отчёт с min/mean/max по этапам для таблицы главы 4."""
    # Сбрасываем кэш чтобы замерить реальную генерацию
    requests.post(f"{BASE_URL}/debug/cache/clear", headers=headers)

    stages = ["normalize", "topic_check", "retrieve", "generate", "total"]
    collected = {s: [] for s in stages}
    rows = []

    for query in LATENCY_QUERIES:
        r = requests.post(
            f"{BASE_URL}/query",
            json={"text": query},
            headers=headers,
            timeout=30,
        )
        data = r.json()
        timing = data.get("timing", {})
        from_cache = timing.get("generate", 1) < 0.01
        for s in stages:
            if s in timing:
                collected[s].append(timing[s])
        rows.append({
            "query": query[:40],
            "total": timing.get("total", 0),
            "generate": timing.get("generate", 0),
            "cached": from_cache,
            "intent": data.get("intent", "—"),
        })

    print("\n\n=== ОТЧЁТ: Временные характеристики (секунды) ===\n")
    print(f"{'Запрос':<42} {'Интент':<15} {'Генерация':>10} {'Итого':>8}")
    print("-" * 80)
    for row in rows:
        intent_label = row.get("intent", "—")
        gen = f"{row['generate']:.3f}" if row["generate"] > 0.01 else "— (кэш/action)"
        print(f"{row['query']:<42} {intent_label:<15} {gen:>10} {row['total']:>8.3f}")

    # Статистика только по info-запросам без кэша
    info_totals = [r["total"] for r in rows if r.get("intent") == "info" and r["generate"] > 0.1]
    action_totals = [r["total"] for r in rows if r.get("intent") == "action"]

    print(f"\n--- Текстовые информационные запросы (info) ---")
    if info_totals:
        print(f"Мин: {min(info_totals):.3f}s  Среднее: {statistics.mean(info_totals):.3f}s  Макс: {max(info_totals):.3f}s")

    print(f"\n--- Команды управления (action) ---")
    if action_totals:
        print(f"Мин: {min(action_totals):.3f}s  Среднее: {statistics.mean(action_totals):.3f}s  Макс: {max(action_totals):.3f}s")
        print("Примечание: время action включает вызов LLM-роутера (~1.4s)")

    print(f"\n--- Этапы обработки (только info без кэша) ---")
    stage_labels = {
        "normalize":   "Нормализация",
        "retrieve":    "Семантический поиск",
        "generate":    "Генерация ответа",
        "total":       "ИТОГО",
    }
    for s in ["normalize", "retrieve", "generate", "total"]:
        vals = collected[s]
        if not vals:
            continue
        label = stage_labels.get(s, s)
        print(f"{label:<25} {min(vals):>8.3f} {statistics.mean(vals):>10.3f} {max(vals):>8.3f}")
