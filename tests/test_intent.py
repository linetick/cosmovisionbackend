"""
4.2 Тестирование классификации интентов
Запуск: pytest tests/test_intent.py -v
"""
import pytest
import requests
from collections import defaultdict

from tests.conftest import BASE_URL, INTENT_TEST_CASES


def classify(query: str, headers: dict) -> str:
    r = requests.post(
        f"{BASE_URL}/query",
        json={"text": query},
        headers=headers,
        timeout=30,
    )
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
    return r.json().get("intent", "")


EXAMPLE_QUERIES = [
    {"query": "Покрути спутник",                              "note": "action — команда вращения"},
    {"query": "Расскажи про Спутник-1",                       "note": "info — информационный запрос"},
    {"query": "Запусти анимацию и расскажи о спутнике",       "note": "hybrid — команда + вопрос"},
    {"query": "Сделай что-нибудь со спутником",               "note": "unknown_command"},
    {"query": "Как дела?",                                    "note": "off_topic"},
    {"query": "Кто вы?",                                      "note": "info — мета-запрос о системе"},
    {"query": "Расскажи о вас и о Метеор-М",                  "note": "info — мета + знания"},
    {"query": "Останови вращение и объясни что такое антенна", "note": "hybrid — стоп + вопрос"},
]


def test_examples_report(headers):
    """Генерирует таблицу примеров для раздела 4.2.1 диплома."""
    import requests

    requests.post(f"{BASE_URL}/debug/cache/clear", headers=headers)

    print("\n\n=== ПРИМЕРЫ ОБРАБОТКИ ЗАПРОСОВ (раздел 4.2.1) ===\n")

    rows = []
    for case in EXAMPLE_QUERIES:
        r = requests.post(
            f"{BASE_URL}/query",
            json={"text": case["query"]},
            headers=headers,
            timeout=30,
        )
        data = r.json()
        intent = data.get("intent", "—")
        answer = data.get("answer", "—")
        command = data.get("client_command") or {}
        cmd_type = command.get("type", "—") if command else "—"
        meta = data.get("meta_request", "")
        rows.append({
            "query": case["query"],
            "note": case["note"],
            "intent": intent,
            "command": cmd_type,
            "answer": answer,
            "meta": meta,
        })

    # Вывод таблицы
    print(f"{'Запрос':<45} {'Интент':<16} {'Команда':<18} {'Ответ'}")
    print("─" * 120)
    for row in rows:
        cmd = row["command"] if row["command"] != "—" else ""
        answer_short = row["answer"][:60] + ("..." if len(row["answer"]) > 60 else "")
        print(f"{row['query']:<45} {row['intent']:<16} {cmd:<18} {answer_short}")

    # Вывод полных ответов
    print("\n\n=== ПОЛНЫЕ ОТВЕТЫ ===\n")
    for row in rows:
        print(f"Запрос:  «{row['query']}»  ({row['note']})")
        print(f"Интент:  {row['intent']}" + (f" → {row['command']}" if row['command'] != "—" else ""))
        if row["meta"]:
            print(f"Мета:    {row['meta']}")
        print(f"Ответ:   {row['answer']}")
        print()


@pytest.mark.parametrize("case", INTENT_TEST_CASES, ids=[c["query"][:40] for c in INTENT_TEST_CASES])
def test_intent_classification(case, headers):
    result = classify(case["query"], headers)
    assert result == case["expected"], (
        f"Запрос: «{case['query']}»\n"
        f"Ожидалось: {case['expected']}\n"
        f"Получено:  {result}"
    )


def test_intent_report(headers):
    """Генерирует отчёт с метриками для таблицы главы 4."""
    classes = ["action", "info", "hybrid", "unknown_command", "off_topic"]
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    total_correct = 0

    results = []
    for case in INTENT_TEST_CASES:
        got = classify(case["query"], headers)
        expected = case["expected"]
        correct = got == expected
        if correct:
            tp[expected] += 1
            total_correct += 1
        else:
            fp[got] += 1
            fn[expected] += 1
        results.append({
            "query": case["query"],
            "expected": expected,
            "got": got,
            "ok": "✅" if correct else "❌",
        })

    print("\n\n=== ОТЧЁТ: Классификация интентов ===\n")
    print(f"{'Запрос':<45} {'Ожидаемый':<15} {'Получено':<15} {'Итог'}")
    print("-" * 85)
    for r in results:
        print(f"{r['query']:<45} {r['expected']:<15} {r['got']:<15} {r['ok']}")

    print("\n--- Метрики по классам ---")
    print(f"{'Интент':<20} {'TP':>5} {'FP':>5} {'FN':>5} {'Precision':>12} {'Recall':>10}")
    print("-" * 60)
    for cls in classes:
        t = tp[cls]
        f_p = fp[cls]
        f_n = fn[cls]
        precision = t / (t + f_p) if (t + f_p) > 0 else 0.0
        recall = t / (t + f_n) if (t + f_n) > 0 else 0.0
        print(f"{cls:<20} {t:>5} {f_p:>5} {f_n:>5} {precision:>11.2f} {recall:>10.2f}")

    accuracy = total_correct / len(INTENT_TEST_CASES)
    print(f"\nОбщая точность (Accuracy): {accuracy:.2f} ({total_correct}/{len(INTENT_TEST_CASES)})")
