"""
4.6 Сравнительный анализ: RAG с контекстом vs без контекста (fallback)
Запуск: pytest tests/test_rag_comparison.py -v -s
"""
import pytest
import requests

from tests.conftest import BASE_URL

# Вопросы, на которые в базе знаний есть ответ
QUERIES_IN_KB = [
    "Когда был запущен Спутник-1?",
    "Какова масса Спутника-1?",
    "Для чего предназначен Метеор-М?",
    "Как работают солнечные панели Метеор-М?",
    "Расскажи об орбите Метеор-М",
]

# Вопросы, которых в базе знаний нет (должен сработать fallback)
QUERIES_NOT_IN_KB = [
    "Расскажи о спутнике МКС",
    "Что такое телескоп Хаббл?",
    "Какой спутник был запущен первым в США?",
]


def query(text: str, headers: dict) -> dict:
    r = requests.post(
        f"{BASE_URL}/query",
        json={"text": text},
        headers=headers,
        timeout=30,
    )
    assert r.status_code == 200
    return r.json()


def test_known_queries_use_context(headers):
    """Вопросы из базы знаний должны возвращать context_used=True."""
    requests.post(f"{BASE_URL}/debug/cache/clear", headers=headers)
    for text in QUERIES_IN_KB:
        result = query(text, headers)
        assert result.get("context_used") is True, (
            f"Запрос «{text}» должен использовать контекст БЗ, но context_used=False\n"
            f"Ответ: {result.get('answer', '')}"
        )


def test_unknown_queries_use_fallback(headers):
    """Вопросы не из БЗ должны возвращать fallback-ответ."""
    for text in QUERIES_NOT_IN_KB:
        result = query(text, headers)
        assert result.get("context_used") is False, (
            f"Запрос «{text}» не должен использовать контекст, но context_used=True"
        )


def test_comparison_report(headers):
    """Отчёт: RAG с контекстом vs fallback без контекста."""
    requests.post(f"{BASE_URL}/debug/cache/clear", headers=headers)

    print("\n\n=== ОТЧЁТ: Эффективность RAG ===\n")

    def print_result(text: str, result: dict, mark: str) -> None:
        answer = result.get("answer", "")
        print(f"\n  Запрос:  {text}")
        print(f"  Контекст: {mark}")
        print(f"  Ответ:   {answer}")

    print("--- Запросы ИЗ базы знаний ---")
    kb_ok = 0
    for text in QUERIES_IN_KB:
        result = query(text, headers)
        used = result.get("context_used", False)
        mark = "✅ использован" if used else "❌ не найден"
        if used:
            kb_ok += 1
        print_result(text, result, mark)

    print(f"\n--- Запросы НЕ из базы знаний (fallback) ---")
    for text in QUERIES_NOT_IN_KB:
        result = query(text, headers)
        used = result.get("context_used", False)
        mark = "✅ использован" if used else "— не найден (ожидаемо)"
        print_result(text, result, mark)

    # Статистика по запросам не из БЗ
    not_kb_ok = 0
    not_kb_total = len(QUERIES_NOT_IN_KB)
    for text in QUERIES_NOT_IN_KB:
        result = query(text, headers)
        if not result.get("context_used", False):
            not_kb_ok += 1

    total_kb = len(QUERIES_IN_KB)

    print(f"\n{'=' * 70}")
    print(f"{'СВОДНАЯ ТАБЛИЦА ДЛЯ ДИПЛОМА':^70}")
    print(f"{'=' * 70}")
    print(f"\n{'Набор запросов':<30} {'Всего':>7} {'С контекстом':>14} {'Без контекста':>14}")
    print("-" * 70)
    print(f"{'Вопросы из базы знаний':<30} {total_kb:>7} {kb_ok:>14} {total_kb - kb_ok:>14}")
    print(f"{'Вопросы вне базы знаний':<30} {not_kb_total:>7} {not_kb_total - not_kb_ok:>14} {not_kb_ok:>14}")
    print("-" * 70)
    total = total_kb + not_kb_total
    total_with = kb_ok + (not_kb_total - not_kb_ok)
    total_without = (total_kb - kb_ok) + not_kb_ok
    print(f"{'ИТОГО':<30} {total:>7} {total_with:>14} {total_without:>14}")
    print(f"\nВывод: RAG нашёл контекст для {kb_ok}/{total_kb} профильных запросов "
          f"({kb_ok/total_kb*100:.0f}%).")
