"""
4.3 Тестирование семантического поиска
Запуск: pytest tests/test_retrieval.py -v
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from app.query.retrieval import retrieve_hits, build_context_from_hits
from tests.conftest import RETRIEVAL_TEST_CASES


def get_hits(query: str, n: int = 3):
    hits, _ = retrieve_hits(query, initial_n=n, max_n=n)
    return hits


def hit_at_k(hits: list, keyword: str, k: int) -> bool:
    for hit in hits[:k]:
        doc = (hit.get("doc") or "").lower()
        if keyword.lower() in doc:
            return True
    return False


def reciprocal_rank(hits: list, keyword: str) -> float:
    for i, hit in enumerate(hits, start=1):
        doc = (hit.get("doc") or "").lower()
        if keyword.lower() in doc:
            return 1.0 / i
    return 0.0


@pytest.mark.parametrize("case", RETRIEVAL_TEST_CASES, ids=[c["query"][:40] for c in RETRIEVAL_TEST_CASES])
def test_retrieval_hit_at_3(case):
    hits = get_hits(case["query"], n=3)
    assert hit_at_k(hits, case["expected_keyword"], k=3), (
        f"Запрос: «{case['query']}»\n"
        f"Ключевое слово «{case['expected_keyword']}» не найдено в top-3\n"
        f"Найдено: {[h['doc'][:60] for h in hits]}"
    )


def test_retrieval_report():
    """Генерирует отчёт с метриками Hit@1, Hit@3, MRR для таблицы главы 4."""
    h1_scores = []
    h3_scores = []
    mrr_scores = []

    print("\n\n=== ОТЧЁТ: Семантический поиск ===\n")
    print(f"{'Запрос':<45} {'Hit@1':>7} {'Hit@3':>7} {'RR':>6}")
    print("-" * 70)

    for case in RETRIEVAL_TEST_CASES:
        hits = get_hits(case["query"], n=3)
        kw = case["expected_keyword"]
        h1 = int(hit_at_k(hits, kw, k=1))
        h3 = int(hit_at_k(hits, kw, k=3))
        rr = reciprocal_rank(hits, kw)
        h1_scores.append(h1)
        h3_scores.append(h3)
        mrr_scores.append(rr)
        print(f"{case['query']:<45} {h1:>7} {h3:>7} {rr:>6.2f}")

    print("-" * 70)
    print(f"{'ИТОГО':<45} {sum(h1_scores)/len(h1_scores):>7.2f} {sum(h3_scores)/len(h3_scores):>7.2f} {sum(mrr_scores)/len(mrr_scores):>6.2f}")
    print(f"\nHit@1: {sum(h1_scores)/len(h1_scores):.2f}")
    print(f"Hit@3: {sum(h3_scores)/len(h3_scores):.2f}")
    print(f"MRR:   {sum(mrr_scores)/len(mrr_scores):.2f}")
