"""
Запуск всех тестов и генерация полного отчёта для главы 4.
Использование: python tests/run_report.py
"""
import subprocess
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run():
    print("=" * 60)
    print("Запуск тестов для главы 4 — Тестирование системы")
    print("=" * 60)
    print("\nУбедись что сервер запущен: uvicorn app.api:app\n")

    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_intent.py::test_examples_report",
        "tests/test_intent.py::test_intent_report",
        "tests/test_retrieval.py::test_retrieval_report",
        "tests/test_latency.py::test_latency_report",
        "tests/test_rag_comparison.py::test_comparison_report",
        "tests/test_asr.py::test_asr_report",
        "tests/test_performance.py::test_concurrent_report",
        "tests/test_performance.py::test_chromadb_report",
        "-v", "-s", "--tb=short",
    ]

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    sys.exit(result.returncode)


if __name__ == "__main__":
    run()
