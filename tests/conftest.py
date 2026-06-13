import pytest
import requests

BASE_URL = "http://localhost:8000"
AUTH_EMAIL = "test_chapter4@cosmovision.ru"
AUTH_PASSWORD = "TestPassword123"


def get_token() -> str:
    try:
        r = requests.post(f"{BASE_URL}/auth/login", json={"email": AUTH_EMAIL, "password": AUTH_PASSWORD})
        if r.status_code == 401:
            r = requests.post(f"{BASE_URL}/auth/register", json={"email": AUTH_EMAIL, "password": AUTH_PASSWORD})
        return r.json()["access_token"]
    except Exception:
        return ""


@pytest.fixture(scope="session")
def token():
    return get_token()


@pytest.fixture(scope="session")
def headers(token):
    return {"Authorization": f"Bearer {token}"}


# ── Тестовые данные: классификация интентов ──────────────────────────────────

INTENT_TEST_CASES = [
    # action
    {"query": "Покрути спутник",                         "expected": "action"},
    {"query": "Можешь повернуть его?",                   "expected": "action"},
    {"query": "Останови вращение",                       "expected": "action"},
    {"query": "Стоп",                                    "expected": "action"},
    {"query": "Увеличь модель",                          "expected": "action"},
    {"query": "Сделай поменьше",                         "expected": "action"},
    {"query": "Верни как было",                          "expected": "action"},
    {"query": "Запусти анимацию",                        "expected": "action"},
    # info
    {"query": "Что такое Спутник-1?",                    "expected": "info"},
    {"query": "Расскажи про Метеор-М",                   "expected": "info"},
    {"query": "Как устроена антенна?",                   "expected": "info"},
    {"query": "Для чего нужны солнечные панели?",        "expected": "info"},
    {"query": "Какова масса Спутника-1?",                "expected": "info"},
    {"query": "Объясни что такое орбита",                "expected": "info"},
    # hybrid
    {"query": "Покрути и расскажи о спутнике",           "expected": "hybrid"},
    {"query": "Запусти анимацию и объясни что это за антенна", "expected": "hybrid"},
    {"query": "Увеличь модель и расскажи о Метеор-М",   "expected": "hybrid"},
    {"query": "Останови вращение и объясни назначение спутника", "expected": "hybrid"},
    # unknown_command
    {"query": "Сделай что-нибудь со спутником",          "expected": "unknown_command"},
    {"query": "Измени цвет модели",                      "expected": "unknown_command"},
    # off_topic
    {"query": "Как дела?",                               "expected": "off_topic"},
    {"query": "Что такое Python?",                       "expected": "off_topic"},
    {"query": "Расскажи анекдот",                        "expected": "off_topic"},
]

# ── Тестовые данные: семантический поиск ─────────────────────────────────────

RETRIEVAL_TEST_CASES = [
    {"query": "Когда был запущен Спутник-1?",            "expected_keyword": "1957"},
    {"query": "Масса Спутника-1",                        "expected_keyword": "83"},
    {"query": "Диаметр корпуса Спутника-1",              "expected_keyword": "58"},
    {"query": "Антенны Спутника-1",                      "expected_keyword": "антенн"},
    {"query": "Назначение Спутника-1",                   "expected_keyword": "спутник"},
    {"query": "Что такое Метеор-М?",                     "expected_keyword": "метеор"},
    {"query": "Назначение Метеор-М",                     "expected_keyword": "гидрометеор"},
    {"query": "Орбита Метеор-М",                         "expected_keyword": "орбит"},
    {"query": "Солнечные панели спутника",               "expected_keyword": "солнечн"},
    {"query": "Система связи спутника",                  "expected_keyword": "связ"},
]
