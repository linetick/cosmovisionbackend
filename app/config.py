import os

from dotenv import load_dotenv


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_DIR = os.path.dirname(BASE_DIR)

for env_path in (
    os.path.join(WORKSPACE_DIR, ".env"),
    os.path.join(BASE_DIR, ".env"),
):
    if os.path.exists(env_path):
        load_dotenv(env_path, override=False)

DB_PATH = os.path.join(BASE_DIR, "knowledge_db")
RAW_KB_DIR = os.path.join(BASE_DIR, "knowledge_raw")
MODEL_STORAGE_DIR = os.path.join(BASE_DIR, "model_storage")
MODEL_FILES_DIR = os.path.join(MODEL_STORAGE_DIR, "files")
MODEL_REGISTRY_PATH = os.path.join(MODEL_STORAGE_DIR, "registry.json")

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

APP_TITLE = "CosmoVision AI Backend"
APP_VERSION = "2.1"
REFUSAL = "В предоставленных данных нет информации."
AR_COMPACT_RESPONSES = os.getenv("AR_COMPACT_RESPONSES", "1").strip().lower() not in {"0", "false", "no"}

MODEL_ID = os.getenv("LLM_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct").strip()
LLM_BACKEND = os.getenv("LLM_BACKEND", "local").strip().lower()
HF_TOKEN = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or "").strip() or None
LLM_USE_4BIT = os.getenv("LLM_USE_4BIT", "1").strip().lower() not in {"0", "false", "no"}
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "64"))
LLM_MAX_INPUT_TOKENS = int(os.getenv("LLM_MAX_INPUT_TOKENS", "2048"))
RAG_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "1200"))
SHORT_LLM_MAX_NEW_TOKENS = int(os.getenv("SHORT_LLM_MAX_NEW_TOKENS", "24"))
SHORT_RAG_MAX_CONTEXT_CHARS = int(os.getenv("SHORT_RAG_MAX_CONTEXT_CHARS", "500"))
RELEVANCE_GATE_ENABLED = os.getenv("RELEVANCE_GATE_ENABLED", "0").strip().lower() not in {"0", "false", "no"}
LLM_FORCE_SINGLE_GPU = os.getenv("LLM_FORCE_SINGLE_GPU", "1").strip().lower() not in {"0", "false", "no"}
LLM_ATTN_IMPLEMENTATION = os.getenv("LLM_ATTN_IMPLEMENTATION", "sdpa").strip()

WHISPER_MODEL_ID = os.getenv("WHISPER_MODEL_ID", "small").strip() or "small"
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
WHISPER_VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "0").strip().lower() not in {"0", "false", "no"}
WHISPER_PRELOAD = os.getenv("WHISPER_PRELOAD", "1").strip().lower() not in {"0", "false", "no"}
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1").strip().rstrip("/")
VLLM_API_KEY = (os.getenv("VLLM_API_KEY") or "token-abc123").strip()
VLLM_MODEL = os.getenv("VLLM_MODEL", MODEL_ID).strip() or MODEL_ID
VLLM_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "120"))
YANDEX_BASE_URL = os.getenv("YANDEX_BASE_URL", "https://ai.api.cloud.yandex.net/v1").strip().rstrip("/")
YANDEX_API_KEY = (os.getenv("YANDEX_API_KEY") or "").strip()
YANDEX_PROJECT_ID = (os.getenv("YANDEX_PROJECT_ID") or "").strip()
YANDEX_MODEL = os.getenv("YANDEX_MODEL", "qwen3.6-35b-a3b/latest").strip() or "qwen3.6-35b-a3b/latest"
YANDEX_TIMEOUT = float(os.getenv("YANDEX_TIMEOUT", "120"))
YANDEX_PROMPT_ID = (os.getenv("YANDEX_PROMPT_ID") or "").strip() or None
YANDEX_ROUTER_PROMPT_ID = (os.getenv("YANDEX_ROUTER_PROMPT_ID") or "").strip() or None
YANDEX_LOG_USAGE = os.getenv("YANDEX_LOG_USAGE", "1").strip().lower() not in {"0", "false", "no"}

QUERY_STOPWORDS = {
    "а", "без", "был", "была", "были", "было", "быть", "в", "во", "вопрос", "все",
    "где", "для", "до", "его", "ее", "если", "есть", "еще", "же", "за", "и", "из",
    "или", "как", "какая", "какие", "какой", "каком", "какому", "какую", "когда",
    "кто", "ли", "м", "меня", "мне", "на", "над", "надо", "не", "нет", "но", "о",
    "об", "обо", "он", "она", "они", "оно", "от", "по", "под", "при", "про", "с",
    "со", "так", "такое", "такой", "там", "то", "только", "у", "что", "это", "этот",
    "эта", "эти", "энергия"
}

WEAK_QUERY_TOKENS = {
    "аппарат", "баз", "дан", "знан", "информац", "космонавтик", "космос", "метеор",
    "модел", "ответ", "спутник", "систем"
}

REFUSAL_PATTERNS = [
    "нет информации", "информация отсутствует", "не найден", "не найдена",
    "не найдено", "не указ", "не опис", "не содерж", "не сказан", "не сообщ",
    "не упомина", "не удалось найти", "нет данных", "отсутствуют данные",
]

ANSWER_MARKERS = (
    "— это", "это ", "предназнач", "используется", "служит для",
    "представляет собой", "позволяет", "состоит", "имеет",
)

SUPPORTED_MODEL_EXTENSIONS = {".fbx", ".glb", ".gltf", ".obj", ".usdz"}
