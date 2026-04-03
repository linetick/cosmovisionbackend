import os

import chromadb
import torch
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from kb_aliases import build_auto_alias_map

from .config import (
    DB_PATH,
    HF_TOKEN,
    LLM_ATTN_IMPLEMENTATION,
    LLM_BACKEND,
    LLM_FORCE_SINGLE_GPU,
    LLM_USE_4BIT,
    MODEL_ID,
    QUERY_STOPWORDS,
    RAW_KB_DIR,
    VLLM_BASE_URL,
    VLLM_MODEL,
    WHISPER_PRELOAD,
    WHISPER_MODEL_ID,
)

print("SERVER DB_PATH =", DB_PATH)
print("SERVER CWD =", os.getcwd())

BACKEND_GPU_ENABLED = torch.cuda.is_available() and LLM_BACKEND != "vllm"
device = "cuda" if BACKEND_GPU_ENABLED else "cpu"
embedder_device = os.getenv("EMBEDDER_DEVICE", "cpu").strip().lower() or "cpu"
if embedder_device == "cuda" and not BACKEND_GPU_ENABLED:
    embedder_device = "cpu"

default_whisper_device = "cpu" if LLM_BACKEND == "vllm" else ("cuda" if torch.cuda.is_available() else "cpu")
WHISPER_DEVICE = (os.getenv("WHISPER_DEVICE") or default_whisper_device).strip().lower()
if WHISPER_DEVICE == "cuda" and not torch.cuda.is_available():
    WHISPER_DEVICE = "cpu"
DEFAULT_WHISPER_COMPUTE = "float16" if WHISPER_DEVICE == "cuda" else "int8"
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", DEFAULT_WHISPER_COMPUTE).strip() or DEFAULT_WHISPER_COMPUTE

print(f"Используемое устройство: {device}")
print(f"Устройство retrieval-эмбеддера: {embedder_device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
elif LLM_BACKEND == "vllm" and torch.cuda.is_available():
    print("GPU зарезервирована под vLLM; backend работает на CPU.")
print(f"Whisper будет загружен по требованию на: {WHISPER_DEVICE} ({WHISPER_COMPUTE_TYPE})")

print("Загрузка модели эмбеддингов...")
embedder = SentenceTransformer("intfloat/multilingual-e5-small", device=embedder_device)

tokenizer = None
model = None

if LLM_BACKEND == "vllm":
    print(f"LLM backend: vLLM ({VLLM_BASE_URL}, model={VLLM_MODEL})")
else:
    print(f"Загрузка LLM ({MODEL_ID})...")
    if not HF_TOKEN and MODEL_ID.startswith("meta-llama/"):
        print("HF_TOKEN не задан. Загрузка с Hugging Face сработает только при локальном кеше или активном huggingface-cli login.")

    tokenizer_kwargs = {}
    if HF_TOKEN:
        tokenizer_kwargs["token"] = HF_TOKEN
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if HF_TOKEN:
        model_kwargs["token"] = HF_TOKEN

    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = {"": 0} if LLM_FORCE_SINGLE_GPU else "auto"
        if LLM_ATTN_IMPLEMENTATION:
            model_kwargs["attn_implementation"] = LLM_ATTN_IMPLEMENTATION
    else:
        model_kwargs["torch_dtype"] = torch.float32
        print("CUDA недоступна, 4-bit квантование отключено.")

    if device == "cuda" and LLM_USE_4BIT:
        quantized_model_kwargs = dict(model_kwargs)
        quantized_model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        try:
            print("LLM загружается в 4-bit (NF4).")
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **quantized_model_kwargs)
        except Exception as exc:
            print(f"⚠️ Не удалось загрузить LLM в 4-bit: {exc}")
            print("⚠️ Переключаемся на обычную загрузку модели без квантования. Если это Colab, причина обычно в несовместимости bitsandbytes/triton.")
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    else:
        if device == "cuda":
            print("LLM загружается без 4-bit квантования.")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)

    if device != "cuda":
        model = model.to("cpu")
    model.eval()

whisper_model = None


def get_whisper_model() -> WhisperModel:
    global whisper_model
    if whisper_model is None:
        print(f"Загрузка Whisper ({WHISPER_MODEL_ID}) на {WHISPER_DEVICE}...")
        whisper_model = WhisperModel(
            WHISPER_MODEL_ID,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
    return whisper_model


def preload_whisper() -> None:
    if not WHISPER_PRELOAD:
        print("Предзагрузка Whisper отключена.")
        return

    try:
        get_whisper_model()
        print("✅ Whisper предзагружен.")
    except Exception as exc:
        print(f"⚠️ Предзагрузка Whisper завершилась с предупреждением: {exc}")


print("Подключение к базе знаний...")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection("satellites")
COLLECTION_COUNT = collection.count()
print(f"Фрагментов в базе знаний: {COLLECTION_COUNT}")
AUTO_QUERY_ALIASES = build_auto_alias_map(RAW_KB_DIR, QUERY_STOPWORDS)
print(f"Автоматических алиасов из БЗ: {len(AUTO_QUERY_ALIASES)}")


def warmup(retrieve_probe, generation_probe) -> None:
    preload_whisper()

    print("Прогрев retrieval...")
    try:
        _ = embedder.encode(
            ["query: что такое спутник"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        print("✅ Retrieval прогрет!")
    except Exception as exc:
        print(f"⚠️ Прогрев retrieval завершён с предупреждением: {exc}")

    print("Прогрев модели...")
    try:
        generation_probe("Что такое спутник?", "Спутник — это аппарат, обращающийся вокруг Земли.")
        print("✅ Сервер готов!")
    except Exception as exc:
        print(f"⚠️ Прогрев завершён с предупреждением: {exc}")
