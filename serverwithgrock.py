# server.py
from fastapi import FastAPI, HTTPException, UploadFile, File
import tempfile, subprocess
from faster_whisper import WhisperModel
from pydantic import BaseModel
import os
import re
import chromadb
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from pyngrok import conf, ngrok

import nest_asyncio
import threading
import uvicorn
import sys
import time

#Токен
NGROK_TOKEN = "39jhsX0Tw6Kp5vbEwB8xqZeaHbs_81R3cLX6c7bpau5HLpypn"
conf.get_default().auth_token = NGROK_TOKEN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "knowledge_db")
print("SERVER DB_PATH =", DB_PATH)
print("SERVER CWD =", os.getcwd())


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

REFUSAL = "В предоставленных данных нет информации."

# === Device ===
#device = "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используемое устройство: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# === Models ===
print("Загрузка модели эмбеддингов...")
#embedder = SentenceTransformer("intfloat/multilingual-e5-small", device="cuda")
embedder = SentenceTransformer("intfloat/multilingual-e5-small", device=device)

print("Загрузка LLM (Phi-3-mini)...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

print("Загрузка Whisper (faster-whisper)...")
whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_compute = "float16" if whisper_device == "cuda" else "int8"
whisper_model = WhisperModel("small", device=whisper_device, compute_type=whisper_compute)

if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        #device_map="auto",
        device_map={"":0},
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to("cpu")

model.eval()

# === Chroma ===
print("Подключение к базе знаний...")
#client = chromadb.PersistentClient(path="knowledge_db")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection("satellites")

# === API ===
app = FastAPI(title="CosmoVision AI Backend", version="2.1")


class QueryRequest(BaseModel):
    text: str


# ---------- Utils ----------
def normalize_query(q: str) -> str:
    """Мини-нормализация: прибираем мусор и частые опечатки."""
    q = (q or "").strip()
    q = re.sub(r"\s+", " ", q)
    # твой кейс: Метреор -> Метеор
    q = re.sub(r"метреор", "метеор", q, flags=re.IGNORECASE)
    return q


def is_off_topic(query: str) -> bool:
    q = query.lower()

    personal = [
        "ты кто", "кто ты", "как тебя зовут",
        "привет", "здравствуй", "как дела",
        "что ты", "ты робот", "ассистент", "помощник"
    ]
    if any(p in q for p in personal):
        return True

    on_topic = [
        "спутник", "космос", "космонавтика", "орбита", "антенна",
        "панел", "метеор", "глонасс", "аппарат", "ракета", "запуск",
        "устройство", "работает", "назначение", "система", "модуль",
        "двигатель", "солнечн", "питание", "передач", "связь",
        "датчик", "камера", "телеметр",
    ]
    return not any(k in q for k in on_topic)


def clean_doc_keep_header(text: str) -> str:
    """
    ВАЖНО: НЕ убиваем заголовки.
    - убираем "passage:"
    - '# Заголовок' превращаем в 'Заголовок.'
    - выкидываем пустые строки
    """
    text = (text or "").replace("passage: ", "").strip()
    if not text:
        return ""

    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            # превращаем '# Метеор-М' -> 'Метеор-М.'
            hdr = s.lstrip("#").strip()
            if hdr:
                if not hdr.endswith("."):
                    hdr += "."
                out.append(hdr)
            continue
        out.append(s)

    return "\n".join(out).strip()


def retrieve_context(query: str, initial_n: int = 3, max_n: int = 8) -> tuple[str, list[float] | None]:
    """
    Достаём контекст так, чтобы он был НЕ пустой и НЕ огрызок.
    Если первые документы пустые/короткие после чистки — расширяем n_results.
    """
    q_emb = embedder.encode([f"query: {query}"], normalize_embeddings=True)

    n = initial_n
    best_distances = None
    best_context = ""

    while n <= max_n:
        try:
            results = collection.query(
                query_embeddings=q_emb,
                n_results=n,
                include=["documents", "distances"],
            )
            distances = results.get("distances", None)
        except TypeError:
            results = collection.query(query_embeddings=q_emb, n_results=n)
            distances = None

        docs = (results.get("documents") or [[]])[0]
        cleaned = []
        for d in docs:
            cd = clean_doc_keep_header(d)
            if cd:
                cleaned.append(cd)

        context = "\n\n".join(cleaned).strip()

        dist_list = None
        if isinstance(distances, list) and distances and isinstance(distances[0], list):
            dist_list = distances[0]

        if context and len(context) > len(best_context):
            best_context = context
            best_distances = dist_list

        if len(context) >= 80:
            return context, dist_list

        n += 2

    return best_context, best_distances


def looks_answerable(query: str, context: str) -> bool:
    if not context.strip():
        return False

    q = query.lower()
    c = context.lower()

    definitional = any(x in q for x in ["что такое", "что значит", "определи", "дать определение"])

    if definitional:
        markers = ["— это", "это ", "предназнач", "используется", "служит для", "представляет собой"]
        has_marker = any(m in c for m in markers)

        if "метеор" in q:
            return has_marker and ("метеор" in c)
        return has_marker

    # не определение: хотя бы какой-то объём
    return len(context) >= 40


def extract_definition_from_context(query: str, context: str) -> str | None:
    q = query.lower()
    c = context

    # если вопрос про Метеор — ищем строку "Метеор-М — это ..."
    if "метеор" in q:
        m = re.search(r"(Метеор[-–— ]?М\s*—\s*это[^\n\.]*[\.]?)", c, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # общий случай: первая строка с "— это"
    m = re.search(r"^(.{0,80}?—\s*это[^\n\.]*[\.]?)", c, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()

    return None

def is_definitional(q: str) -> bool:
    ql = q.lower()
    return any(x in ql for x in ["что такое", "что значит", "определи", "дать определение"])


def generate_answer_strict(query: str, context: str) -> str:
    # 1) если можем ответить без LLM — отвечаем без LLM (0 галлюцинаций)
    if is_definitional(query):
        defin = extract_definition_from_context(query, context)
        if defin:
            return defin
        return REFUSAL

    # 2) для не-определений — LLM, но БЕЗ FACT/ANSWER формата
    if not context.strip():
        return REFUSAL

    system = (
        "Ты — ИИ-ассистент по космонавтике.\n"
        "Отвечай только по контексту. Нельзя добавлять факты не из контекста.\n"
        f"Если в контексте нет ответа, верни ровно: {REFUSAL}"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"КОНТЕКСТ:\n{context}\n\nВОПРОС: {query}"},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3800).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            #max_new_tokens=120,
            max_new_tokens=60,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs.input_ids.shape[1]
    text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    # если модель несёт что-то совсем левое — хотя бы отрежем и подстрахуем
    if not text or len(text) < 3:
        return REFUSAL
    return text

import tempfile, subprocess, os

def wav_to_16k_mono_wav_bytes(wav_bytes: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fin:
        fin.write(wav_bytes)
        fin.flush()
        in_path = fin.name

    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(out_fd)

    try:
        # -ac 1: mono, -ar 16000: 16kHz, -f wav: wav container
        subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", "16000", "-f", "wav", out_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in (in_path, out_path):
            try: os.remove(p)
            except: pass

def transcribe_audio_bytes(data: bytes, language: str = "ru") -> str:
    data = wav_to_16k_mono_wav_bytes(data)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(data); tmp.flush()
        segments, info = whisper_model.transcribe(
            tmp.name,
            language=language,
            vad_filter=True,
            beam_size=5,
        )
        return " ".join(seg.text for seg in segments).strip()


# ---------- Endpoint ----------
@app.post("/query")
def handle_query(req: QueryRequest):
    try:
        raw = req.text or ""
        q = normalize_query(raw)

        if not q:
            return {"query": raw, "answer": REFUSAL, "context_used": False}

        if is_off_topic(q):
            return {
                "query": q,
                "answer": (
                    "Я отвечаю только по космонавтике из базы знаний. "
                    "Спроси, например: «Как устроены солнечные панели на Метеоре-М?»"
                ),
                "context_used": False,
            }

        context, distances = retrieve_context(q, initial_n=3, max_n=9)

        # Мягкий отсев по distance (если есть). Порог пусть будет консервативный.
        if distances is not None and len(distances) > 0:
            best = distances[0]
            if best is not None and best > 0.9:
                return {"query": q, "answer": REFUSAL, "context_used": False}

        answer = generate_answer_strict(q, context)

        return {
            "query": q,
            "answer": answer,
            "context_used": (answer != REFUSAL),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

@app.post("/query_audio")
async def handle_query_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            return {"answer": REFUSAL, "context_used": False, "transcript": ""}

        t0 = time.time()
        transcript = transcribe_audio_bytes(audio_bytes, language="ru")
        t1 = time.time()

        q = normalize_query(transcript)

        if not q:
            return {"answer": REFUSAL, "context_used": False, "transcript": transcript}

        if is_off_topic(q):
            return {
                "transcript": transcript,
                "query": q,
                "answer": (
                    "Я отвечаю только по космонавтике из базы знаний. "
                    "Спроси, например: «Как устроены солнечные панели на Метеоре-М?»"
                ),
                "context_used": False,
                "timing": {"asr": round(t1 - t0, 3)}
            }

        t2 = time.time()
        context, distances = retrieve_context(q, initial_n=3, max_n=9)
        t3 = time.time()

        if distances is not None and len(distances) > 0:
            best = distances[0]
            if best is not None and best > 0.9:
                return {
                    "transcript": transcript,
                    "query": q,
                    "answer": REFUSAL,
                    "context_used": False,
                    "timing": {"asr": round(t1 - t0, 3), "retrieve": round(t3 - t2, 3)}
                }

        answer = generate_answer_strict(q, context)
        t4 = time.time()

        return {
            "transcript": transcript,
            "query": q,
            "answer": answer,
            "context_used": (answer != REFUSAL),
            "timing": {
                "asr": round(t1 - t0, 3),
                "retrieve": round(t3 - t2, 3),
                "generate": round(t4 - t3, 3),
                "total": round(t4 - t0, 3),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки аудио: {str(e)}")



@app.get("/debug/kb")
def debug_kb():
    return {
        "db_path": DB_PATH,
        "cwd": os.getcwd(),
        "count": collection.count(),
    }

@app.post("/debug/search")
def debug_search(req: QueryRequest):
    q = normalize_query(req.text)

    q_emb = embedder.encode([f"query: {q}"], normalize_embeddings=True)

    try:
        res = collection.query(
            query_embeddings=q_emb,
            n_results=5,
            include=["documents", "distances", "metadatas"],
        )
    except TypeError:
        res = collection.query(query_embeddings=q_emb, n_results=5)

    docs = (res.get("documents") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    n = max(len(docs), len(dists), len(metas))

    top = []
    for i in range(n):
        top.append({
            "i": i,
            "distance": dists[i] if i < len(dists) else None,
            "meta": metas[i] if i < len(metas) else None,
            "doc": docs[i] if i < len(docs) else None,
        })

    return {
        "query": q,
        "server_cwd": os.getcwd(),
        "db_path": DB_PATH if "DB_PATH" in globals() else "unknown",
        "collection_count": collection.count(),
        "lens": {"docs": len(docs), "dists": len(dists), "metas": len(metas)},
        "top": top,
    }


# === Warmup ===
print("Прогрев модели...")
try:
    test_context = "Спутник — это аппарат, обращающийся вокруг Земли."
    _ = generate_answer_strict("Что такое спутник?", test_context)
    print("✅ Сервер готов!")
except Exception as e:
    print(f"⚠️ Прогрев завершён с предупреждением: {e}")



nest_asyncio.apply()
sys.path.insert(0, os.getcwd())

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

thread = threading.Thread(target=run_server, daemon=True)
thread.start()

time.sleep(3)
print("\n🌐 Проброс порта через ngrok...")
try:
    public_url = ngrok.connect(8000)
    NGROK_URL = str(public_url)
    
    print("\n" + "="*60)
    print("🚀 СЕРВЕР РАБОТАЕТ!")
    print("="*60)
    print(f"🔗 Swagger UI: {NGROK_URL}/docs")
    print(f"🔗 Debug: {NGROK_URL}/debug/kb")
    print("="*60)
    
    print("Ожидание подключений (не закрывайте эту ячейку)...\n")
    while True:
        time.sleep(1)
        
except Exception as e:
    print(f"Ошибка ngrok: {e}")
    while True:
        time.sleep(1)