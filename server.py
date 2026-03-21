# server.py
from fastapi import FastAPI, HTTPException, UploadFile, File
import tempfile
from faster_whisper import WhisperModel
from pydantic import BaseModel
import os
import re
import chromadb
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "knowledge_db")
print("SERVER DB_PATH =", DB_PATH)
print("SERVER CWD =", os.getcwd())


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

REFUSAL = "В предоставленных данных нет информации."
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

# === Device ===
device = "cpu"
#device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"Используемое устройство: {device}")
#if device == "cuda":
 #   print(f"GPU: {torch.cuda.get_device_name(0)}")

# === Models ===
print("Загрузка модели эмбеддингов...")
embedder = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")

print("Загрузка LLM (Phi-3-mini)...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        device_map="auto",
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

print("Загрузка Whisper (faster-whisper)...")
whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_compute = "float16" if whisper_device == "cuda" else "int8"
whisper_model = WhisperModel("small", device=whisper_device, compute_type=whisper_compute)

# === Chroma ===
print("Подключение к базе знаний...")
#client = chromadb.PersistentClient(path="knowledge_db")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection("satellites")

app = FastAPI(title="CosmoVision AI Backend", version="2.1")


class QueryRequest(BaseModel):
    text: str


# ---------- Utils ----------
def normalize_query(q: str) -> str:
    """Мини-нормализация: прибираем мусор и частые опечатки."""
    q = (q or "").strip()
    q = re.sub(r"\s+", " ", q)
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
    text = (text or "").replace("passage: ", "").strip()
    if not text:
        return ""

    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            hdr = s.lstrip("#").strip()
            if hdr:
                if not hdr.endswith("."):
                    hdr += "."
                out.append(hdr)
            continue
        out.append(s)

    return "\n".join(out).strip()


def normalize_match_token(token: str) -> str:
    token = token.lower().replace("ё", "е").strip("-")
    if not token:
        return ""

    suffixes = (
        "иями", "ями", "ами", "иях", "ев", "ов", "ие", "ые", "ое", "ее",
        "ий", "ый", "ой", "ая", "яя", "ам", "ям", "ах", "ях", "ом", "ем", "ую",
        "юю", "ого", "ему", "ому", "ыми", "ими", "ия", "ья", "ье", "иям", "ием",
        "ию", "ью", "а", "я", "ы", "и", "е", "у", "ю", "о"
    )
    for suffix in suffixes:
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            token = token[:-len(suffix)]
            break
    return token


def tokenize_for_match(text: str) -> list[str]:
    raw_tokens = re.findall(r"[0-9a-zA-Zа-яА-ЯёЁ-]+", (text or "").lower().replace("ё", "е"))
    tokens: list[str] = []
    for raw in raw_tokens:
        parts = [p for p in raw.split("-") if p]
        for part in parts:
            norm = normalize_match_token(part)
            if len(norm) >= 2:
                tokens.append(norm)
    return tokens


def query_token_sets(query: str) -> tuple[set[str], set[str]]:
    query_tokens = {
        tok for tok in tokenize_for_match(query)
        if tok not in QUERY_STOPWORDS
    }
    strong_tokens = {tok for tok in query_tokens if tok not in WEAK_QUERY_TOKENS}
    return query_tokens, strong_tokens


def split_doc_candidates(doc: str) -> list[str]:
    lines = [line.strip() for line in (doc or "").splitlines() if line.strip()]
    if not lines:
        return []

    candidates: list[str] = []
    if len(lines) >= 2 and len(lines[0]) <= 40 and not re.search(r"[.!?]$", lines[0]):
        candidates.append(f"{lines[0]}. {' '.join(lines[1:])}".strip())

    candidates.extend(lines)

    merged_text = " ".join(lines)
    candidates.extend(
        chunk.strip()
        for chunk in re.split(r"(?<=[.!?])\s+", merged_text)
        if chunk.strip()
    )
    candidates.append(merged_text)

    deduped: list[str] = []
    seen = set()
    for candidate in candidates:
        key = re.sub(r"\s+", " ", candidate).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def find_extractive_answer(query: str, hits: list[dict]) -> str | None:
    query_tokens, strong_tokens = query_token_sets(query)
    if not query_tokens:
        return None

    best_text = None
    best_score = -1
    best_strong_overlap = 0
    best_total_overlap = 0

    for hit in hits:
        doc = (hit.get("doc") or "").strip()
        meta = hit.get("meta") or {}
        section = meta.get("section") or ""
        section_tokens = set(tokenize_for_match(section))

        for candidate in split_doc_candidates(doc):
            candidate_tokens = set(tokenize_for_match(candidate))
            if not candidate_tokens:
                continue

            total_overlap = len(query_tokens & candidate_tokens)
            strong_overlap = len(strong_tokens & candidate_tokens)
            score = strong_overlap * 5 + total_overlap
            if strong_tokens and section_tokens:
                score += len(strong_tokens & section_tokens) * 3

            if score > best_score:
                best_text = candidate
                best_score = score
                best_strong_overlap = strong_overlap
                best_total_overlap = total_overlap

    if not best_text:
        return None
    if strong_tokens and best_strong_overlap == 0:
        return None
    if best_total_overlap == 0:
        return None

    return best_text


def is_refusal_like(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if not normalized:
        return True
    return any(pattern in normalized for pattern in REFUSAL_PATTERNS)


def retrieve_hits(query: str, initial_n: int = 3, max_n: int = 8) -> list[dict]:
    total_docs = collection.count()
    if total_docs <= 0:
        return []

    q_emb = embedder.encode([f"query: {query}"], normalize_embeddings=True)

    n = min(max(initial_n, 1), total_docs)
    limit = min(max_n, total_docs)
    best_hits: list[dict] = []
    best_chars = 0

    while True:
        try:
            results = collection.query(
                query_embeddings=q_emb,
                n_results=n,
                include=["documents", "distances", "metadatas"],
            )
        except TypeError:
            results = collection.query(query_embeddings=q_emb, n_results=n)

        docs = (results.get("documents") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]

        hits: list[dict] = []
        seen_docs = set()
        for i, d in enumerate(docs):
            cd = clean_doc_keep_header(d)
            if not cd or cd in seen_docs:
                continue
            seen_docs.add(cd)
            hits.append({
                "doc": cd,
                "distance": distances[i] if i < len(distances) else None,
                "meta": metadatas[i] if i < len(metadatas) else None,
            })

        context_chars = sum(len(hit["doc"]) for hit in hits)
        if context_chars > best_chars:
            best_hits = hits
            best_chars = context_chars

        if context_chars >= 80 or n >= limit:
            return best_hits

        n = min(n + 2, limit)


def build_context_from_hits(hits: list[dict], max_chars: int = 1800) -> str:
    parts = []
    total = 0

    for hit in hits:
        doc = (hit.get("doc") or "").strip()
        if not doc:
            continue
        if parts and total + len(doc) + 2 > max_chars:
            break
        parts.append(doc)
        total += len(doc) + 2

    return "\n\n".join(parts).strip()


def retrieve_context(query: str, initial_n: int = 3, max_n: int = 8) -> tuple[str, list[dict]]:
    hits = retrieve_hits(query, initial_n=initial_n, max_n=max_n)
    return build_context_from_hits(hits), hits


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

    return len(context) >= 40


def extract_definition_from_context(query: str, context: str) -> str | None:
    q = query.lower()
    c = context

    if "метеор" in q:
        m = re.search(r"(Метеор[-–— ]?М\s*—\s*это[^\n\.]*[\.]?)", c, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

    m = re.search(r"^(.{0,80}?—\s*это[^\n\.]*[\.]?)", c, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()

    return None

def is_definitional(q: str) -> bool:
    ql = q.lower()
    return any(x in ql for x in ["что такое", "что значит", "определи", "дать определение"])


def generate_answer_strict(query: str, context: str, hits: list[dict] | None = None) -> str:
    if is_definitional(query):
        defin = extract_definition_from_context(query, context)
        if defin:
            return defin

    extractive = find_extractive_answer(query, hits or [])
    if extractive:
        return extractive

    if not context.strip():
        return REFUSAL

    system = (
        "Ты — ИИ-ассистент по космонавтике с RAG.\n"
        "Отвечай только по фрагментам базы знаний из блока КОНТЕКСТ.\n"
        "Нельзя добавлять факты не из контекста.\n"
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
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs.input_ids.shape[1]
    text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^ответ:\s*", "", text, flags=re.IGNORECASE)

    if not text or len(text) < 3:
        return REFUSAL
    if is_refusal_like(text):
        return REFUSAL
    return text

def transcribe_audio_bytes(data: bytes, language: str = "ru") -> str:
    # faster-whisper удобнее скармливать файлом
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()

        segments, info = whisper_model.transcribe(
            tmp.name,
            language=language,   # "ru" или None (авто)
            vad_filter=True,     # ускоряет на паузах
            beam_size=5
        )

        text = " ".join(seg.text for seg in segments).strip()
        return text


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
        
        t0 = time.time()
        context, hits = retrieve_context(q, initial_n=3, max_n=9)
        t1 = time.time()

        if not context:
            return {"query": q, "answer": REFUSAL, "context_used": False}

        answer = generate_answer_strict(q, context, hits)
        t2 = time.time()
        print(f"[timing] retrieve={t1-t0:.3f}s | generate={t2-t1:.3f}s | total={t2-t0:.3f}s")

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
        context, hits = retrieve_context(q, initial_n=3, max_n=9)
        t3 = time.time()

        if not context:
            return {
                "transcript": transcript,
                "query": q,
                "answer": REFUSAL,
                "context_used": False,
                "timing": {"asr": round(t1 - t0, 3), "retrieve": round(t3 - t2, 3)}
            }

        answer = generate_answer_strict(q, context, hits)
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
    hits = retrieve_hits(q, initial_n=5, max_n=8)
    context = build_context_from_hits(hits)
    extractive = find_extractive_answer(q, hits)

    top = []
    for i, hit in enumerate(hits):
        top.append({
            "i": i,
            "distance": hit.get("distance"),
            "meta": hit.get("meta"),
            "doc": hit.get("doc"),
        })

    return {
        "query": q,
        "server_cwd": os.getcwd(),
        "db_path": DB_PATH if "DB_PATH" in globals() else "unknown",
        "collection_count": collection.count(),
        "retrieved_docs": len(hits),
        "context_chars": len(context),
        "context": context,
        "extractive_answer": extractive,
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
