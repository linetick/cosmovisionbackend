import os
import glob
import re
from typing import List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

RAW_DIR = "knowledge_raw/*.md"
DB_PATH = "knowledge_db"
COLLECTION = "satellites"

MIN_CHUNK_CHARS = 120
MAX_CHUNK_CHARS = 900

DEF_MARKERS = ["— это", "это ", "предназнач", "представляет собой", "служит для", "используется"]

print("Загрузка модели эмбеддингов...")
embedder = SentenceTransformer("intfloat/multilingual-e5-small")


def split_by_sections(md_text: str) -> List[Tuple[str, str]]:
    """Секции только по # и ##. Всё остальное (# ### ####) оставляем внутри тела."""
    lines = md_text.splitlines()
    sections: List[Tuple[str, str]] = []

    current_title = ""
    current_body: List[str] = []

    def flush():
        nonlocal current_title, current_body
        body = "\n".join(current_body).strip()
        if body or current_title:
            sections.append((current_title.strip(), body))
        current_body = []

    for line in lines:
        s = line.strip()
        if re.match(r"^#{1,2}\s+", s):
            flush()
            current_title = s.lstrip("#").strip()
        else:
            current_body.append(line)

    flush()
    return sections


def paragraphize(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def has_definition_marker(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in DEF_MARKERS)


def build_chunks(title: str, body: str) -> List[str]:
    paras = paragraphize(body)
    if not paras:
        return []

    header = f"{title}\n" if title else ""

    # 1) Если первый абзац похож на определение — делаем его отдельным чанком
    chunks: List[str] = []
    start_idx = 0
    if has_definition_marker(paras[0]):
        first = (header + paras[0]).strip()
        chunks.append(first)
        start_idx = 1

    # 2) Остальное склеиваем как раньше
    buf = ""

    def push_buf():
        nonlocal buf
        if not buf.strip():
            buf = ""
            return
        t = (header + buf).strip()
        chunks.append(t)
        buf = ""

    for p in paras[start_idx:]:
        if not buf:
            buf = p
        else:
            if len(buf) + 2 + len(p) <= MAX_CHUNK_CHARS:
                buf = buf + "\n\n" + p
            else:
                push_buf()
                buf = p

        if len(buf) >= MIN_CHUNK_CHARS:
            push_buf()

    if buf:
        push_buf()

    # 3) Склейка слишком коротких чанков к предыдущему
    merged: List[str] = []
    for ch in chunks:
        if merged and len(ch) < MIN_CHUNK_CHARS:
            merged[-1] = (merged[-1] + "\n\n" + ch).strip()
        else:
            merged.append(ch)

    return merged


print("Чтение исходных данных...")
docs_clean: List[str] = []
metadatas = []
ids = []

for file_path in glob.glob(RAW_DIR):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    source = os.path.basename(file_path)
    sections = split_by_sections(content)

    chunk_idx = 0
    for title, body in sections:
        if not (title or body.strip()):
            continue

        chunks = build_chunks(title, body)
        for ch in chunks:
            ch = ch.strip()
            if not ch:
                continue

            docs_clean.append(ch)
            metadatas.append({"source": source, "section": title, "chunk": chunk_idx})
            ids.append(f"{source}_{chunk_idx}")
            chunk_idx += 1

print("Создание векторной базы...")
client = chromadb.PersistentClient(path=DB_PATH)

try:
    client.delete_collection(COLLECTION)
except Exception:
    pass

collection = client.get_or_create_collection(
    COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

print("Генерация эмбеддингов...")
to_embed = ["passage: " + d for d in docs_clean]
embeddings = embedder.encode(to_embed, normalize_embeddings=True)

print("Сохранение в ChromaDB...")
collection.add(
    embeddings=embeddings,
    documents=docs_clean,
    metadatas=metadatas,
    ids=ids,
)

print(f"✅ База знаний создана! Сохранено {len(docs_clean)} фрагментов.")