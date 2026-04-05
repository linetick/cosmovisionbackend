import os
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import APP_TITLE, APP_VERSION, DB_PATH, MODEL_FILES_DIR, MODEL_REGISTRY_PATH, REFUSAL
from .model_store import ensure_model_storage, get_model_file_path, get_model_metadata, list_models
from .query_logic import (
    build_context_from_hits,
    classify_client_command_with_llm,
    detect_client_command,
    find_extractive_answer,
    generate_answer_llm_only,
    generate_answer_strict,
    has_sufficient_context_relevance,
    is_off_topic,
    looks_like_client_command,
    normalize_query,
    retrieve_context,
    retrieve_hits,
    transcribe_audio_bytes_detailed,
)
from .runtime import collection, warmup


class QueryRequest(BaseModel):
    text: str


app = FastAPI(title=APP_TITLE, version=APP_VERSION)
ensure_model_storage()


def command_response(query: str, command_type: str, answer: str) -> dict:
    return {
        "query": query,
        "intent": "client_command",
        "client_command": {
            "type": command_type,
        },
        "answer": answer,
        "context_used": False,
    }


def unknown_command_response(query: str) -> dict:
    return {
        "query": query,
        "intent": "unknown_command",
        "client_command": None,
        "answer": "Команда не распознана.",
        "context_used": False,
    }


def resolve_client_command(query: str) -> tuple[dict | None, dict]:
    debug = {
        "normalized_query": query,
        "rule_match": None,
        "looks_like_command": False,
        "llm_match": None,
        "resolution": "knowledge_answer",
    }

    matched_command = detect_client_command(query)
    if matched_command:
        debug["rule_match"] = matched_command["type"]
        debug["resolution"] = "rule_match"
        return matched_command, debug

    looks_like = looks_like_client_command(query)
    debug["looks_like_command"] = looks_like
    if not looks_like:
        return None, debug

    llm_command = classify_client_command_with_llm(query)
    if llm_command:
        debug["llm_match"] = llm_command["type"]
        debug["resolution"] = "llm_match"
        return llm_command, debug

    debug["resolution"] = "unknown_command"
    return None, debug


@app.post("/query")
def handle_query(req: QueryRequest):
    try:
        t0 = time.time()
        raw = req.text or ""
        q = normalize_query(raw)
        t1 = time.time()

        if not q:
            return {
                "query": raw,
                "intent": "knowledge_answer",
                "client_command": None,
                "answer": REFUSAL,
                "context_used": False,
                "timing": {
                    "normalize": round(t1 - t0, 3),
                    "total": round(t1 - t0, 3),
                },
            }

        matched_command, command_debug = resolve_client_command(q)
        if matched_command:
            result = command_response(q, matched_command["type"], matched_command["answer"])
            result["timing"] = {
                "normalize": round(t1 - t0, 3),
                "total": round(t1 - t0, 3),
            }
            return result

        if command_debug["resolution"] == "unknown_command":
            result = unknown_command_response(q)
            result["timing"] = {
                "normalize": round(t1 - t0, 3),
                "total": round(t1 - t0, 3),
            }
            return result

        off_topic = is_off_topic(q)
        t2 = time.time()
        if off_topic:
            return {
                "query": q,
                "intent": "off_topic",
                "client_command": None,
                "answer": (
                    "Я отвечаю только по космонавтике из базы знаний. "
                    "Спроси, например: «Как устроены солнечные панели на Метеоре-М?»"
                ),
                "context_used": False,
                "timing": {
                    "normalize": round(t1 - t0, 3),
                    "topic_check": round(t2 - t1, 3),
                    "total": round(t2 - t0, 3),
                },
            }

        t3 = time.time()
        context, hits, retrieve_stats = retrieve_context(q, initial_n=3, max_n=9)
        t4 = time.time()
        if not context:
            return {
                "query": q,
                "intent": "knowledge_answer",
                "client_command": None,
                "answer": REFUSAL,
                "context_used": False,
                "timing": {
                    "normalize": round(t1 - t0, 3),
                    "topic_check": round(t2 - t1, 3),
                    "retrieve": round(t4 - t3, 3),
                    "retrieve_embed": round(retrieve_stats["embed"], 3),
                    "retrieve_search": round(retrieve_stats["search"], 3),
                    "retrieve_postprocess": round(retrieve_stats["postprocess"], 3),
                    "retrieve_context_build": round(retrieve_stats["context_build"], 3),
                    "total": round(t4 - t0, 3),
                },
            }

        t5 = time.time()
        answer = generate_answer_strict(q, context, hits)
        t6 = time.time()
        return {
            "query": q,
            "intent": "knowledge_answer",
            "client_command": None,
            "answer": answer,
            "context_used": (answer != REFUSAL),
            "timing": {
                "normalize": round(t1 - t0, 3),
                "topic_check": round(t2 - t1, 3),
                "retrieve": round(t4 - t3, 3),
                "retrieve_embed": round(retrieve_stats["embed"], 3),
                "retrieve_search": round(retrieve_stats["search"], 3),
                "retrieve_postprocess": round(retrieve_stats["postprocess"], 3),
                "retrieve_context_build": round(retrieve_stats["context_build"], 3),
                "generate": round(t6 - t5, 3),
                "total": round(t6 - t0, 3),
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(exc)}")


@app.post("/query_llm_only")
def handle_query_llm_only(req: QueryRequest):
    try:
        t0 = time.time()
        raw = req.text or ""
        q = normalize_query(raw)
        t1 = time.time()

        if not q:
            return {
                "mode": "llm_only",
                "query": raw,
                "intent": "knowledge_answer",
                "client_command": None,
                "answer": REFUSAL,
                "context_used": False,
                "timing": {
                    "normalize": round(t1 - t0, 3),
                    "total": round(t1 - t0, 3),
                },
            }

        t2 = time.time()
        answer = generate_answer_llm_only(q)
        t3 = time.time()
        return {
            "mode": "llm_only",
            "query": q,
            "intent": "knowledge_answer",
            "client_command": None,
            "answer": answer,
            "context_used": False,
            "timing": {
                "normalize": round(t1 - t0, 3),
                "generate": round(t3 - t2, 3),
                "total": round(t3 - t0, 3),
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка LLM-only обработки: {str(exc)}")


@app.post("/query_audio")
async def handle_query_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            return {
                "answer": REFUSAL,
                "intent": "knowledge_answer",
                "client_command": None,
                "context_used": False,
                "transcript": "",
            }

        t0 = time.time()
        transcript, asr_stats = transcribe_audio_bytes_detailed(audio_bytes, language="ru")
        t1 = time.time()
        q = normalize_query(transcript)
        t2 = time.time()

        if not q:
            return {
                "answer": REFUSAL,
                "intent": "knowledge_answer",
                "client_command": None,
                "context_used": False,
                "transcript": transcript,
                "timing": {
                    "asr": round(t1 - t0, 3),
                    "audio_prepare": round(asr_stats["audio_prepare"], 3),
                    "transcribe": round(asr_stats["transcribe"], 3),
                    "normalize": round(t2 - t1, 3),
                    "total": round(t2 - t0, 3),
                },
            }

        matched_command, command_debug = resolve_client_command(q)
        if matched_command:
            result = command_response(q, matched_command["type"], matched_command["answer"])
            result["transcript"] = transcript
            result["timing"] = {
                "asr": round(t1 - t0, 3),
                "audio_prepare": round(asr_stats["audio_prepare"], 3),
                "transcribe": round(asr_stats["transcribe"], 3),
                "normalize": round(t2 - t1, 3),
                "total": round(t2 - t0, 3),
            }
            return result

        if command_debug["resolution"] == "unknown_command":
            result = unknown_command_response(q)
            result["transcript"] = transcript
            result["timing"] = {
                "asr": round(t1 - t0, 3),
                "audio_prepare": round(asr_stats["audio_prepare"], 3),
                "transcribe": round(asr_stats["transcribe"], 3),
                "normalize": round(t2 - t1, 3),
                "total": round(t2 - t0, 3),
            }
            return result

        off_topic = is_off_topic(q)
        t3 = time.time()
        if off_topic:
            return {
                "transcript": transcript,
                "query": q,
                "intent": "off_topic",
                "client_command": None,
                "answer": (
                    "Я отвечаю только по космонавтике из базы знаний. "
                    "Спроси, например: «Как устроены солнечные панели на Метеоре-М?»"
                ),
                "context_used": False,
                "timing": {
                    "asr": round(t1 - t0, 3),
                    "audio_prepare": round(asr_stats["audio_prepare"], 3),
                    "transcribe": round(asr_stats["transcribe"], 3),
                    "normalize": round(t2 - t1, 3),
                    "topic_check": round(t3 - t2, 3),
                    "total": round(t3 - t0, 3),
                },
            }

        t4 = time.time()
        context, hits, retrieve_stats = retrieve_context(q, initial_n=3, max_n=9)
        t5 = time.time()
        if not context:
            return {
                "transcript": transcript,
                "query": q,
                "intent": "knowledge_answer",
                "client_command": None,
                "answer": REFUSAL,
                "context_used": False,
                "timing": {
                    "asr": round(t1 - t0, 3),
                    "audio_prepare": round(asr_stats["audio_prepare"], 3),
                    "transcribe": round(asr_stats["transcribe"], 3),
                    "normalize": round(t2 - t1, 3),
                    "topic_check": round(t3 - t2, 3),
                    "retrieve": round(t5 - t4, 3),
                    "retrieve_embed": round(retrieve_stats["embed"], 3),
                    "retrieve_search": round(retrieve_stats["search"], 3),
                    "retrieve_postprocess": round(retrieve_stats["postprocess"], 3),
                    "retrieve_context_build": round(retrieve_stats["context_build"], 3),
                    "total": round(t5 - t0, 3),
                },
            }

        t6 = time.time()
        answer = generate_answer_strict(q, context, hits)
        t7 = time.time()
        return {
            "transcript": transcript,
            "query": q,
            "intent": "knowledge_answer",
            "client_command": None,
            "answer": answer,
            "context_used": (answer != REFUSAL),
            "timing": {
                "asr": round(t1 - t0, 3),
                "audio_prepare": round(asr_stats["audio_prepare"], 3),
                "transcribe": round(asr_stats["transcribe"], 3),
                "normalize": round(t2 - t1, 3),
                "topic_check": round(t3 - t2, 3),
                "retrieve": round(t5 - t4, 3),
                "retrieve_embed": round(retrieve_stats["embed"], 3),
                "retrieve_search": round(retrieve_stats["search"], 3),
                "retrieve_postprocess": round(retrieve_stats["postprocess"], 3),
                "retrieve_context_build": round(retrieve_stats["context_build"], 3),
                "generate": round(t7 - t6, 3),
                "total": round(t7 - t0, 3),
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки аудио: {str(exc)}")


@app.get("/models")
def get_models():
    items = list_models()
    return {
        "items": items,
        "count": len(items),
        "storage_dir": MODEL_FILES_DIR,
    }


@app.get("/models/{model_id}")
def get_model(model_id: str):
    item = get_model_metadata(model_id)
    if not item:
        raise HTTPException(status_code=404, detail="3D-модель не найдена")
    return item


@app.get("/models/{model_id}/download")
def download_model(model_id: str):
    item = get_model_metadata(model_id)
    if not item:
        raise HTTPException(status_code=404, detail="3D-модель не найдена")

    file_path = get_model_file_path(model_id)
    if file_path is None:
        raise HTTPException(status_code=404, detail="Файл 3D-модели не найден")

    return FileResponse(
        path=file_path,
        filename=item["file_name"],
        media_type="application/octet-stream",
    )


@app.get("/debug/kb")
def debug_kb():
    return {
        "db_path": DB_PATH,
        "cwd": os.getcwd(),
        "count": collection.count(),
        "models_registry_path": MODEL_REGISTRY_PATH,
    }


@app.post("/debug/search")
def debug_search(req: QueryRequest):
    q = normalize_query(req.text)
    hits, retrieve_stats = retrieve_hits(q, initial_n=5, max_n=8)
    t_context = time.perf_counter()
    context = build_context_from_hits(hits)
    retrieve_stats["context_build"] = time.perf_counter() - t_context
    retrieve_stats["total"] = (
        retrieve_stats["embed"] + retrieve_stats["search"] +
        retrieve_stats["postprocess"] + retrieve_stats["context_build"]
    )
    extractive = find_extractive_answer(q, hits)
    relevant, relevance = has_sufficient_context_relevance(q, context)
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
        "db_path": DB_PATH,
        "collection_count": collection.count(),
        "retrieved_docs": len(hits),
        "context_chars": len(context),
        "context": context,
        "extractive_answer": extractive,
        "relevance": relevance,
        "context_relevant": relevant,
        "timing": {
            "retrieve_embed": round(retrieve_stats["embed"], 3),
            "retrieve_search": round(retrieve_stats["search"], 3),
            "retrieve_postprocess": round(retrieve_stats["postprocess"], 3),
            "retrieve_context_build": round(retrieve_stats["context_build"], 3),
            "retrieve_total": round(retrieve_stats["total"], 3),
        },
        "top": top,
    }


@app.post("/debug/command")
def debug_command(req: QueryRequest):
    t0 = time.time()
    raw = req.text or ""
    q = normalize_query(raw)
    t1 = time.time()

    matched_command, command_debug = resolve_client_command(q)
    if matched_command:
        result = command_response(q, matched_command["type"], matched_command["answer"])
    elif command_debug["resolution"] == "unknown_command":
        result = unknown_command_response(q)
    else:
        result = {
            "query": q,
            "intent": "knowledge_answer",
            "client_command": None,
            "answer": "Запрос не классифицирован как команда управления.",
            "context_used": False,
        }

    result["raw_query"] = raw
    result["command_debug"] = command_debug
    result["timing"] = {
        "normalize": round(t1 - t0, 3),
        "total": round(time.time() - t0, 3),
    }
    return result


warmup(retrieve_hits, generate_answer_strict)
