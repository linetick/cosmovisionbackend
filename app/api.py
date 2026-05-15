import os
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import AR_COMPACT_RESPONSES, APP_TITLE, APP_VERSION, DB_PATH, MODEL_FILES_DIR, MODEL_REGISTRY_PATH, REFUSAL
from .model_store import ensure_model_storage, get_model_file_path, get_model_metadata, list_models
from .query_logic import (
    COMMAND_ANSWERS,
    build_context_from_hits,
    classify_query_with_llm,
    detect_client_command,
    extract_knowledge_query_from_compound,
    find_extractive_answer,
    generate_answer_compact,
    generate_answer_fallback,
    generate_answer_llm_only,
    generate_answer_strict,
    has_sufficient_context_relevance,
    infer_client_command_from_markers,
    inject_spacecraft_context,
    is_off_topic,
    looks_like_knowledge_request,
    looks_like_client_command,
    looks_like_meta_request,
    normalize_query,
    retrieve_context,
    retrieve_hits,
    split_meta_and_knowledge_request,
    transcribe_audio_bytes_detailed,
)
from .runtime import collection, warmup


class QueryRequest(BaseModel):
    text: str
    current_model_id: str | None = None
    current_spacecraft: str | None = None


app = FastAPI(title=APP_TITLE, version=APP_VERSION)
ensure_model_storage()


def use_compact_generation() -> bool:
    return AR_COMPACT_RESPONSES


def command_response(query: str, command_type: str, answer: str, intent: str = "action") -> dict:
    return {
        "query": query,
        "intent": intent,
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


def build_compound_answer(command_answer: str, knowledge_answer: str) -> str:
    knowledge_answer = (knowledge_answer or "").strip()
    if not knowledge_answer or knowledge_answer == REFUSAL:
        return f"{command_answer} {REFUSAL}"
    return f"{command_answer} {knowledge_answer}"


def join_answer_parts(*parts: str | None) -> str:
    cleaned = [part.strip() for part in parts if part and part.strip()]
    return " ".join(cleaned).strip()


def build_meta_answer(meta_answer: str, knowledge_answer: str | None = None) -> str:
    meta_answer = (meta_answer or "").strip()
    knowledge_answer = (knowledge_answer or "").strip()
    if not meta_answer:
        return knowledge_answer or REFUSAL
    if not knowledge_answer:
        return meta_answer
    if knowledge_answer == REFUSAL:
        return join_answer_parts(meta_answer, REFUSAL)
    return join_answer_parts(meta_answer, knowledge_answer)


def generate_meta_answer(meta_request: str) -> str:
    meta_request = (meta_request or "").strip()
    if not meta_request:
        return REFUSAL
    return generate_answer_llm_only(
        f"{meta_request}\n\nОтветь одним коротким предложением про ассистента, приложение или текущий режим работы."
    )


def resolve_current_spacecraft(
    current_model_id: str | None = None,
    current_spacecraft: str | None = None,
) -> str | None:
    spacecraft = (current_spacecraft or "").strip()
    if spacecraft:
        return spacecraft

    model_id = (current_model_id or "").strip()
    if not model_id:
        return None

    item = get_model_metadata(model_id)
    if not item:
        return None

    spacecraft = (item.get("spacecraft") or item.get("name") or "").strip()
    return spacecraft or None


def resolve_client_command(query: str) -> tuple[dict | None, dict]:
    debug = {
        "normalized_query": query,
        "llm_route": None,
        "rule_match": None,
        "marker_match": None,
        "looks_like_command": False,
        "resolution": "info",
    }

    if not looks_like_client_command(query) and (looks_like_meta_request(query) or looks_like_knowledge_request(query)):
        meta_request, knowledge_text = split_meta_and_knowledge_request(query)
        if meta_request or knowledge_text:
            debug["resolution"] = "fast_info_local"
            route = {"intent": "info"}
            if meta_request:
                route["meta_request"] = meta_request
            if knowledge_text:
                route["knowledge_text"] = knowledge_text
            elif not meta_request:
                route["knowledge_text"] = query
            return route, debug

    llm_route = classify_query_with_llm(query)
    if llm_route:
        debug["llm_route"] = llm_route
        if llm_route["intent"] == "action":
            debug["resolution"] = "llm_action"
            return {
                "intent": "action",
                "type": llm_route["command_type"],
                "answer": COMMAND_ANSWERS[llm_route["command_type"]],
            }, debug
        if llm_route["intent"] == "hybrid":
            debug["resolution"] = "llm_hybrid"
            return {
                "intent": "hybrid",
                "type": llm_route["command_type"],
                "answer": COMMAND_ANSWERS[llm_route["command_type"]],
                "knowledge_text": (llm_route.get("knowledge_text") or "").strip() or None,
                "meta_request": (llm_route.get("meta_request") or "").strip() or None,
            }, debug
        if llm_route["intent"] == "unknown_command":
            marker_command = infer_client_command_from_markers(query)
            debug["marker_match"] = marker_command
            if marker_command and looks_like_knowledge_request(query):
                debug["resolution"] = "llm_unknown_to_fallback_hybrid"
                return {
                    "intent": "hybrid",
                    "type": marker_command,
                    "answer": COMMAND_ANSWERS[marker_command],
                    "knowledge_text": None,
                }, debug
            if marker_command:
                debug["resolution"] = "llm_unknown_to_fallback_action"
                return {
                    "intent": "action",
                    "type": marker_command,
                    "answer": COMMAND_ANSWERS[marker_command],
                }, debug
            debug["resolution"] = "llm_unknown_command"
            return {"intent": "unknown_command"}, debug

        marker_command = infer_client_command_from_markers(query)
        debug["marker_match"] = marker_command
        if marker_command and looks_like_knowledge_request(query):
            debug["resolution"] = "llm_info_to_fallback_hybrid"
            return {
                "intent": "hybrid",
                "type": marker_command,
                "answer": COMMAND_ANSWERS[marker_command],
                "knowledge_text": (llm_route.get("knowledge_text") or "").strip() or None,
                "meta_request": (llm_route.get("meta_request") or "").strip() or None,
            }, debug
        debug["resolution"] = "llm_info"
        route = {
            "intent": "info",
        }
        knowledge_text = (llm_route.get("knowledge_text") or "").strip()
        meta_request = (llm_route.get("meta_request") or "").strip()
        if knowledge_text:
            route["knowledge_text"] = knowledge_text
        elif not meta_request:
            route["knowledge_text"] = query
        if meta_request:
            route["meta_request"] = meta_request
        return route, debug

    matched_command = detect_client_command(query)
    if matched_command:
        debug["rule_match"] = matched_command["type"]
        debug["resolution"] = "fallback_rule_match"
        return {
            "intent": "action",
            "type": matched_command["type"],
            "answer": matched_command["answer"],
        }, debug

    looks_like = looks_like_client_command(query)
    debug["looks_like_command"] = looks_like
    if looks_like:
        debug["resolution"] = "fallback_unknown_command"
        return {"intent": "unknown_command"}, debug

    debug["resolution"] = "fallback_info"
    return {
        "intent": "info",
        "knowledge_text": query,
    }, debug


@app.post("/query")
def handle_query(req: QueryRequest):
    try:
        t0 = time.time()
        raw = req.text or ""
        q = normalize_query(raw)
        t1 = time.time()
        current_spacecraft = resolve_current_spacecraft(
            current_model_id=req.current_model_id,
            current_spacecraft=req.current_spacecraft,
        )
        if current_spacecraft:
            q = inject_spacecraft_context(q, current_spacecraft)

        if not q:
            return {
                "query": raw,
                "intent": "info",
                "client_command": None,
                "answer": REFUSAL,
                "context_used": False,
                "timing": {
                    "normalize": round(t1 - t0, 3),
                    "total": round(t1 - t0, 3),
                },
            }

        matched_command, command_debug = resolve_client_command(q)
        if matched_command and matched_command["intent"] == "action":
            result = command_response(q, matched_command["type"], matched_command["answer"])
            result["timing"] = {
                "normalize": round(t1 - t0, 3),
                "total": round(t1 - t0, 3),
            }
            return result

        if matched_command and matched_command["intent"] == "unknown_command":
            result = unknown_command_response(q)
            result["timing"] = {
                "normalize": round(t1 - t0, 3),
                "total": round(t1 - t0, 3),
            }
            return result

        meta_request = None
        if matched_command and matched_command["intent"] in {"info", "hybrid"}:
            meta_request = normalize_query((matched_command.get("meta_request") or "").strip()) or None

        knowledge_query = q
        if matched_command and matched_command["intent"] == "hybrid":
            knowledge_query = (matched_command.get("knowledge_text") or "").strip()
            if not knowledge_query:
                knowledge_query = extract_knowledge_query_from_compound(
                    q,
                    matched_command["type"],
                    current_spacecraft=current_spacecraft,
                )
            else:
                knowledge_query = normalize_query(knowledge_query)
                if current_spacecraft:
                    knowledge_query = inject_spacecraft_context(knowledge_query, current_spacecraft)
        elif matched_command and matched_command["intent"] == "info":
            routed_knowledge_text = (matched_command.get("knowledge_text") or "").strip()
            if routed_knowledge_text:
                knowledge_query = normalize_query(routed_knowledge_text)
            elif meta_request:
                knowledge_query = ""
            else:
                knowledge_query = q
            if knowledge_query and current_spacecraft:
                knowledge_query = inject_spacecraft_context(knowledge_query, current_spacecraft)

        if meta_request and not knowledge_query:
            t2 = time.time()
            meta_answer = generate_meta_answer(meta_request)
            t3 = time.time()
            return {
                "query": q,
                "intent": "info",
                "client_command": None,
                "meta_request": meta_request,
                "answer": meta_answer,
                "context_used": False,
                "timing": {
                    "normalize": round(t1 - t0, 3),
                    "meta_generate": round(t3 - t2, 3),
                    "total": round(t3 - t0, 3),
                },
            }

        topic_check_start = t1
        off_topic = is_off_topic(knowledge_query)
        t2 = time.time()
        if off_topic and not (matched_command and matched_command["intent"] == "hybrid"):
            answer = (
                "Я отвечаю только по космонавтике из базы знаний. "
                "Спроси, например: «Как устроены солнечные панели на Метеоре-М?»"
            )
            result = {
                "query": q,
                "intent": "off_topic",
                "client_command": None,
                "answer": answer,
                "context_used": False,
                "timing": {
                    "normalize": round(t1 - t0, 3),
                    "topic_check": round(t2 - topic_check_start, 3),
                    "total": round(t2 - t0, 3),
                },
            }
            return result

        t3 = time.time()
        context, hits, retrieve_stats = retrieve_context(knowledge_query, initial_n=3, max_n=9)
        t4 = time.time()
        if not context:
            t_fallback0 = time.time()
            fallback_answer = generate_answer_fallback(knowledge_query, meta_request=meta_request)
            t_fallback1 = time.time()
            if fallback_answer != REFUSAL:
                if matched_command and matched_command["intent"] == "hybrid":
                    result = {
                        "query": q,
                        "intent": "hybrid",
                        "client_command": {
                            "type": matched_command["type"],
                        },
                        "knowledge_query": knowledge_query,
                        "answer": build_compound_answer(matched_command["answer"], fallback_answer),
                        "context_used": False,
                        "fallback_used": True,
                        "timing": {
                            "normalize": round(t1 - t0, 3),
                            "topic_check": round(t2 - topic_check_start, 3),
                            "retrieve": round(t4 - t3, 3),
                            "retrieve_embed": round(retrieve_stats["embed"], 3),
                            "retrieve_search": round(retrieve_stats["search"], 3),
                            "retrieve_postprocess": round(retrieve_stats["postprocess"], 3),
                            "retrieve_context_build": round(retrieve_stats["context_build"], 3),
                            "fallback_generate": round(t_fallback1 - t_fallback0, 3),
                            "total": round(t_fallback1 - t0, 3),
                        },
                    }
                    if meta_request:
                        result["meta_request"] = meta_request
                    return result
                result = {
                    "query": q,
                    "intent": "info",
                    "client_command": None,
                    "answer": fallback_answer,
                    "context_used": False,
                    "fallback_used": True,
                    "timing": {
                        "normalize": round(t1 - t0, 3),
                        "topic_check": round(t2 - topic_check_start, 3),
                        "retrieve": round(t4 - t3, 3),
                        "retrieve_embed": round(retrieve_stats["embed"], 3),
                        "retrieve_search": round(retrieve_stats["search"], 3),
                        "retrieve_postprocess": round(retrieve_stats["postprocess"], 3),
                        "retrieve_context_build": round(retrieve_stats["context_build"], 3),
                        "fallback_generate": round(t_fallback1 - t_fallback0, 3),
                        "total": round(t_fallback1 - t0, 3),
                    },
                }
                if meta_request:
                    result["meta_request"] = meta_request
                return result
            if matched_command and matched_command["intent"] == "hybrid":
                return {
                    "query": q,
                    "intent": "hybrid",
                    "client_command": {
                        "type": matched_command["type"],
                    },
                    "knowledge_query": knowledge_query,
                    "answer": build_compound_answer(matched_command["answer"], REFUSAL),
                    "context_used": False,
                    "timing": {
                        "normalize": round(t1 - t0, 3),
                        "topic_check": round(t2 - topic_check_start, 3),
                        "retrieve": round(t4 - t3, 3),
                        "retrieve_embed": round(retrieve_stats["embed"], 3),
                        "retrieve_search": round(retrieve_stats["search"], 3),
                        "retrieve_postprocess": round(retrieve_stats["postprocess"], 3),
                        "retrieve_context_build": round(retrieve_stats["context_build"], 3),
                        "total": round(t4 - t0, 3),
                    },
                }
            return {
                "query": q,
                "intent": "info",
                "client_command": None,
                "answer": REFUSAL,
                "context_used": False,
                "timing": {
                    "normalize": round(t1 - t0, 3),
                    "topic_check": round(t2 - topic_check_start, 3),
                    "retrieve": round(t4 - t3, 3),
                    "retrieve_embed": round(retrieve_stats["embed"], 3),
                    "retrieve_search": round(retrieve_stats["search"], 3),
                    "retrieve_postprocess": round(retrieve_stats["postprocess"], 3),
                    "retrieve_context_build": round(retrieve_stats["context_build"], 3),
                    "total": round(t4 - t0, 3),
                },
            }

        t5 = time.time()
        if use_compact_generation():
            answer = generate_answer_compact(knowledge_query, context, hits, meta_request=meta_request)
        elif matched_command and matched_command["intent"] == "hybrid":
            answer = generate_answer_compact(knowledge_query, context, hits, meta_request=meta_request)
        else:
            answer = generate_answer_strict(knowledge_query, context, hits, meta_request=meta_request)
        t6 = time.time()
        if answer == REFUSAL:
            t_fallback0 = time.time()
            fallback_answer = generate_answer_fallback(knowledge_query, meta_request=meta_request)
            t_fallback1 = time.time()
            if fallback_answer != REFUSAL:
                if matched_command and matched_command["intent"] == "hybrid":
                    result = {
                        "query": q,
                        "intent": "hybrid",
                        "client_command": {
                            "type": matched_command["type"],
                        },
                        "knowledge_query": knowledge_query,
                        "answer": build_compound_answer(matched_command["answer"], fallback_answer),
                        "context_used": False,
                        "fallback_used": True,
                        "timing": {
                            "normalize": round(t1 - t0, 3),
                            "topic_check": round(t2 - topic_check_start, 3),
                            "retrieve": round(t4 - t3, 3),
                            "retrieve_embed": round(retrieve_stats["embed"], 3),
                            "retrieve_search": round(retrieve_stats["search"], 3),
                            "retrieve_postprocess": round(retrieve_stats["postprocess"], 3),
                            "retrieve_context_build": round(retrieve_stats["context_build"], 3),
                            "generate": round(t6 - t5, 3),
                            "fallback_generate": round(t_fallback1 - t_fallback0, 3),
                            "total": round(t_fallback1 - t0, 3),
                        },
                    }
                    if meta_request:
                        result["meta_request"] = meta_request
                    return result
                result = {
                    "query": q,
                    "intent": "info",
                    "client_command": None,
                    "answer": fallback_answer,
                    "context_used": False,
                    "fallback_used": True,
                    "timing": {
                        "normalize": round(t1 - t0, 3),
                        "topic_check": round(t2 - topic_check_start, 3),
                        "retrieve": round(t4 - t3, 3),
                        "retrieve_embed": round(retrieve_stats["embed"], 3),
                        "retrieve_search": round(retrieve_stats["search"], 3),
                        "retrieve_postprocess": round(retrieve_stats["postprocess"], 3),
                        "retrieve_context_build": round(retrieve_stats["context_build"], 3),
                        "generate": round(t6 - t5, 3),
                        "fallback_generate": round(t_fallback1 - t_fallback0, 3),
                        "total": round(t_fallback1 - t0, 3),
                    },
                }
                if meta_request:
                    result["meta_request"] = meta_request
                return result
        if matched_command and matched_command["intent"] == "hybrid":
            return {
                "query": q,
                "intent": "hybrid",
                "client_command": {
                    "type": matched_command["type"],
                },
                "knowledge_query": knowledge_query,
                "answer": build_compound_answer(matched_command["answer"], answer),
                "context_used": (answer != REFUSAL),
                "timing": {
                    "normalize": round(t1 - t0, 3),
                    "topic_check": round(t2 - topic_check_start, 3),
                    "retrieve": round(t4 - t3, 3),
                    "retrieve_embed": round(retrieve_stats["embed"], 3),
                    "retrieve_search": round(retrieve_stats["search"], 3),
                    "retrieve_postprocess": round(retrieve_stats["postprocess"], 3),
                    "retrieve_context_build": round(retrieve_stats["context_build"], 3),
                    "generate": round(t6 - t5, 3),
                    "total": round(t6 - t0, 3),
                },
            }
        result = {
            "query": q,
            "intent": "info",
            "client_command": None,
            "answer": answer,
            "context_used": (answer != REFUSAL),
            "timing": {
                "normalize": round(t1 - t0, 3),
                "topic_check": round(t2 - topic_check_start, 3),
                "retrieve": round(t4 - t3, 3),
                "retrieve_embed": round(retrieve_stats["embed"], 3),
                "retrieve_search": round(retrieve_stats["search"], 3),
                "retrieve_postprocess": round(retrieve_stats["postprocess"], 3),
                "retrieve_context_build": round(retrieve_stats["context_build"], 3),
                "generate": round(t6 - t5, 3),
                "total": round(t6 - t0, 3),
            },
        }
        if meta_request:
            result["meta_request"] = meta_request
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(exc)}")


@app.post("/query_llm_only")
def handle_query_llm_only(req: QueryRequest):
    try:
        t0 = time.time()
        raw = req.text or ""
        q = normalize_query(raw)
        t1 = time.time()
        current_spacecraft = resolve_current_spacecraft(
            current_model_id=req.current_model_id,
            current_spacecraft=req.current_spacecraft,
        )
        if current_spacecraft:
            q = inject_spacecraft_context(q, current_spacecraft)

        if not q:
            return {
                "mode": "llm_only",
                "query": raw,
                "intent": "info",
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
            "intent": "info",
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
async def handle_query_audio(
    file: UploadFile = File(...),
    current_model_id: str | None = Form(None),
    current_spacecraft: str | None = Form(None),
):
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            return {
                "answer": REFUSAL,
                "intent": "info",
                "client_command": None,
                "context_used": False,
                "transcript": "",
            }

        t0 = time.time()
        transcript, asr_stats = transcribe_audio_bytes_detailed(audio_bytes, language="ru")
        t1 = time.time()
        q = normalize_query(transcript)
        t2 = time.time()
        resolved_spacecraft = resolve_current_spacecraft(
            current_model_id=current_model_id,
            current_spacecraft=current_spacecraft,
        )
        if resolved_spacecraft:
            q = inject_spacecraft_context(q, resolved_spacecraft)

        if not q:
            return {
                "answer": REFUSAL,
                "intent": "info",
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
        if matched_command and matched_command["intent"] == "action":
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

        if matched_command and matched_command["intent"] == "unknown_command":
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

        meta_request = None
        if matched_command and matched_command["intent"] in {"info", "hybrid"}:
            meta_request = normalize_query((matched_command.get("meta_request") or "").strip()) or None

        knowledge_query = q
        if matched_command and matched_command["intent"] == "hybrid":
            knowledge_query = (matched_command.get("knowledge_text") or "").strip()
            if not knowledge_query:
                knowledge_query = extract_knowledge_query_from_compound(
                    q,
                    matched_command["type"],
                    current_spacecraft=resolved_spacecraft,
                )
            else:
                knowledge_query = normalize_query(knowledge_query)
                if resolved_spacecraft:
                    knowledge_query = inject_spacecraft_context(knowledge_query, resolved_spacecraft)
        elif matched_command and matched_command["intent"] == "info":
            routed_knowledge_text = (matched_command.get("knowledge_text") or "").strip()
            if routed_knowledge_text:
                knowledge_query = normalize_query(routed_knowledge_text)
            elif meta_request:
                knowledge_query = ""
            else:
                knowledge_query = q
            if knowledge_query and resolved_spacecraft:
                knowledge_query = inject_spacecraft_context(knowledge_query, resolved_spacecraft)

        if meta_request and not knowledge_query:
            t3 = time.time()
            meta_answer = generate_meta_answer(meta_request)
            t4 = time.time()
            return {
                "transcript": transcript,
                "query": q,
                "intent": "info",
                "client_command": None,
                "meta_request": meta_request,
                "answer": meta_answer,
                "context_used": False,
                "timing": {
                    "asr": round(t1 - t0, 3),
                    "audio_prepare": round(asr_stats["audio_prepare"], 3),
                    "transcribe": round(asr_stats["transcribe"], 3),
                    "normalize": round(t2 - t1, 3),
                    "meta_generate": round(t4 - t3, 3),
                    "total": round(t4 - t0, 3),
                },
            }

        topic_check_start = t2
        off_topic = is_off_topic(knowledge_query)
        t3 = time.time()
        if off_topic and not (matched_command and matched_command["intent"] == "hybrid"):
            answer = (
                "Я отвечаю только по космонавтике из базы знаний. "
                "Спроси, например: «Как устроены солнечные панели на Метеоре-М?»"
            )
            result = {
                "transcript": transcript,
                "query": q,
                "intent": "off_topic",
                "client_command": None,
                "answer": answer,
                "context_used": False,
                "timing": {
                    "asr": round(t1 - t0, 3),
                    "audio_prepare": round(asr_stats["audio_prepare"], 3),
                    "transcribe": round(asr_stats["transcribe"], 3),
                    "normalize": round(t2 - t1, 3),
                    "topic_check": round(t3 - topic_check_start, 3),
                    "total": round(t3 - t0, 3),
                },
            }
            return result

        t4 = time.time()
        context, hits, retrieve_stats = retrieve_context(knowledge_query, initial_n=3, max_n=9)
        t5 = time.time()
        if not context:
            t_fallback0 = time.time()
            fallback_answer = generate_answer_fallback(knowledge_query, meta_request=meta_request)
            t_fallback1 = time.time()
            if fallback_answer != REFUSAL:
                if matched_command and matched_command["intent"] == "hybrid":
                    result = {
                        "transcript": transcript,
                        "query": q,
                        "intent": "hybrid",
                        "client_command": {
                            "type": matched_command["type"],
                        },
                        "knowledge_query": knowledge_query,
                        "answer": build_compound_answer(matched_command["answer"], fallback_answer),
                        "context_used": False,
                        "fallback_used": True,
                        "timing": {
                            "asr": round(t1 - t0, 3),
                            "audio_prepare": round(asr_stats["audio_prepare"], 3),
                            "transcribe": round(asr_stats["transcribe"], 3),
                            "normalize": round(t2 - t1, 3),
                            "topic_check": round(t3 - topic_check_start, 3),
                            "retrieve": round(t5 - t4, 3),
                            "retrieve_embed": round(retrieve_stats["embed"], 3),
                            "retrieve_search": round(retrieve_stats["search"], 3),
                            "retrieve_postprocess": round(retrieve_stats["postprocess"], 3),
                            "retrieve_context_build": round(retrieve_stats["context_build"], 3),
                            "fallback_generate": round(t_fallback1 - t_fallback0, 3),
                            "total": round(t_fallback1 - t0, 3),
                        },
                    }
                    if meta_request:
                        result["meta_request"] = meta_request
                    return result
                result = {
                    "transcript": transcript,
                    "query": q,
                    "intent": "info",
                    "client_command": None,
                    "answer": fallback_answer,
                    "context_used": False,
                    "fallback_used": True,
                    "timing": {
                        "asr": round(t1 - t0, 3),
                        "audio_prepare": round(asr_stats["audio_prepare"], 3),
                        "transcribe": round(asr_stats["transcribe"], 3),
                        "normalize": round(t2 - t1, 3),
                        "topic_check": round(t3 - topic_check_start, 3),
                        "retrieve": round(t5 - t4, 3),
                        "retrieve_embed": round(retrieve_stats["embed"], 3),
                        "retrieve_search": round(retrieve_stats["search"], 3),
                        "retrieve_postprocess": round(retrieve_stats["postprocess"], 3),
                        "retrieve_context_build": round(retrieve_stats["context_build"], 3),
                        "fallback_generate": round(t_fallback1 - t_fallback0, 3),
                        "total": round(t_fallback1 - t0, 3),
                    },
                }
                if meta_request:
                    result["meta_request"] = meta_request
                return result
            if matched_command and matched_command["intent"] == "hybrid":
                return {
                    "transcript": transcript,
                    "query": q,
                    "intent": "hybrid",
                    "client_command": {
                        "type": matched_command["type"],
                    },
                    "knowledge_query": knowledge_query,
                    "answer": build_compound_answer(matched_command["answer"], REFUSAL),
                    "context_used": False,
                    "timing": {
                        "asr": round(t1 - t0, 3),
                        "audio_prepare": round(asr_stats["audio_prepare"], 3),
                        "transcribe": round(asr_stats["transcribe"], 3),
                        "normalize": round(t2 - t1, 3),
                        "topic_check": round(t3 - topic_check_start, 3),
                        "retrieve": round(t5 - t4, 3),
                        "retrieve_embed": round(retrieve_stats["embed"], 3),
                        "retrieve_search": round(retrieve_stats["search"], 3),
                        "retrieve_postprocess": round(retrieve_stats["postprocess"], 3),
                        "retrieve_context_build": round(retrieve_stats["context_build"], 3),
                        "total": round(t5 - t0, 3),
                    },
                }
            return {
                "transcript": transcript,
                "query": q,
                "intent": "info",
                "client_command": None,
                "answer": REFUSAL,
                "context_used": False,
                "timing": {
                    "asr": round(t1 - t0, 3),
                    "audio_prepare": round(asr_stats["audio_prepare"], 3),
                    "transcribe": round(asr_stats["transcribe"], 3),
                    "normalize": round(t2 - t1, 3),
                    "topic_check": round(t3 - topic_check_start, 3),
                    "retrieve": round(t5 - t4, 3),
                    "retrieve_embed": round(retrieve_stats["embed"], 3),
                    "retrieve_search": round(retrieve_stats["search"], 3),
                    "retrieve_postprocess": round(retrieve_stats["postprocess"], 3),
                    "retrieve_context_build": round(retrieve_stats["context_build"], 3),
                    "total": round(t5 - t0, 3),
                },
            }

        t6 = time.time()
        if use_compact_generation():
            answer = generate_answer_compact(knowledge_query, context, hits, meta_request=meta_request)
        elif matched_command and matched_command["intent"] == "hybrid":
            answer = generate_answer_compact(knowledge_query, context, hits, meta_request=meta_request)
        else:
            answer = generate_answer_strict(knowledge_query, context, hits, meta_request=meta_request)
        t7 = time.time()
        if answer == REFUSAL:
            t_fallback0 = time.time()
            fallback_answer = generate_answer_fallback(knowledge_query, meta_request=meta_request)
            t_fallback1 = time.time()
            if fallback_answer != REFUSAL:
                if matched_command and matched_command["intent"] == "hybrid":
                    result = {
                        "transcript": transcript,
                        "query": q,
                        "intent": "hybrid",
                        "client_command": {
                            "type": matched_command["type"],
                        },
                        "knowledge_query": knowledge_query,
                        "answer": build_compound_answer(matched_command["answer"], fallback_answer),
                        "context_used": False,
                        "fallback_used": True,
                        "timing": {
                            "asr": round(t1 - t0, 3),
                            "audio_prepare": round(asr_stats["audio_prepare"], 3),
                            "transcribe": round(asr_stats["transcribe"], 3),
                            "normalize": round(t2 - t1, 3),
                            "topic_check": round(t3 - topic_check_start, 3),
                            "retrieve": round(t5 - t4, 3),
                            "retrieve_embed": round(retrieve_stats["embed"], 3),
                            "retrieve_search": round(retrieve_stats["search"], 3),
                            "retrieve_postprocess": round(retrieve_stats["postprocess"], 3),
                            "retrieve_context_build": round(retrieve_stats["context_build"], 3),
                            "generate": round(t7 - t6, 3),
                            "fallback_generate": round(t_fallback1 - t_fallback0, 3),
                            "total": round(t_fallback1 - t0, 3),
                        },
                    }
                    if meta_request:
                        result["meta_request"] = meta_request
                    return result
                result = {
                    "transcript": transcript,
                    "query": q,
                    "intent": "info",
                    "client_command": None,
                    "answer": fallback_answer,
                    "context_used": False,
                    "fallback_used": True,
                    "timing": {
                        "asr": round(t1 - t0, 3),
                        "audio_prepare": round(asr_stats["audio_prepare"], 3),
                        "transcribe": round(asr_stats["transcribe"], 3),
                        "normalize": round(t2 - t1, 3),
                        "topic_check": round(t3 - topic_check_start, 3),
                        "retrieve": round(t5 - t4, 3),
                        "retrieve_embed": round(retrieve_stats["embed"], 3),
                        "retrieve_search": round(retrieve_stats["search"], 3),
                        "retrieve_postprocess": round(retrieve_stats["postprocess"], 3),
                        "retrieve_context_build": round(retrieve_stats["context_build"], 3),
                        "generate": round(t7 - t6, 3),
                        "fallback_generate": round(t_fallback1 - t_fallback0, 3),
                        "total": round(t_fallback1 - t0, 3),
                    },
                }
                if meta_request:
                    result["meta_request"] = meta_request
                return result
        if matched_command and matched_command["intent"] == "hybrid":
            return {
                "transcript": transcript,
                "query": q,
                "intent": "hybrid",
                "client_command": {
                    "type": matched_command["type"],
                },
                "knowledge_query": knowledge_query,
                "answer": build_compound_answer(matched_command["answer"], answer),
                "context_used": (answer != REFUSAL),
                "timing": {
                    "asr": round(t1 - t0, 3),
                    "audio_prepare": round(asr_stats["audio_prepare"], 3),
                    "transcribe": round(asr_stats["transcribe"], 3),
                    "normalize": round(t2 - t1, 3),
                    "topic_check": round(t3 - topic_check_start, 3),
                    "retrieve": round(t5 - t4, 3),
                    "retrieve_embed": round(retrieve_stats["embed"], 3),
                    "retrieve_search": round(retrieve_stats["search"], 3),
                    "retrieve_postprocess": round(retrieve_stats["postprocess"], 3),
                    "retrieve_context_build": round(retrieve_stats["context_build"], 3),
                    "generate": round(t7 - t6, 3),
                    "total": round(t7 - t0, 3),
                },
            }
        result = {
            "transcript": transcript,
            "query": q,
            "intent": "info",
            "client_command": None,
            "answer": answer,
            "context_used": (answer != REFUSAL),
            "timing": {
                "asr": round(t1 - t0, 3),
                "audio_prepare": round(asr_stats["audio_prepare"], 3),
                "transcribe": round(asr_stats["transcribe"], 3),
                "normalize": round(t2 - t1, 3),
                "topic_check": round(t3 - topic_check_start, 3),
                "retrieve": round(t5 - t4, 3),
                "retrieve_embed": round(retrieve_stats["embed"], 3),
                "retrieve_search": round(retrieve_stats["search"], 3),
                "retrieve_postprocess": round(retrieve_stats["postprocess"], 3),
                "retrieve_context_build": round(retrieve_stats["context_build"], 3),
                "generate": round(t7 - t6, 3),
                "total": round(t7 - t0, 3),
            },
        }
        if meta_request:
            result["meta_request"] = meta_request
        return result
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
    current_spacecraft = resolve_current_spacecraft(
        current_model_id=req.current_model_id,
        current_spacecraft=req.current_spacecraft,
    )
    if current_spacecraft:
        q = inject_spacecraft_context(q, current_spacecraft)

    matched_command, command_debug = resolve_client_command(q)
    if matched_command:
        if matched_command["intent"] == "hybrid":
            result = command_response(
                q,
                matched_command["type"],
                "Запрос содержит и команду управления, и запрос на получение информации.",
                intent="hybrid",
            )
            result["knowledge_text"] = matched_command.get("knowledge_text")
            result["meta_request"] = matched_command.get("meta_request")
        elif matched_command["intent"] == "action":
            result = command_response(q, matched_command["type"], matched_command["answer"])
        elif matched_command["intent"] == "unknown_command":
            result = unknown_command_response(q)
        else:
            result = {
                "query": q,
                "intent": "info",
                "client_command": None,
                "answer": "Запрос не классифицирован как команда управления.",
                "context_used": False,
                "knowledge_text": matched_command.get("knowledge_text"),
                "meta_request": matched_command.get("meta_request"),
            }
    else:
        result = {
            "query": q,
            "intent": "info",
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
