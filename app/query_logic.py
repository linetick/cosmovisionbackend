import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

import torch

from kb_aliases import apply_alias_map

from .config import (
    ANSWER_MARKERS,
    LLM_BACKEND,
    LLM_MAX_INPUT_TOKENS,
    LLM_MAX_NEW_TOKENS,
    QUERY_STOPWORDS,
    RAG_MAX_CONTEXT_CHARS,
    RELEVANCE_GATE_ENABLED,
    REFUSAL,
    REFUSAL_PATTERNS,
    SHORT_LLM_MAX_NEW_TOKENS,
    SHORT_RAG_MAX_CONTEXT_CHARS,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    VLLM_MODEL,
    VLLM_TIMEOUT,
    YANDEX_API_KEY,
    YANDEX_BASE_URL,
    YANDEX_LOG_USAGE,
    YANDEX_MODEL,
    YANDEX_PROJECT_ID,
    YANDEX_PROMPT_ID,
    YANDEX_ROUTER_PROMPT_ID,
    YANDEX_TIMEOUT,
    WEAK_QUERY_TOKENS,
    WHISPER_BEAM_SIZE,
    WHISPER_VAD_FILTER,
)
from .runtime import (
    AUTO_QUERY_ALIASES,
    COLLECTION_COUNT,
    collection,
    device,
    embedder,
    get_whisper_model,
    model,
    tokenizer,
)

COMMAND_PHRASES = {
    "start_rotation": (
        "вращай", "повращай", "крути", "покрути", "начни вращение",
        "запусти вращение", "вращение", "верти", "заверти",
        "поверни", "повернуть", "разверни", "развернуть",
        "поверни спутник", "повернуть спутник",
        "поверни текущий спутник", "повернуть текущий спутник",
    ),
    "stop_rotation": (
        "останови вращение", "останови спутник", "не вращай", "хватит вращать",
        "перестань вращать", "стоп", "остановись", "остановка",
    ),
    "increase_scale": (
        "увеличь", "увеличить", "приблизь", "сделай больше",
        "увеличь спутник", "увеличь модель", "приблизи",
    ),
    "decrease_scale": (
        "уменьши", "уменьшить", "отдали", "сделай меньше",
        "уменьши спутник", "уменьши модель",
    ),
    "reset_view": (
        "сбрось", "сброс", "верни обратно", "исходный вид",
        "верни как было", "сбрось спутник", "сбрось модель",
    ),
    "play_animation": (
        "запусти анимацию", "включи анимацию", "анимация", "оживи спутник",
        "анимируй спутник", "запусти движение", "включи движение",
    ),
}

COMMAND_ANSWERS = {
    "start_rotation": "Запускаю вращение спутника.",
    "stop_rotation": "Останавливаю вращение спутника.",
    "increase_scale": "Увеличиваю модель.",
    "decrease_scale": "Уменьшаю модель.",
    "reset_view": "Возвращаю исходный вид модели.",
    "play_animation": "Запускаю анимацию спутника.",
}

COMMAND_LIKE_MARKERS = (
    "вращ", "крут", "верт", "поверн", "разверн", "останов", "стоп",
    "увелич", "приблиз", "уменьш", "отдал",
    "сброс", "верни обратно", "исходный вид",
    "анимац", "анимир", "оживи", "движени",
)
COMMAND_TYPES = tuple(COMMAND_PHRASES.keys())

COMMAND_MARKER_GROUPS = (
    ("stop_rotation", ("останов", "перестан", "не вращ", "хватит вращ", "стоп")),
    ("play_animation", ("анимац", "анимир", "ожив", "включи движение", "запусти движение")),
    ("increase_scale", ("увелич", "приблиз", "больше")),
    ("decrease_scale", ("уменьш", "отдал", "меньше")),
    ("reset_view", ("сброс", "исходный вид", "верни обратно", "как было")),
    ("start_rotation", ("вращ", "крут", "верт", "поверн", "разверн")),
)

KNOWLEDGE_REQUEST_MARKERS = (
    "расскажи", "объясни", "что такое", "что это", "что за",
    "для чего", "зачем", "как устро", "какой", "какая", "какие",
    "что делает", "что умеет", "информация", "опиши",
)

KNOWLEDGE_REQUEST_STEMS = (
    "расскаж", "объясн", "что такое", "что это", "что за",
    "для чего", "зачем", "как устро", "какой", "какая", "какие",
    "что делает", "что умеет", "информац", "опиш",
)

META_REQUEST_MARKERS = (
    "привет", "здравствуйте", "кто вы", "кто ты", "чем вы занимаетесь",
    "чем ты занимаешься", "что вы умеете", "что ты умеешь",
    "как вас зовут", "как тебя зовут", "что вообще происходит",
)

GENERIC_CONCEPT_MARKERS = (
    "в целом",
    "вообще",
    "в общем",
)

GENERIC_CONCEPT_TERMS = (
    "спутник",
    "космический аппарат",
    "аппарат",
    "орбита",
    "антенна",
    "солнечная панель",
    "ракета",
)


def normalize_query(q: str) -> str:
    q = (q or "").strip()
    q = re.sub(r"\s+", " ", q)
    q = apply_alias_map(q, AUTO_QUERY_ALIASES)
    q = re.sub(r"метреор", "метеор", q, flags=re.IGNORECASE)
    q = re.sub(r"\bтакй\b", "такой", q, flags=re.IGNORECASE)
    q = re.sub(r"\bтакя\b", "такая", q, flags=re.IGNORECASE)
    q = re.sub(r"\bтаке\b", "такое", q, flags=re.IGNORECASE)
    return q


def inject_spacecraft_context(query: str, current_spacecraft: str | None = None) -> str:
    q = (query or "").strip()
    spacecraft = (current_spacecraft or "").strip()
    if not q or not spacecraft:
        return q

    lowered = q.lower()
    if spacecraft.lower() in lowered:
        return q

    ambiguous_markers = (
        "о спутнике",
        "про спутник",
        "про спутнике",
        "о нем",
        "о нём",
        "про него",
        "расскажи о спутнике",
        "расскажи про спутник",
        "объясни про спутник",
        "расскажи о нем",
        "расскажи о нём",
        "что это за спутник",
    )
    if any(marker in lowered for marker in ambiguous_markers):
        return f"{q} {spacecraft}"

    return q


def extract_knowledge_query_from_compound(
    query: str,
    command_type: str,
    current_spacecraft: str | None = None,
) -> str:
    q = (query or "").strip()
    if not q:
        return q

    current_spacecraft = (current_spacecraft or "").strip() or None
    lowered = q.lower()

    segments = [
        part.strip(" ,.")
        for part in re.split(r"\b(?:и|а|затем|потом)\b", q, flags=re.IGNORECASE)
        if part.strip(" ,.")
    ]

    for segment in segments:
        segment_lower = segment.lower()
        if any(marker in segment_lower for marker in KNOWLEDGE_REQUEST_MARKERS):
            extracted = segment.strip()
            if current_spacecraft:
                extracted = inject_spacecraft_context(extracted, current_spacecraft)
            return extracted

    command_phrases = COMMAND_PHRASES.get(command_type, ())
    cleaned = q
    for phrase in sorted(command_phrases, key=len, reverse=True):
        cleaned = re.sub(
            rf"\b{re.escape(phrase)}\b",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )

    cleaned = re.sub(
        r"\b(можешь|пожалуйста|давай|ну|текущий|текущего|текущую|текущее)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.")

    if current_spacecraft:
        cleaned = inject_spacecraft_context(cleaned, current_spacecraft)

    if cleaned and any(marker in cleaned.lower() for marker in KNOWLEDGE_REQUEST_MARKERS):
        return cleaned

    if current_spacecraft:
        return f"расскажи о спутнике {current_spacecraft}"

    return "расскажи о спутнике"


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


def _matches_any(text: str, phrases: tuple[str, ...]) -> bool:
    normalized_text = re.sub(r"\s+", " ", (text or "").lower()).strip()
    normalized = f" {normalized_text} "
    for phrase in phrases:
        candidate_text = re.sub(r"\s+", " ", phrase.lower()).strip()
        candidate = f" {candidate_text} "
        if candidate in normalized:
            return True
    return False


def detect_client_command(query: str) -> dict | None:
    for command_type, phrases in COMMAND_PHRASES.items():
        if _matches_any(query, phrases):
            return {
                "type": command_type,
                "answer": COMMAND_ANSWERS[command_type],
            }
    return None


def looks_like_client_command(query: str) -> bool:
    lowered = (query or "").lower()
    return any(marker in lowered for marker in COMMAND_LIKE_MARKERS)


def infer_client_command_from_markers(query: str) -> str | None:
    lowered = (query or "").lower()
    for command_type, markers in COMMAND_MARKER_GROUPS:
        if any(marker in lowered for marker in markers):
            return command_type
    return None


def looks_like_knowledge_request(query: str) -> bool:
    lowered = (query or "").lower()
    return any(marker in lowered for marker in KNOWLEDGE_REQUEST_STEMS)


def looks_like_meta_request(query: str) -> bool:
    lowered = (query or "").lower()
    return any(marker in lowered for marker in META_REQUEST_MARKERS)


def is_general_concept_query(query: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").lower()).strip()
    if not q or not is_definitional(q):
        return False
    if extract_designation_tokens(q):
        return False
    if any(marker in q for marker in GENERIC_CONCEPT_MARKERS):
        return True
    return any(term in q for term in GENERIC_CONCEPT_TERMS)


def split_meta_and_knowledge_request(query: str) -> tuple[str | None, str | None]:
    raw_segments = re.split(
        r"(?<=[.!?])\s+|\s*,\s*|\s+\b(?:и|а|а еще|а ещё|также|так же)\b\s+",
        (query or "").strip(),
        flags=re.IGNORECASE,
    )
    meta_parts: list[str] = []
    knowledge_parts: list[str] = []

    for segment in raw_segments:
        cleaned = segment.strip(" ,.")
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if looks_like_meta_request(lowered):
            meta_parts.append(cleaned)
            continue
        if looks_like_knowledge_request(lowered) or is_off_topic(cleaned) is False:
            knowledge_parts.append(cleaned)
            continue
        if "спутник" in lowered or "космос" in lowered or "метеор" in lowered:
            knowledge_parts.append(cleaned)

    meta_request = normalize_query(" ".join(meta_parts)) if meta_parts else None
    knowledge_text = normalize_query(" ".join(knowledge_parts)) if knowledge_parts else None
    return (meta_request or None, knowledge_text or None)


def has_command_markers(query: str) -> bool:
    return detect_client_command(query) is not None or looks_like_client_command(query)


def extract_designation_tokens(text: str) -> set[str]:
    normalized = (text or "").lower().replace("ё", "е")
    matches = re.findall(r"[0-9a-zа-я]+(?:[-–—][0-9a-zа-я]+)+", normalized, flags=re.IGNORECASE)
    designations: set[str] = set()
    for match in matches:
        candidate = re.sub(r"[-–—]+", "-", match.strip("-"))
        parts = [part for part in candidate.split("-") if part]
        if len(parts) < 2:
            continue
        if any(any(ch.isdigit() for ch in part) for part in parts) or any(len(part) <= 2 for part in parts[1:]):
            designations.add(candidate)
    return designations


def is_multi_entity_query(query: str) -> bool:
    return len(extract_designation_tokens(query)) >= 2


def _extract_json_object(text: str) -> dict | None:
    if not text:
        return None

    candidate = text.strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def classify_query_with_llm(query: str) -> dict | None:
    if LLM_BACKEND == "yandex" and YANDEX_ROUTER_PROMPT_ID:
        messages = [
            {"role": "user", "content": query},
        ]
        raw = run_chat_generation(
            messages,
            max_new_tokens=SHORT_LLM_MAX_NEW_TOKENS,
            prompt_id=YANDEX_ROUTER_PROMPT_ID,
            usage_label="router",
        )
        parsed = _extract_json_object(raw)
        if not parsed:
            return None

        intent = (parsed.get("intent") or "").strip()
        meta_request = normalize_query((parsed.get("meta_request") or "").strip())
        if intent == "info":
            knowledge_text = (parsed.get("knowledge_text") or "").strip()
            route = {
                "intent": "info",
            }
            if knowledge_text:
                route["knowledge_text"] = knowledge_text
            elif not meta_request:
                route["knowledge_text"] = query
            if meta_request:
                route["meta_request"] = meta_request
            return route
        if intent == "unknown_command":
            return {"intent": "unknown_command"}
        if intent not in {"action", "hybrid"}:
            return None

        command_type = (parsed.get("command_type") or "").strip()
        if command_type not in COMMAND_ANSWERS:
            return None

        route = {
            "intent": intent,
            "command_type": command_type,
        }
        if intent == "hybrid":
            knowledge_text = (parsed.get("knowledge_text") or "").strip()
            route["knowledge_text"] = knowledge_text or ""
        if meta_request:
            route["meta_request"] = meta_request
        return route

    allowed = ", ".join(COMMAND_TYPES)
    system = (
        "Ты маршрутизатор запросов для AR-приложения про космические аппараты.\n"
        "Нужно отнести запрос ровно к одной категории и вернуть только JSON.\n"
        "Считай, что пользователь часто пишет разговорно, с лишними словами, вежливыми оборотами и смешивает несколько действий в одном предложении.\n"
        "Твоя задача не ответить пользователю, а только разобрать запрос по смыслу.\n"
        "Нужно отнести запрос ровно к одной категории:\n"
        "1. action — если пользователь хочет управлять 3D-моделью.\n"
        "2. info — если пользователь задаёт вопрос по знаниям о космическом аппарате.\n"
        "3. hybrid — если в одном запросе одновременно есть команда управления и просьба рассказать/объяснить что-то.\n"
        "4. unknown_command — если пользователь, вероятно, хочет управлять моделью, но команда не входит в допустимый список.\n"
        f"Допустимые action-команды: {allowed}.\n"
        "Смысл команд:\n"
        "- start_rotation: начать вращение, крутить, вертеть, повернуть, развернуть спутник или модель.\n"
        "- stop_rotation: остановить вращение, прекратить кручение, перестать вращать, остановить спутник.\n"
        "- increase_scale: увеличить, приблизить, сделать больше модель или спутник.\n"
        "- decrease_scale: уменьшить, отдалить, сделать меньше модель или спутник.\n"
        "- reset_view: сбросить вид, вернуть обратно, вернуть как было, исходный вид.\n"
        "- play_animation: запустить анимацию, включить движение, оживить, анимировать спутник.\n"
        "Учитывай разговорные формулировки, склонения слов, падежи и синонимы.\n"
        "Очень важные правила выбора intent:\n"
        "- Если в запросе есть понятная команда из допустимого списка и больше ничего, выбирай action.\n"
        "- Если в запросе есть понятная команда из допустимого списка и одновременно просьба рассказать, объяснить, описать, что это такое или для чего это нужно, выбирай hybrid.\n"
        "- Если в запросе есть слова 'можешь', 'пожалуйста', 'и', 'а ещё', 'текущий', название спутника или другие лишние слова, это не меняет intent.\n"
        "- Не выбирай unknown_command, если по смыслу запрос можно отнести к одной из допустимых команд.\n"
        "- unknown_command выбирай только тогда, когда пользователь явно хочет управлять моделью, но команда семантически не соответствует ни одной допустимой команде.\n"
        "- Если запрос смешанный, в knowledge_text оставляй только информационную часть без команды управления.\n"
        "Если intent = info, обязательно верни поле knowledge_text.\n"
        "Если intent = hybrid, обязательно верни поле knowledge_text и отдели из исходной фразы только информационную часть, без команды управления.\n"
        "knowledge_text должен содержать только ту часть запроса, которую нужно отправить в retrieval/RAG.\n"
        "Верни только JSON без пояснений.\n"
        "Форматы ответа:\n"
        '{"intent":"action","command_type":"start_rotation"}\n'
        '{"intent":"info","knowledge_text":"что такое спутник"}\n'
        '{"intent":"hybrid","command_type":"play_animation","knowledge_text":"расскажи о спутнике"}\n'
        '{"intent":"unknown_command"}'
    )
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": "Покрути спутник",
        },
        {
            "role": "assistant",
            "content": '{"intent":"action","command_type":"start_rotation"}',
        },
        {
            "role": "user",
            "content": "Останови вращение текущего спутника",
        },
        {
            "role": "assistant",
            "content": '{"intent":"action","command_type":"stop_rotation"}',
        },
        {
            "role": "user",
            "content": "Сделай модель больше",
        },
        {
            "role": "assistant",
            "content": '{"intent":"action","command_type":"increase_scale"}',
        },
        {
            "role": "user",
            "content": "Верни как было",
        },
        {
            "role": "assistant",
            "content": '{"intent":"action","command_type":"reset_view"}',
        },
        {
            "role": "user",
            "content": "Запусти анимацию спутника",
        },
        {
            "role": "assistant",
            "content": '{"intent":"action","command_type":"play_animation"}',
        },
        {
            "role": "user",
            "content": "Можешь запустить анимацию текущего спутника",
        },
        {
            "role": "assistant",
            "content": '{"intent":"action","command_type":"play_animation"}',
        },
        {
            "role": "user",
            "content": "Запусти анимацию и расскажи о спутнике",
        },
        {
            "role": "assistant",
            "content": '{"intent":"hybrid","command_type":"play_animation","knowledge_text":"расскажи о спутнике"}',
        },
        {
            "role": "user",
            "content": "Можешь запустить анимацию и рассказать о спутнике Метеор-М",
        },
        {
            "role": "assistant",
            "content": '{"intent":"hybrid","command_type":"play_animation","knowledge_text":"расскажи о спутнике Метеор-М"}',
        },
        {
            "role": "user",
            "content": "Останови вращение и объясни, что это за антенна",
        },
        {
            "role": "assistant",
            "content": '{"intent":"hybrid","command_type":"stop_rotation","knowledge_text":"объясни, что это за антенна"}',
        },
        {
            "role": "user",
            "content": "Поверни модель и объясни, что это за антенна",
        },
        {
            "role": "assistant",
            "content": '{"intent":"hybrid","command_type":"start_rotation","knowledge_text":"объясни, что это за антенна"}',
        },
        {
            "role": "user",
            "content": "Что такое спутник?",
        },
        {
            "role": "assistant",
            "content": '{"intent":"info","knowledge_text":"что такое спутник"}',
        },
        {
            "role": "user",
            "content": "Расскажи о Метеор-М",
        },
        {
            "role": "assistant",
            "content": '{"intent":"info","knowledge_text":"расскажи о Метеор-М"}',
        },
        {
            "role": "user",
            "content": "Сделай что-нибудь со спутником",
        },
        {
            "role": "assistant",
            "content": '{"intent":"unknown_command"}',
        },
        {
            "role": "user",
            "content": query,
        },
    ]

    raw = run_chat_generation(
        messages,
        max_new_tokens=SHORT_LLM_MAX_NEW_TOKENS,
        prompt_id=YANDEX_ROUTER_PROMPT_ID,
        usage_label="router",
    )

    parsed = _extract_json_object(raw)
    if not parsed:
        return None

    intent = (parsed.get("intent") or "").strip()
    meta_request = normalize_query((parsed.get("meta_request") or "").strip())
    if intent == "info":
        knowledge_text = (parsed.get("knowledge_text") or "").strip()
        route = {
            "intent": "info",
        }
        if knowledge_text:
            route["knowledge_text"] = knowledge_text
        elif not meta_request:
            route["knowledge_text"] = query
        if meta_request:
            route["meta_request"] = meta_request
        return route
    if intent == "unknown_command":
        return {"intent": "unknown_command"}
    if intent not in {"action", "hybrid"}:
        return None

    command_type = (parsed.get("command_type") or "").strip()
    if command_type not in COMMAND_ANSWERS:
        return None

    route = {
        "intent": intent,
        "command_type": command_type,
    }
    if intent == "hybrid":
        knowledge_text = (parsed.get("knowledge_text") or "").strip()
        route["knowledge_text"] = knowledge_text or ""
    if meta_request:
        route["meta_request"] = meta_request
    return route


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
    query_tokens = {tok for tok in tokenize_for_match(query) if tok not in QUERY_STOPWORDS}
    strong_tokens = {tok for tok in query_tokens if tok not in WEAK_QUERY_TOKENS}
    return query_tokens, strong_tokens


def assess_context_relevance(query: str, context: str) -> dict:
    query_tokens, strong_tokens = query_token_sets(query)
    context_tokens = set(tokenize_for_match(context))
    query_designations = extract_designation_tokens(query)
    context_designations = extract_designation_tokens(context)
    total_overlap = len(query_tokens & context_tokens)
    strong_overlap = len(strong_tokens & context_tokens)
    return {
        "query_tokens": len(query_tokens),
        "strong_tokens": len(strong_tokens),
        "total_overlap": total_overlap,
        "strong_overlap": strong_overlap,
        "query_designations": query_designations,
        "context_designations": context_designations,
        "designation_overlap": len(query_designations & context_designations),
        "has_answer_marker": any(marker in context.lower() for marker in ANSWER_MARKERS),
    }


def has_sufficient_context_relevance(query: str, context: str) -> tuple[bool, dict]:
    metrics = assess_context_relevance(query, context)
    if metrics["query_tokens"] == 0:
        return False, metrics
    if is_general_concept_query(query) and metrics["context_designations"]:
        return False, metrics
    if metrics["query_designations"] and metrics["designation_overlap"] == 0:
        return False, metrics
    if metrics["strong_overlap"] > 0:
        return True, metrics
    if metrics["total_overlap"] >= 2:
        return True, metrics
    if metrics["total_overlap"] >= 1 and metrics["has_answer_marker"]:
        return True, metrics
    return False, metrics


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

    generic_query = is_general_concept_query(query)
    best_text = None
    best_score = -1
    best_strong_overlap = 0
    best_total_overlap = 0

    for hit in hits:
        doc = (hit.get("doc") or "").strip()
        meta = hit.get("meta") or {}
        section = meta.get("section") or ""
        section_tokens = set(tokenize_for_match(section))
        if generic_query and extract_designation_tokens(doc):
            continue

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


def retrieve_hits(query: str, initial_n: int = 3, max_n: int = 8) -> tuple[list[dict], dict]:
    stats = {
        "embed": 0.0,
        "search": 0.0,
        "postprocess": 0.0,
        "context_build": 0.0,
        "total": 0.0,
    }
    if COLLECTION_COUNT <= 0:
        return [], stats

    t_embed = time.perf_counter()
    q_emb = embedder.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    stats["embed"] = time.perf_counter() - t_embed

    n = min(max(initial_n, 1), COLLECTION_COUNT)
    limit = min(max_n, COLLECTION_COUNT)
    best_hits: list[dict] = []
    best_chars = 0

    while True:
        t_search = time.perf_counter()
        try:
            results = collection.query(
                query_embeddings=q_emb,
                n_results=n,
                include=["documents", "distances", "metadatas"],
            )
        except TypeError:
            results = collection.query(query_embeddings=q_emb, n_results=n)
        stats["search"] += time.perf_counter() - t_search

        t_post = time.perf_counter()
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
            stats["postprocess"] += time.perf_counter() - t_post
            stats["total"] = stats["embed"] + stats["search"] + stats["postprocess"]
            return best_hits, stats

        stats["postprocess"] += time.perf_counter() - t_post
        n = min(n + 2, limit)


def build_context_from_hits(hits: list[dict], max_chars: int = RAG_MAX_CONTEXT_CHARS) -> str:
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


def retrieve_context(query: str, initial_n: int = 3, max_n: int = 8) -> tuple[str, list[dict], dict]:
    hits, stats = retrieve_hits(query, initial_n=initial_n, max_n=max_n)
    t_context = time.perf_counter()
    context = build_context_from_hits(hits)
    stats["context_build"] = time.perf_counter() - t_context
    stats["total"] = stats["embed"] + stats["search"] + stats["postprocess"] + stats["context_build"]
    return context, hits, stats


def extract_definition_from_context(query: str, context: str) -> str | None:
    q = query.lower()
    c = context
    if is_general_concept_query(query) and extract_designation_tokens(context):
        return None
    if "метеор" in q:
        match = re.search(r"(Метеор[-–— ]?М\s*—\s*это[^\n\.]*[\.]?)", c, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    match = re.search(r"^(.{0,80}?—\s*это[^\n\.]*[\.]?)", c, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def is_definitional(q: str) -> bool:
    ql = q.lower()
    if any(x in ql for x in ["что такое", "что значит", "определи", "дать определение"]):
        return True
    return re.search(r"\b(кто|что)\s+так[а-яё]*\b", ql) is not None


def wants_brief_answer(q: str) -> bool:
    ql = q.lower()
    markers = [
        "кратко", "коротко", "вкратце", "одним предложением",
        "в одном предложении", "в двух словах", "краткое описание",
    ]
    return any(marker in ql for marker in markers)


def squeeze_to_one_sentence(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return normalized

    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    if not parts:
        return normalized
    if len(parts) >= 2 and len(parts[0].split()) <= 3 and len(parts[0]) <= 30:
        header = parts[0].rstrip(".!?")
        body = parts[1].rstrip()
        return f"{header}: {body}"
    return parts[0]


def generate_with_vllm(messages: list[dict], max_new_tokens: int) -> str:
    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0,
    }
    request = urllib.request.Request(
        f"{VLLM_BASE_URL}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {VLLM_API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=VLLM_TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"vLLM request failed: {exc}") from exc

    choices = body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return (message.get("content") or "").strip()


def _normalize_yandex_input(messages: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for message in messages:
        role = (message.get("role") or "user").strip() or "user"
        content = message.get("content") or ""
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    text = (item.get("text") or "").strip()
                    if text:
                        text_parts.append(text)
                elif isinstance(item, str):
                    text = item.strip()
                    if text:
                        text_parts.append(text)
            content = "\n".join(text_parts).strip()
        else:
            content = str(content).strip()
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _extract_yandex_output_text(body: dict) -> str:
    output = body.get("output") or []
    for item in output:
        if item.get("type") != "message":
            continue
        contents = item.get("content") or []
        text_parts = []
        for part in contents:
            if part.get("type") == "output_text":
                text = (part.get("text") or "").strip()
                if text:
                    text_parts.append(text)
        if text_parts:
            return "\n".join(text_parts).strip()
    return ""


def _extract_yandex_usage(body: dict) -> dict:
    usage = body.get("usage") or {}
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    prompt_info = body.get("prompt") or {}
    return {
        "status": body.get("status") or "",
        "model": body.get("model") or "",
        "prompt_id": prompt_info.get("id") or "",
        "input_tokens": usage.get("input_tokens") or 0,
        "cached_tokens": input_details.get("cached_tokens") or 0,
        "tool_tokens": input_details.get("tool_tokens") or 0,
        "output_tokens": usage.get("output_tokens") or 0,
        "reasoning_tokens": output_details.get("reasoning_tokens") or 0,
        "total_tokens": usage.get("total_tokens") or 0,
    }


def _log_yandex_usage(label: str, body: dict) -> None:
    if not YANDEX_LOG_USAGE:
        return
    usage = _extract_yandex_usage(body)
    model_name = usage["model"] or f"gpt://{YANDEX_PROJECT_ID}/{YANDEX_MODEL}"
    prompt_part = f" prompt={usage['prompt_id']}" if usage["prompt_id"] else ""
    print(
        f"[Yandex usage][{label}]"
        f" input={usage['input_tokens']}"
        f" cached={usage['cached_tokens']}"
        f" tool={usage['tool_tokens']}"
        f" output={usage['output_tokens']}"
        f" reasoning={usage['reasoning_tokens']}"
        f" total={usage['total_tokens']}"
        f" status={usage['status']}{prompt_part}"
        f" model={model_name}"
    )


def generate_with_yandex(
    messages: list[dict],
    max_new_tokens: int,
    prompt_id: str | None = None,
    usage_label: str = "default",
) -> str:
    if not YANDEX_API_KEY:
        raise RuntimeError("YANDEX_API_KEY не задан.")
    if not YANDEX_PROJECT_ID:
        raise RuntimeError("YANDEX_PROJECT_ID не задан.")

    effective_prompt_id = prompt_id if prompt_id is not None else YANDEX_PROMPT_ID

    payload: dict = {
        "input": _normalize_yandex_input(messages),
        "temperature": 0,
        "max_output_tokens": max_new_tokens,
    }

    if effective_prompt_id:
        payload["prompt"] = {"id": effective_prompt_id}
    else:
        payload["model"] = f"gpt://{YANDEX_PROJECT_ID}/{YANDEX_MODEL}"

    request = urllib.request.Request(
        f"{YANDEX_BASE_URL}/responses",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {YANDEX_API_KEY}",
            "OpenAI-Project": YANDEX_PROJECT_ID,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=YANDEX_TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Yandex Cloud HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"Yandex Cloud request failed: {exc}") from exc

    _log_yandex_usage(usage_label, body)
    return _extract_yandex_output_text(body)


def run_chat_generation(
    messages: list[dict],
    max_new_tokens: int,
    prompt_id: str | None = None,
    usage_label: str = "default",
) -> str:
    if LLM_BACKEND == "vllm":
        return generate_with_vllm(messages, max_new_tokens)
    if LLM_BACKEND == "yandex":
        return generate_with_yandex(messages, max_new_tokens, prompt_id=prompt_id, usage_label=usage_label)

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=LLM_MAX_INPUT_TOKENS,
    ).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs.input_ids.shape[1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def generate_answer_llm_only(query: str) -> str:
    brief_answer = wants_brief_answer(query)
    max_new_tokens = SHORT_LLM_MAX_NEW_TOKENS if brief_answer else LLM_MAX_NEW_TOKENS

    user_content = query
    if brief_answer:
        user_content += "\n\nОтветь одним коротким предложением."

    if LLM_BACKEND == "yandex" and YANDEX_PROMPT_ID:
        messages = [
            {"role": "user", "content": user_content},
        ]
    else:
        system = (
            "Ты ИИ-ассистент по космонавтике.\n"
            "Отвечай по существу и без лишней воды.\n"
            "Если пользователь просит кратко, отвечай кратко."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    text = run_chat_generation(messages, max_new_tokens, usage_label="llm_only")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^ответ:\s*", "", text, flags=re.IGNORECASE)
    if not text or len(text) < 3:
        return REFUSAL
    return squeeze_to_one_sentence(text) if brief_answer else text


def generate_answer_fallback(query: str, meta_request: str | None = None) -> str:
    meta_request = (meta_request or "").strip() or None
    max_new_tokens = max(SHORT_LLM_MAX_NEW_TOKENS, 48)

    user_parts = [
        "РЕЖИМ FALLBACK.\nВ базе знаний по этому вопросу нет информации.",
    ]
    if meta_request:
        user_parts.append(f"ВОПРОС О СИСТЕМЕ:\n{meta_request}")
    user_parts.append(f"ВОПРОС О КОСМИЧЕСКОМ АППАРАТЕ:\n{query}")
    user_parts.append(
        "Кратко ответь по общим знаниям. "
        "Если у тебя есть доступ к интернет-поиску или внешним инструментам, можешь использовать их. "
        "Обязательно явно скажи, что в базе знаний нет информации, а в конце добавь: "
        "«Ответ может быть неточным.» "
        "Ответ должен быть коротким и понятным."
    )
    user_content = "\n\n".join(user_parts)

    if LLM_BACKEND == "yandex" and YANDEX_PROMPT_ID:
        messages = [
            {"role": "user", "content": user_content},
        ]
    else:
        system = (
            "Ты ИИ-ассистент по космонавтике.\n"
            "Если база знаний не дала ответа, разрешено кратко отвечать по общим знаниям.\n"
            "Если у тебя есть доступ к внешним инструментам, можно их использовать.\n"
            "Нужно явно сказать, что в базе знаний нет информации, и предупредить, что ответ может быть неточным."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    text = run_chat_generation(messages, max_new_tokens, usage_label="rag_fallback")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^ответ:\s*", "", text, flags=re.IGNORECASE)
    if not text or len(text) < 3:
        return REFUSAL
    return text


def generate_answer_strict(
    query: str,
    context: str,
    hits: list[dict] | None = None,
    meta_request: str | None = None,
) -> str:
    brief_answer = wants_brief_answer(query)
    meta_request = (meta_request or "").strip() or None
    multi_entity = is_multi_entity_query(query)

    if not meta_request and not multi_entity and (is_definitional(query) or brief_answer):
        defin = extract_definition_from_context(query, context)
        if defin:
            return squeeze_to_one_sentence(defin) if brief_answer else defin

    if not meta_request and not multi_entity:
        extractive = find_extractive_answer(query, hits or [])
        if extractive:
            return squeeze_to_one_sentence(extractive) if brief_answer else extractive

    if not context.strip():
        return REFUSAL

    if brief_answer and hits:
        shorter_context = build_context_from_hits(hits, max_chars=SHORT_RAG_MAX_CONTEXT_CHARS)
        if shorter_context:
            context = shorter_context

    if RELEVANCE_GATE_ENABLED:
        relevant, _ = has_sufficient_context_relevance(query, context)
        if not relevant:
            return REFUSAL

    user_parts: list[str] = []
    if meta_request:
        user_parts.append(f"ВОПРОС О СИСТЕМЕ:\n{meta_request}")
    user_parts.append(f"КОНТЕКСТ:\n{context}")
    user_parts.append(f"ВОПРОС О КОСМИЧЕСКОМ АППАРАТЕ:\n{query}")
    user_content = "\n\n".join(user_parts)
    if meta_request:
        user_content += (
            "\n\nЕсли есть вопрос о системе, ответь на него из своей инструкции. "
            "Факты о космическом аппарате бери только из блока КОНТЕКСТ."
        )
    if multi_entity:
        user_content += "\n\nВ вопросе упомянуто несколько космических аппаратов. Если контекст позволяет, кратко ответь по каждому из них."
    if brief_answer:
        if meta_request:
            user_content += "\n\nОтветь очень кратко одним предложением, сначала про систему, затем про аппарат."
        else:
            user_content += "\n\nОтветь одним коротким предложением без вводных слов."

    if LLM_BACKEND == "yandex" and YANDEX_PROMPT_ID:
        messages = [
            {"role": "user", "content": user_content},
        ]
    else:
        system = (
            "Ты — ИИ-ассистент по космонавтике с RAG.\n"
            "Отвечай только по фрагментам базы знаний из блока КОНТЕКСТ.\n"
            "Нельзя добавлять факты не из контекста.\n"
            "Если в запросе есть вопрос о системе, отвечай на него из своей инструкции, а не из контекста.\n"
            f"Если в контексте нет ответа, верни ровно: {REFUSAL}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    max_new_tokens = SHORT_LLM_MAX_NEW_TOKENS if brief_answer else LLM_MAX_NEW_TOKENS
    text = run_chat_generation(messages, max_new_tokens, usage_label="rag_strict")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^ответ:\s*", "", text, flags=re.IGNORECASE)
    if not text or len(text) < 3:
        return REFUSAL
    if is_refusal_like(text):
        return REFUSAL
    return squeeze_to_one_sentence(text) if brief_answer else text


def generate_answer_compact(
    query: str,
    context: str,
    hits: list[dict] | None = None,
    meta_request: str | None = None,
) -> str:
    meta_request = (meta_request or "").strip() or None
    multi_entity = is_multi_entity_query(query)

    if not meta_request and not multi_entity and is_definitional(query):
        defin = extract_definition_from_context(query, context)
        if defin:
            return squeeze_to_one_sentence(defin)

    if not meta_request and not multi_entity:
        extractive = find_extractive_answer(query, hits or [])
        if extractive:
            return squeeze_to_one_sentence(extractive)

    if not context.strip():
        return REFUSAL

    if hits:
        shorter_context = build_context_from_hits(hits, max_chars=SHORT_RAG_MAX_CONTEXT_CHARS)
        if shorter_context:
            context = shorter_context

    if RELEVANCE_GATE_ENABLED:
        relevant, _ = has_sufficient_context_relevance(query, context)
        if not relevant:
            return REFUSAL

    user_parts: list[str] = []
    if meta_request:
        user_parts.append(f"ВОПРОС О СИСТЕМЕ:\n{meta_request}")
    user_parts.append(f"КОНТЕКСТ:\n{context}")
    user_parts.append(f"ВОПРОС О КОСМИЧЕСКОМ АППАРАТЕ:\n{query}")
    user_content = "\n\n".join(user_parts)
    if meta_request:
        user_content += (
            "\n\nЕсли есть вопрос о системе, ответь на него из своей инструкции. "
            "Факты о космическом аппарате бери только из блока КОНТЕКСТ."
        )
    if multi_entity:
        user_content += "\n\nВ вопросе упомянуто несколько космических аппаратов. Если контекст позволяет, кратко ответь по каждому из них."
    user_content += "\n\nОтветь одним коротким предложением."

    if LLM_BACKEND == "yandex" and YANDEX_PROMPT_ID:
        messages = [
            {"role": "user", "content": user_content},
        ]
    else:
        system = (
            "Ты — ИИ-ассистент по космонавтике с RAG.\n"
            "Отвечай только по фрагментам базы знаний из блока КОНТЕКСТ.\n"
            "Нельзя добавлять факты не из контекста.\n"
            "Если в запросе есть вопрос о системе, отвечай на него из своей инструкции, а не из контекста.\n"
            "Ответ должен быть кратким: одно короткое содержательное предложение без вводных слов.\n"
            f"Если в контексте нет ответа, верни ровно: {REFUSAL}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    text = run_chat_generation(messages, SHORT_LLM_MAX_NEW_TOKENS, usage_label="rag_compact")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^ответ:\s*", "", text, flags=re.IGNORECASE)
    if not text or len(text) < 3:
        return REFUSAL
    if is_refusal_like(text):
        return REFUSAL
    return squeeze_to_one_sentence(text)


def wav_to_16k_mono_wav_bytes(wav_bytes: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fin:
        fin.write(wav_bytes)
        fin.flush()
        in_path = fin.name

    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(out_fd)

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", "16000", "-f", "wav", out_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        with open(out_path, "rb") as file:
            return file.read()
    finally:
        for path in (in_path, out_path):
            try:
                os.remove(path)
            except OSError:
                pass


def transcribe_audio_bytes_detailed(data: bytes, language: str = "ru") -> tuple[str, dict]:
    stats = {
        "audio_prepare": 0.0,
        "transcribe": 0.0,
        "total": 0.0,
    }

    t_prepare = time.perf_counter()
    data = wav_to_16k_mono_wav_bytes(data)
    stats["audio_prepare"] = time.perf_counter() - t_prepare

    t_transcribe = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        segments, _ = get_whisper_model().transcribe(
            tmp.name,
            language=language,
            vad_filter=WHISPER_VAD_FILTER,
            beam_size=WHISPER_BEAM_SIZE,
            condition_on_previous_text=False,
        )
        transcript = " ".join(seg.text for seg in segments).strip()
    stats["transcribe"] = time.perf_counter() - t_transcribe
    stats["total"] = stats["audio_prepare"] + stats["transcribe"]
    return transcript, stats


def transcribe_audio_bytes(data: bytes, language: str = "ru") -> str:
    transcript, _ = transcribe_audio_bytes_detailed(data, language=language)
    return transcript
