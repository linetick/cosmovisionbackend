import json
import re

from kb_aliases import apply_alias_map

from ..config import LLM_BACKEND, SHORT_LLM_MAX_NEW_TOKENS, YANDEX_ROUTER_PROMPT_ID
from ..runtime import AUTO_QUERY_ALIASES
from .llm import run_chat_generation

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
            route: dict = {"intent": "info"}
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

        route = {"intent": intent, "command_type": command_type}
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
        {"role": "user", "content": "Покрути спутник"},
        {"role": "assistant", "content": '{"intent":"action","command_type":"start_rotation"}'},
        {"role": "user", "content": "Останови вращение текущего спутника"},
        {"role": "assistant", "content": '{"intent":"action","command_type":"stop_rotation"}'},
        {"role": "user", "content": "Сделай модель больше"},
        {"role": "assistant", "content": '{"intent":"action","command_type":"increase_scale"}'},
        {"role": "user", "content": "Верни как было"},
        {"role": "assistant", "content": '{"intent":"action","command_type":"reset_view"}'},
        {"role": "user", "content": "Запусти анимацию спутника"},
        {"role": "assistant", "content": '{"intent":"action","command_type":"play_animation"}'},
        {"role": "user", "content": "Можешь запустить анимацию текущего спутника"},
        {"role": "assistant", "content": '{"intent":"action","command_type":"play_animation"}'},
        {"role": "user", "content": "Запусти анимацию и расскажи о спутнике"},
        {"role": "assistant", "content": '{"intent":"hybrid","command_type":"play_animation","knowledge_text":"расскажи о спутнике"}'},
        {"role": "user", "content": "Можешь запустить анимацию и рассказать о спутнике Метеор-М"},
        {"role": "assistant", "content": '{"intent":"hybrid","command_type":"play_animation","knowledge_text":"расскажи о спутнике Метеор-М"}'},
        {"role": "user", "content": "Останови вращение и объясни, что это за антенна"},
        {"role": "assistant", "content": '{"intent":"hybrid","command_type":"stop_rotation","knowledge_text":"объясни, что это за антенна"}'},
        {"role": "user", "content": "Поверни модель и объясни, что это за антенна"},
        {"role": "assistant", "content": '{"intent":"hybrid","command_type":"start_rotation","knowledge_text":"объясни, что это за антенна"}'},
        {"role": "user", "content": "Что такое спутник?"},
        {"role": "assistant", "content": '{"intent":"info","knowledge_text":"что такое спутник"}'},
        {"role": "user", "content": "Расскажи о Метеор-М"},
        {"role": "assistant", "content": '{"intent":"info","knowledge_text":"расскажи о Метеор-М"}'},
        {"role": "user", "content": "Сделай что-нибудь со спутником"},
        {"role": "assistant", "content": '{"intent":"unknown_command"}'},
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
        route = {"intent": "info"}
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

    route = {"intent": intent, "command_type": command_type}
    if intent == "hybrid":
        knowledge_text = (parsed.get("knowledge_text") or "").strip()
        route["knowledge_text"] = knowledge_text or ""
    if meta_request:
        route["meta_request"] = meta_request
    return route
