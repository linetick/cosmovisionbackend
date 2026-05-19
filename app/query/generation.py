import re

from ..config import (
    LLM_BACKEND,
    LLM_MAX_NEW_TOKENS,
    REFUSAL,
    RELEVANCE_GATE_ENABLED,
    SHORT_LLM_MAX_NEW_TOKENS,
    SHORT_RAG_MAX_CONTEXT_CHARS,
    YANDEX_PROMPT_ID,
)
from .llm import run_chat_generation
from .retrieval import build_context_from_hits
from .text_utils import (
    find_extractive_answer,
    has_sufficient_context_relevance,
    is_definitional,
    is_multi_entity_query,
    is_refusal_like,
    squeeze_to_one_sentence,
    wants_brief_answer,
)


def extract_definition_from_context(query: str, context: str) -> str | None:
    q = query.lower()
    c = context
    from .text_utils import extract_designation_tokens
    if extract_designation_tokens(context) and not any(k in q for k in ["метеор"]):
        pass
    if "метеор" in q:
        match = re.search(r"(Метеор[-–— ]?М\s*—\s*это[^\n\.]*[\.]?)", c, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    match = re.search(r"^(.{0,80}?—\s*это[^\n\.]*[\.]?)", c, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


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
