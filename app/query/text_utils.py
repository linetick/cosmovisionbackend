import re

from ..config import ANSWER_MARKERS, QUERY_STOPWORDS, REFUSAL_PATTERNS, WEAK_QUERY_TOKENS

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


def is_definitional(q: str) -> bool:
    ql = q.lower()
    if any(x in ql for x in ["что такое", "что значит", "определи", "дать определение"]):
        return True
    return re.search(r"\b(кто|что)\s+так[а-яё]*\b", ql) is not None


def is_general_concept_query(query: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").lower()).strip()
    if not q or not is_definitional(q):
        return False
    if extract_designation_tokens(q):
        return False
    if any(marker in q for marker in GENERIC_CONCEPT_MARKERS):
        return True
    return any(term in q for term in GENERIC_CONCEPT_TERMS)


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
        if body.lower().startswith(header.lower()):
            return body
        return f"{header}: {body}"
    return parts[0]


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
        body = " ".join(lines[1:])
        if not body.lower().startswith(lines[0].lower()):
            candidates.append(f"{lines[0]}. {body}".strip())

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
            # Заголовки секций (< 5 слов без знаков препинания) не используем как ответ
            if len(candidate.split()) < 5 and not re.search(r"[.!?—]", candidate):
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
