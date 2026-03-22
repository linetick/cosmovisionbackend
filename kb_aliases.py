import os
import re
from typing import Dict, Iterable


CYR_TO_LAT = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "kh",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}


def transliterate_ru_to_lat(text: str) -> str:
    out = []
    for ch in (text or "").lower():
        out.append(CYR_TO_LAT.get(ch, ch))
    return "".join(out)


def _normalized_words(text: str) -> list[str]:
    prepared = (text or "").replace("№", " n ")
    return [w for w in re.split(r"[^0-9A-Za-zА-Яа-яЁё]+", prepared) if w]


def _phrase_variants(text: str) -> set[str]:
    words = _normalized_words(text)
    if not words:
        return set()

    variants = {
        " ".join(words),
        "-".join(words),
        "".join(words),
    }
    normalized_text = re.sub(r"\s+", " ", (text or "").strip())
    if normalized_text:
        variants.add(normalized_text)
    return {variant.lower() for variant in variants if variant.strip()}


def _iter_markdown_entries(raw_dir: str) -> Iterable[tuple[str, str, str]]:
    if not os.path.isdir(raw_dir):
        return

    for name in sorted(os.listdir(raw_dir)):
        if not name.endswith(".md"):
            continue

        path = os.path.join(raw_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
        except OSError:
            continue

        match = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
        title = match.group(1).strip() if match else ""
        source_stem = os.path.splitext(name)[0]
        yield source_stem, title, content


def build_auto_alias_map(raw_dir: str, stopwords: set[str] | None = None) -> Dict[str, str]:
    stopwords = {word.lower() for word in (stopwords or set())}
    alias_map: Dict[str, str] = {}

    def add_alias(alias: str, replacement: str) -> None:
        alias = re.sub(r"\s+", " ", (alias or "").strip().lower())
        replacement = re.sub(r"\s+", " ", (replacement or "").strip())
        if not alias or not replacement:
            return
        compact_len = len(re.sub(r"[^0-9A-Za-zА-Яа-яЁё]+", "", alias))
        if compact_len < 3:
            return
        alias_map.setdefault(alias, replacement)

    for source_stem, title, content in _iter_markdown_entries(raw_dir):
        canonical = title or source_stem.replace("_", " ").replace("-", " ")

        for alias in _phrase_variants(canonical):
            add_alias(alias, canonical)
        for alias in _phrase_variants(transliterate_ru_to_lat(canonical)):
            add_alias(alias, canonical)
        for alias in _phrase_variants(source_stem):
            add_alias(alias, canonical)

        tokens = re.findall(r"[А-Яа-яЁё-]{4,}", f"{canonical}\n{content}")
        for token in tokens:
            normalized = token.lower().replace("ё", "е").strip("-")
            if not normalized or normalized in stopwords:
                continue
            for alias in _phrase_variants(transliterate_ru_to_lat(normalized)):
                add_alias(alias, normalized)

    return alias_map


def apply_alias_map(text: str, alias_map: Dict[str, str]) -> str:
    result = text or ""
    if not result or not alias_map:
        return result

    aliases = sorted(alias_map.items(), key=lambda item: len(item[0]), reverse=True)
    for alias, replacement in aliases:
        pattern = rf"(?<![0-9A-Za-zА-Яа-яЁё]){re.escape(alias)}(?![0-9A-Za-zА-Яа-яЁё])"
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", result).strip()
