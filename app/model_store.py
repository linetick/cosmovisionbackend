import json
import os
import re
from pathlib import Path

from .config import MODEL_FILES_DIR, MODEL_REGISTRY_PATH, SUPPORTED_MODEL_EXTENSIONS


def ensure_model_storage() -> None:
    os.makedirs(MODEL_FILES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_REGISTRY_PATH), exist_ok=True)
    if not os.path.exists(MODEL_REGISTRY_PATH):
        with open(MODEL_REGISTRY_PATH, "w", encoding="utf-8") as file:
            json.dump([], file, ensure_ascii=False, indent=2)


def slugify(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9а-яё_-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "model"


def load_registry() -> list[dict]:
    ensure_model_storage()
    try:
        with open(MODEL_REGISTRY_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def resolve_model_path(relative_path: str) -> Path | None:
    if not relative_path:
        return None
    base = Path(MODEL_FILES_DIR).resolve()
    candidate = (base / relative_path).resolve()
    if base not in candidate.parents and candidate != base:
        return None
    return candidate


def auto_discover_models() -> list[dict]:
    ensure_model_storage()
    discovered = []
    base = Path(MODEL_FILES_DIR)
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_MODEL_EXTENSIONS:
            continue
        relative_path = path.relative_to(base).as_posix()
        discovered.append({
            "id": slugify(path.stem),
            "name": path.stem,
            "file_name": path.name,
            "relative_path": relative_path,
            "format": path.suffix.lower().lstrip("."),
            "description": None,
        })
    return discovered


def _normalize_record(record: dict) -> dict | None:
    model_id = slugify(str(record.get("id") or record.get("name") or ""))
    relative_path = record.get("relative_path") or record.get("file_path") or ""
    resolved = resolve_model_path(relative_path)
    if not model_id or resolved is None or not resolved.is_file():
        return None

    stat = resolved.stat()
    return {
        "id": model_id,
        "name": record.get("name") or resolved.stem,
        "description": record.get("description"),
        "spacecraft": record.get("spacecraft"),
        "format": record.get("format") or resolved.suffix.lower().lstrip("."),
        "file_name": record.get("file_name") or resolved.name,
        "relative_path": relative_path,
        "size_bytes": stat.st_size,
        "download_url": f"/models/{model_id}/download",
    }


def list_models() -> list[dict]:
    registry = load_registry()
    normalized = []
    seen_ids = set()
    for record in registry:
        item = _normalize_record(record)
        if not item or item["id"] in seen_ids:
            continue
        normalized.append(item)
        seen_ids.add(item["id"])

    if normalized:
        return normalized

    for record in auto_discover_models():
        item = _normalize_record(record)
        if not item or item["id"] in seen_ids:
            continue
        normalized.append(item)
        seen_ids.add(item["id"])
    return normalized


def get_model_metadata(model_id: str) -> dict | None:
    normalized_id = slugify(model_id)
    for item in list_models():
        if item["id"] == normalized_id:
            return item
    return None


def get_model_file_path(model_id: str) -> Path | None:
    item = get_model_metadata(model_id)
    if not item:
        return None
    resolved = resolve_model_path(item["relative_path"])
    if resolved is None or not resolved.is_file():
        return None
    return resolved
