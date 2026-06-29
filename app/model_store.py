import json
import os
import re
from pathlib import Path

from .config import MODEL_FILES_DIR, MODEL_REGISTRY_PATH
from .database import SessionLocal
from .models import Spacecraft


def ensure_model_storage() -> None:
    os.makedirs(MODEL_FILES_DIR, exist_ok=True)
    _migrate_registry_to_db()


def slugify(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9а-яё_-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "model"


def _migrate_registry_to_db() -> None:
    if not os.path.exists(MODEL_REGISTRY_PATH):
        return
    try:
        with open(MODEL_REGISTRY_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list) or not records:
            return
    except (OSError, json.JSONDecodeError):
        return

    db = SessionLocal()
    try:
        if db.query(Spacecraft).count() > 0:
            return
        for record in records:
            model_id = slugify(str(record.get("id") or record.get("name") or ""))
            if not model_id:
                continue
            sc = Spacecraft(
                id=model_id,
                name=record.get("name") or model_id,
                description=record.get("description"),
                spacecraft_name=record.get("spacecraft"),
                format=record.get("format") or "fbx",
                file_name=record.get("file_name") or "",
                relative_path=record.get("relative_path") or "",
                scene=record.get("scene") if isinstance(record.get("scene"), dict) else None,
            )
            db.merge(sc)
        db.commit()
        print(f"[model_store] Migrated {len(records)} record(s) from registry.json to DB")
    except Exception as exc:
        db.rollback()
        print(f"[model_store] Migration failed: {exc}")
    finally:
        db.close()


def resolve_model_path(relative_path: str) -> Path | None:
    if not relative_path:
        return None
    base = Path(MODEL_FILES_DIR).resolve()
    candidate = (base / relative_path).resolve()
    if base not in candidate.parents and candidate != base:
        return None
    return candidate


def _spacecraft_to_dict(sc: Spacecraft) -> dict | None:
    resolved = resolve_model_path(sc.relative_path)
    if resolved is None or not resolved.is_file():
        return None
    stat = resolved.stat()
    return {
        "id": sc.id,
        "name": sc.name,
        "description": sc.description,
        "spacecraft": sc.spacecraft_name,
        "format": sc.format,
        "file_name": sc.file_name,
        "relative_path": sc.relative_path,
        "size_bytes": stat.st_size,
        "download_url": f"/models/{sc.id}/download",
        "scene": sc.scene,
    }


def list_models() -> list[dict]:
    db = SessionLocal()
    try:
        rows = db.query(Spacecraft).all()
        result = [_spacecraft_to_dict(sc) for sc in rows]
        return [r for r in result if r is not None]
    finally:
        db.close()


def get_model_metadata(model_id: str) -> dict | None:
    normalized_id = slugify(model_id)
    db = SessionLocal()
    try:
        sc = db.get(Spacecraft, normalized_id)
        if not sc:
            return None
        return _spacecraft_to_dict(sc)
    finally:
        db.close()


def get_model_file_path(model_id: str) -> Path | None:
    item = get_model_metadata(model_id)
    if not item:
        return None
    resolved = resolve_model_path(item["relative_path"])
    if resolved is None or not resolved.is_file():
        return None
    return resolved
