import json
import urllib.error
import urllib.request
from functools import lru_cache

import torch

from ..config import (
    LLM_BACKEND,
    LLM_MAX_INPUT_TOKENS,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    VLLM_MODEL,
    VLLM_TIMEOUT,
    YANDEX_API_KEY,
    YANDEX_BASE_URL,
    YANDEX_LOG_USAGE,
    YANDEX_MODEL,
    YANDEX_PROJECT_ID,
    YANDEX_TIMEOUT,
)
from ..runtime import device, model, tokenizer


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

    from ..config import YANDEX_PROMPT_ID
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


@lru_cache(maxsize=256)
def _run_chat_generation_cached(messages_json: str, max_new_tokens: int, prompt_id: str | None) -> str:
    messages = json.loads(messages_json)
    if LLM_BACKEND == "vllm":
        return generate_with_vllm(messages, max_new_tokens)
    if LLM_BACKEND == "yandex":
        return generate_with_yandex(messages, max_new_tokens, prompt_id=prompt_id, usage_label="cached")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=LLM_MAX_INPUT_TOKENS).to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs.input_ids.shape[1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def run_chat_generation(
    messages: list[dict],
    max_new_tokens: int,
    prompt_id: str | None = None,
    usage_label: str = "default",
) -> str:
    if usage_label in ("rag_compact", "router"):
        return _run_chat_generation_cached(
            json.dumps(messages, ensure_ascii=False, sort_keys=True),
            max_new_tokens,
            prompt_id,
        )

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
