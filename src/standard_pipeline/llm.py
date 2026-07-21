from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Protocol

import requests


class ChatClient(Protocol):
    def complete_json(self, system_prompt: str, user_content: str) -> str:
        ...


@dataclass(frozen=True)
class ApiClientConfig:
    api_key: str
    base_url: str | None
    model: str
    temperature: float = 0.0
    json_response_format: bool = True
    max_tokens: int | None = None
    extra_body: dict | None = None


class OpenAICompatibleClient:
    def __init__(self, config: ApiClientConfig):
        self._config = config

    def complete_json(self, system_prompt: str, user_content: str) -> str:
        base_url = self._config.base_url
        if not base_url:
            raise ValueError("Missing API base_url for OpenAI-compatible /chat/completions request.")
        url = base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self._config.temperature,
        }
        if self._config.json_response_format:
            payload["response_format"] = {"type": "json_object"}
        if self._config.max_tokens:
            payload["max_tokens"] = self._config.max_tokens
        if self._config.extra_body:
            payload.update(self._config.extra_body)

        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=300,
        )
        if not 200 <= response.status_code < 300:
            raise RuntimeError(
                "OpenAI-compatible API request failed: "
                f"status_code={response.status_code}; "
                f"url={url}; "
                f"model={self._config.model}; "
                f"response_text={response.text[:2000]}"
            )
        data = response.json()
        return data["choices"][0]["message"]["content"]


@dataclass(frozen=True)
class LocalClientConfig:
    model_path: str
    device_map: str = "auto"
    torch_dtype: str = "auto"
    max_new_tokens: int = 1024
    max_input_chars: int | None = None
    load_in_4bit: bool = False


class LocalModelClient:
    def __init__(self, config: LocalClientConfig):
        self._config = config
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Local model mode requires torch, transformers, accelerate, and sentencepiece."
            ) from exc

        model_kwargs = {
            "device_map": self._config.device_map,
            "torch_dtype": self._config.torch_dtype,
            "trust_remote_code": True,
        }
        if self._config.load_in_4bit:
            from transformers import BitsAndBytesConfig

            compute_dtype = torch.bfloat16
            if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                compute_dtype = torch.float16
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self._config.model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(self._config.model_path, **model_kwargs)
        self._model.eval()

        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer, self._model

    def complete_json(self, system_prompt: str, user_content: str) -> str:
        tokenizer, model = self._load()
        if self._config.max_input_chars:
            user_content = user_content[: self._config.max_input_chars]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt_text = f"System:\n{system_prompt}\n\nUser:\n{user_content}\n\nAssistant:\n"

        inputs = tokenizer(prompt_text, return_tensors="pt")
        input_length = inputs["input_ids"].shape[-1]
        model_device = next(model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        import torch

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self._config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][input_length:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def api_config_from_dict(data: dict, api_key: str | None = None) -> ApiClientConfig:
    key_env = data.get("api_key_env", "STANDARD_PIPELINE_API_KEY")
    base_url_env = data.get("base_url_env", "STANDARD_PIPELINE_BASE_URL")
    model_env = data.get("model_env", "STANDARD_PIPELINE_MODEL")
    resolved_key = api_key or os.environ.get(key_env)
    if not resolved_key:
        raise ValueError(f"Missing API key. Set {key_env} or pass --api-key.")
    max_tokens = int(data.get("max_tokens", 0) or 0)
    extra_body = data.get("extra_body")
    return ApiClientConfig(
        api_key=resolved_key,
        base_url=os.environ.get(base_url_env) or data.get("base_url"),
        model=os.environ.get(model_env) or data.get("model", "deepseek-chat"),
        temperature=float(data.get("temperature", 0.0)),
        json_response_format=bool(data.get("json_response_format", True)),
        max_tokens=max_tokens or None,
        extra_body=extra_body if isinstance(extra_body, dict) else None,
    )


def local_config_from_dict(data: dict, model_path: str | None = None) -> LocalClientConfig:
    path_env = data.get("model_path_env", "LOCAL_MODEL_PATH")
    resolved_path = model_path or os.environ.get(path_env) or data.get("model_path")
    if not resolved_path:
        raise ValueError(f"Missing local model path. Set {path_env} or pass --model-path.")
    max_input_chars = int(data.get("max_input_chars", 0) or 0)
    return LocalClientConfig(
        model_path=resolved_path,
        device_map=data.get("device_map", "auto"),
        torch_dtype=data.get("torch_dtype", "auto"),
        max_new_tokens=int(data.get("max_new_tokens", 1024)),
        max_input_chars=max_input_chars or None,
        load_in_4bit=bool(data.get("load_in_4bit", False)),
    )
