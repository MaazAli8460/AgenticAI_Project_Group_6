from __future__ import annotations

import json
from typing import Any

import httpx


class GroqClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_s: float = 60.0,
    ) -> None:
        self._model = model
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    @property
    def model(self) -> str:
        return self._model

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        response = self._client.post("/chat/completions", json=payload)
        if response.status_code == 400:
            payload.pop("response_format", None)
            response = self._client.post("/chat/completions", json=payload)
        if response.status_code >= 400:
            raise RuntimeError(self._format_error(response))
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return self._parse_json(content)

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            return json.loads(content[start : end + 1])

    @staticmethod
    def _format_error(response: httpx.Response) -> str:
        detail = response.text.strip()
        try:
            data = response.json()
        except ValueError:
            data = None

        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                error_type = error.get("type")
                if message and error_type:
                    detail = f"{message} ({error_type})"
                elif message:
                    detail = message
            elif data:
                detail = json.dumps(data)

        return f"Groq API error {response.status_code}: {detail}"
