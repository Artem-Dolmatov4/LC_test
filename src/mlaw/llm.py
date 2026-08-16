"""Клиент к DeepSeek — для генерации вопросов корзины и ответов с цитатами.

Модель используется в двух местах, и в обоих её роль вспомогательная:

* шаг 5 — сочиняет вопросы по известному фрагменту, чтобы получить корзину
  с эталоном без ручной разметки;
* шаг 6 — собирает ответ из найденных фрагментов и расставляет ссылки.

Ни там, ни там модель не является предметом измерения. Качество поиска
меряется отдельно от качества формулировок, а корректность ссылок
проверяется программно, а не доверием к модели.
"""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

__all__ = ["DeepSeek", "LLMResult"]


@dataclass(slots=True)
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def json(self) -> dict | list:
        """Разбирает ответ как JSON, снимая ограждение ```json если оно есть."""
        raw = self.text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            raw = raw.rsplit("```", 1)[0]
        return json.loads(raw)


class DeepSeek:
    ENDPOINT = "https://api.deepseek.com/chat/completions"

    def __init__(
        self,
        model: str = "deepseek-v4-pro",
        *,
        api_key: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 4,
        temperature: float = 0.3,
    ):
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("нет DEEPSEEK_API_KEY — ни в аргументе, ни в окружении")
        self._key = key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int = 2000,
    ) -> LLMResult:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        request = urllib.request.Request(
            self.ENDPOINT,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key}",
            },
        )

        delay = 3.0
        started = time.time()
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read())
                usage = body.get("usage") or {}
                self.prompt_tokens += usage.get("prompt_tokens", 0)
                self.completion_tokens += usage.get("completion_tokens", 0)
                return LLMResult(
                    text=body["choices"][0]["message"]["content"] or "",
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    seconds=time.time() - started,
                )
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:300]
                if exc.code not in (429, 500, 502, 503, 504) or attempt == self.max_retries - 1:
                    raise RuntimeError(f"DeepSeek вернул HTTP {exc.code}: {detail}") from exc
                time.sleep(delay)
                delay *= 2
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                http.client.HTTPException,
            ):
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("недостижимо")
