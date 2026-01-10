from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI


class OpenRouterClient:
    """
    OpenRouter is OpenAI-compatible.
    We enforce *free* models by ensuring the model ends with ':free'.
    If OpenRouter rate-limits the free tier, we return an empty string so callers can fall back safely.
    """

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.site_url = os.getenv("APP_URL", "").strip()      # optional
        self.app_title = os.getenv("APP_NAME", "Copilot Survey Research Demo").strip()

        self.client: Optional[OpenAI] = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url="https://openrouter.ai/api/v1")

    def is_configured(self) -> bool:
        return self.client is not None

    @staticmethod
    def ensure_free_model(model: str) -> str:
        model = (model or "").strip()
        if not model:
            return "mistral/devstral-2-2512:free"
        if not model.endswith(":free"):
            model = model + ":free"
        return model

    def chat(self, prompt: str, model: str, temperature: float = 0.2) -> str:
        if not self.client:
            return ""

        model = self.ensure_free_model(model)

        extra_headers = {}
        if self.site_url:
            extra_headers["HTTP-Referer"] = self.site_url
        if self.app_title:
            extra_headers["X-Title"] = self.app_title

        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a survey research expert. Be precise, unbiased, and concise. Return JSON only when asked."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                extra_headers=extra_headers,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            # Fail closed: return empty string so the app continues (falls back to deterministic mode).
            return ""
