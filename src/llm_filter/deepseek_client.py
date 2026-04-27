"""
DeepSeek API Client (OpenAI-compatible)

Description:
- Uses requests to call DeepSeek's chat.completions-style API (OpenAI compatible).
- By default, reads the API key from an environment variable to avoid embedding secrets
  in code or configuration repositories.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass(frozen=True)
class DeepSeekClientConfig:
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    model: str = "deepseek-chat"
    timeout_s: int = 60
    max_retries: int = 3


class DeepSeekClient:
    def __init__(self, cfg: DeepSeekClientConfig) -> None:
        self.cfg = cfg
        self.api_key = os.getenv(cfg.api_key_env)
        if not self.api_key:
            raise RuntimeError(f"Missing API key env: {cfg.api_key_env}")

    def chat(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        url = self.cfg.base_url.rstrip("/") + "/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.cfg.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            try:
                r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.cfg.timeout_s)
                if r.status_code >= 400:
                    # retry on rate limit / transient
                    if r.status_code in (429, 500, 502, 503, 504):
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    raise RuntimeError(f"DeepSeek HTTP {r.status_code}: {r.text[:500]}")
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError("DeepSeek request failed") from last_err


