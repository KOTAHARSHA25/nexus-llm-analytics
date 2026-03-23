"""Online API Clients — Nexus LLM Analytics
==========================================

All cloud API clients used when the system operates in ONLINE mode.
Each client is self-contained and raises a typed exception on failure
so the ModeManager fallback chain can catch and route to the next tier.

Clients implemented:
    GroqClient          — Groq LPU inference (primary LLM)
    GeminiClient        — Google Gemini (secondary fallback LLM)
    OpenRouterClient    — OpenRouter (third fallback LLM)
    CohereEmbedClient   — Cohere embed-english-v3.0 (primary embeddings)
    HuggingFaceEmbedClient — HF Inference API (fallback embeddings)
    FirecrawlClient     — Firecrawl clean markdown scraper (primary)
    JinaClient          — Jina Reader (fallback scraper, no key required)

Model tier mapping (mirrors existing Ollama complexity tiers):
    simple  → was TinyLlama      → Groq llama-3.1-8b-instant / Gemini Flash
    medium  → was Phi-3          → Groq mixtral-8x7b-32768 / Gemini Flash
    complex → was Llama-3        → Groq llama-3.3-70b-versatile / Gemini Pro
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Generator, List, Optional

import urllib.request
import urllib.error
import urllib.parse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions — all inherit from a common base so callers can catch
# either specifically or broadly.
# ---------------------------------------------------------------------------


class OnlineClientError(Exception):
    """Base exception for all online client failures."""


class GroqAPIException(OnlineClientError):
    """Raised when the Groq API returns an error or times out."""


class OpenRouterAPIException(OnlineClientError):
    """Raised when the OpenRouter API returns an error or times out."""


class GeminiAPIException(OnlineClientError):
    """Raised when the Gemini API returns an error or times out."""


class CohereAPIException(OnlineClientError):
    """Raised when the Cohere Embed API returns an error or times out."""


class HuggingFaceAPIException(OnlineClientError):
    """Raised when the HuggingFace Inference API returns an error."""


class FirecrawlAPIException(OnlineClientError):
    """Raised when the Firecrawl API returns an error or times out."""


class JinaAPIException(OnlineClientError):
    """Raised when the Jina Reader endpoint fails."""


# ---------------------------------------------------------------------------
# Tier → model name resolution helpers
# ---------------------------------------------------------------------------

_GROQ_MODELS: Dict[str, str] = {
    "simple": os.getenv("GROQ_LLM_SIMPLE", "llama-3.1-8b-instant"),
    "medium": os.getenv("GROQ_LLM_MEDIUM", "llama-3.3-70b-versatile"),
    "complex": os.getenv("GROQ_LLM_COMPLEX", "llama-3.3-70b-versatile"),
}

_OPENROUTER_MODELS: Dict[str, str] = {
    "simple": os.getenv("OPENROUTER_LLM_SIMPLE", "meta-llama/llama-3.2-3b-instruct:free"),
    "medium": os.getenv("OPENROUTER_LLM_MEDIUM", "meta-llama/llama-3.3-70b-instruct:free"),
    "complex": os.getenv("OPENROUTER_LLM_COMPLEX", "meta-llama/llama-3.3-70b-instruct:free"),
}


def _resolve_tier(tier: str) -> str:
    """Normalise a fuzzy tier name to simple/medium/complex."""
    t = tier.lower()
    if t in ("fast", "simple", "tiny"):
        return "simple"
    if t in ("balanced", "medium", "mid"):
        return "medium"
    return "complex"


# ---------------------------------------------------------------------------
# _post_json — shared thin HTTP helper (stdlib only, no extra deps)
# ---------------------------------------------------------------------------


def _post_json(
    url: str,
    payload: dict,
    headers: dict,
    timeout: int = 60,
) -> dict:
    """POST *payload* as JSON to *url* and return the parsed JSON response.

    Uses only the standard library (``urllib``) to avoid adding extra
    dependencies. Raises ``urllib.error.HTTPError`` or ``urllib.error.URLError``
    on failure — callers convert to typed exceptions.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_url(url: str, headers: dict, timeout: int = 30) -> str:
    """GET *url* and return the response body as a string."""
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


# ---------------------------------------------------------------------------
# GroqClient
# ---------------------------------------------------------------------------


class GroqClient:
    """Groq LPU Inference API client.

    Supports streaming (SSE) and non-streaming completion.
    Falls back to non-streaming if SSE parsing fails.

    Args:
        api_key: Groq API key (from env var ``GROQ_API_KEY``).
    """

    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
    TIMEOUT = 60

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Nexus-LLM-Analytics/2.0",
        }

    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        tier: str = "medium",
        stream: bool = False,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request and return the assistant reply.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            tier: Complexity tier — ``"simple"``, ``"medium"``, or ``"complex"``.
            stream: Whether to use SSE streaming (returns full text after draining).

        Returns:
            The assistant message content as a plain string.

        Raises:
            GroqAPIException: On any API or network error.
        """
        model = _GROQ_MODELS[_resolve_tier(tier)]
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        try:
            if stream:
                return self._stream_chat(payload)
            resp = _post_json(self.BASE_URL, payload, self._headers, self.TIMEOUT)
            return resp["choices"][0]["message"]["content"]
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
            raise GroqAPIException(f"Groq API error (model={model}): {e}") from e

    def _stream_chat(self, payload: dict) -> str:
        """Drain an SSE stream and return the accumulated text."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.BASE_URL, data=data, headers=self._headers, method="POST"
        )
        accumulated = []
        try:
            with urllib.request.urlopen(req, timeout=self.TIMEOUT) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    json_str = line[6:]
                    if json_str == "[DONE]":
                        break
                    chunk = json.loads(json_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if token := delta.get("content"):
                        accumulated.append(token)
        except Exception as e:
            raise GroqAPIException(f"Groq streaming error: {e}") from e
        return "".join(accumulated)

    def generate(self, prompt: str, tier: str = "medium", **kwargs: Any) -> str:
        """Convenience wrapper that builds a single-turn user message."""
        return self.chat([{"role": "user", "content": prompt}], tier=tier, **kwargs)


# ---------------------------------------------------------------------------
# OpenRouterClient
# ---------------------------------------------------------------------------


class OpenRouterClient:
    """OpenRouter API client — fallback when Groq is unavailable.

    Args:
        api_key: OpenRouter API key (from env var ``OPENROUTER_API_KEY``).
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    TIMEOUT = 90

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://nexus-llm-analytics",
            "X-Title": "Nexus LLM Analytics",
            "User-Agent": "Nexus-LLM-Analytics/2.0",
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        tier: str = "medium",
        stream: bool = False,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request (same interface as GroqClient).

        Raises:
            OpenRouterAPIException: On any API or network error.
        """
        model = _OPENROUTER_MODELS[_resolve_tier(tier)]
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        try:
            if stream:
                return self._stream_chat(payload)
            resp = _post_json(self.BASE_URL, payload, self._headers, self.TIMEOUT)
            return resp["choices"][0]["message"]["content"]
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
            raise OpenRouterAPIException(f"OpenRouter API error (model={model}): {e}") from e

    def _stream_chat(self, payload: dict) -> str:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.BASE_URL, data=data, headers=self._headers, method="POST"
        )
        accumulated = []
        try:
            with urllib.request.urlopen(req, timeout=self.TIMEOUT) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    json_str = line[6:]
                    if json_str == "[DONE]":
                        break
                    chunk = json.loads(json_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if token := delta.get("content"):
                        accumulated.append(token)
        except Exception as e:
            raise OpenRouterAPIException(f"OpenRouter streaming error: {e}") from e
        return "".join(accumulated)

    def generate(self, prompt: str, tier: str = "medium", **kwargs: Any) -> str:
        return self.chat([{"role": "user", "content": prompt}], tier=tier, **kwargs)


# ---------------------------------------------------------------------------
# GeminiClient
# ---------------------------------------------------------------------------


class GeminiClient:
    """Google Gemini API client — fallback between Groq and OpenRouter.

    Args:
        api_key: Gemini API key (from env var ``GEMINI_API_KEY``).
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}"
    TIMEOUT = 60

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._headers = {
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        tier: str = "medium",
        stream: bool = False,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request to Gemini."""
        # Use flash for all tiers as pro has strict free-tier 429 limits
        model = "gemini-2.5-flash"
            
        gemini_contents = []
        for msg in messages:
            role = "user" if msg["role"] in ("user", "system") else "model"
            gemini_contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
            
        payload = {"contents": gemini_contents}
        url = self.BASE_URL.format(model, self.api_key)
        
        try:
            resp = _post_json(url, payload, self._headers, self.TIMEOUT)
            
            if "error" in resp:
                raise GeminiAPIException(resp["error"].get("message", "Unknown error"))
                
            return resp["candidates"][0]["content"]["parts"][0]["text"]
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as e:
            raise GeminiAPIException(f"Gemini API error (model={model}): {e}") from e

    def generate(self, prompt: str, tier: str = "medium", **kwargs: Any) -> str:
        return self.chat([{"role": "user", "content": prompt}], tier=tier, **kwargs)


# ---------------------------------------------------------------------------
# CohereEmbedClient
# ---------------------------------------------------------------------------


class CohereEmbedClient:
    """Cohere Embed v3 API client.

    Args:
        api_key: Cohere API key (from env var ``COHERE_API_KEY``).
    """

    BASE_URL = "https://api.cohere.ai/v1/embed"
    MODEL = "embed-english-v3.0"
    TIMEOUT = 30

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def embed(
        self,
        texts: List[str],
        input_type: str = "search_document",
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed.
            input_type: ``"search_document"`` for indexing or
                        ``"search_query"`` for query-time retrieval.

        Returns:
            List of embedding vectors (one per input text).

        Raises:
            CohereAPIException: On any API or network error.
        """
        payload = {
            "model": self.MODEL,
            "texts": texts,
            "input_type": input_type,
            "embedding_types": ["float"],
        }
        try:
            resp = _post_json(self.BASE_URL, payload, self._headers, self.TIMEOUT)
            return resp["embeddings"]["float"]
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
            raise CohereAPIException(f"Cohere API error: {e}") from e

    def embed_query(self, text: str) -> List[float]:
        """Convenience method for single query embedding."""
        return self.embed([text], input_type="search_query")[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Convenience method for document indexing embeddings."""
        return self.embed(texts, input_type="search_document")


# ---------------------------------------------------------------------------
# HuggingFaceEmbedClient
# ---------------------------------------------------------------------------


class HuggingFaceEmbedClient:
    """HuggingFace Inference API embedding client.

    Uses ``sentence-transformers/all-MiniLM-L6-v2`` via the free
    HuggingFace Inference API endpoint.

    Args:
        api_key: HuggingFace API key (from env var ``HUGGINGFACE_API_KEY``).
    """

    MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    BASE_URL = (
        "https://api-inference.huggingface.co/pipeline/feature-extraction/"
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    TIMEOUT = 30

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Returns:
            List of float vectors.

        Raises:
            HuggingFaceAPIException: On any API or network error.
        """
        payload = {"inputs": texts}
        try:
            resp = _post_json(self.BASE_URL, payload, self._headers, self.TIMEOUT)
            # HF returns a list of vectors directly
            if isinstance(resp, list):
                return resp
            raise HuggingFaceAPIException("Unexpected response shape from HuggingFace API")
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as e:
            raise HuggingFaceAPIException(f"HuggingFace API error: {e}") from e

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts)


# ---------------------------------------------------------------------------
# FirecrawlClient
# ---------------------------------------------------------------------------


class FirecrawlClient:
    """Firecrawl web scraping client — returns clean markdown.

    Args:
        api_key: Firecrawl API key (from env var ``FIRECRAWL_API_KEY``).
    """

    BASE_URL = "https://api.firecrawl.dev/v1/scrape"
    TIMEOUT = 45

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def scrape(self, url: str) -> str:
        """Scrape a URL and return its content as clean markdown.

        Args:
            url: The URL to scrape.

        Returns:
            Markdown string of the page content.

        Raises:
            FirecrawlAPIException: If the API call fails.
        """
        payload = {"url": url, "formats": ["markdown"]}
        try:
            resp = _post_json(self.BASE_URL, payload, self._headers, self.TIMEOUT)
            if not resp.get("success"):
                raise FirecrawlAPIException(
                    f"Firecrawl returned success=false for {url}: {resp}"
                )
            return resp.get("data", {}).get("markdown", "")
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
            raise FirecrawlAPIException(f"Firecrawl API error for {url}: {e}") from e


# ---------------------------------------------------------------------------
# JinaClient
# ---------------------------------------------------------------------------


class JinaClient:
    """Jina Reader API client — no key required.

    Wraps URLs via ``https://r.jina.ai/{url}`` to get clean content.
    """

    BASE = "https://r.jina.ai/"
    TIMEOUT = 30

    def scrape(self, url: str) -> str:
        """Fetch a URL through Jina Reader and return the cleaned text.

        Args:
            url: The URL to fetch.

        Returns:
            Clean text/markdown of the page.

        Raises:
            JinaAPIException: If the fetch fails.
        """
        safe_url = urllib.parse.quote(url, safe=":/?=&%#@!$'()*+,;")
        jina_url = f"{self.BASE}{safe_url}"
        try:
            return _get_url(jina_url, headers={"Accept": "text/plain"}, timeout=self.TIMEOUT)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            raise JinaAPIException(f"Jina API error for {url}: {e}") from e
