"""Mode Manager — Online/Offline Runtime Switch
===============================================

Singleton class that controls whether the system uses cloud APIs (Online)
or local Ollama models (Offline). Defaults to OFFLINE on every startup.

Fallback chains (per the spec):
  LLM:        Groq → OpenRouter → Ollama
  Embeddings: Cohere → HuggingFace → local
  Scraping:   Firecrawl → Jina → local scraper

All clients are lazily initialized on first use. Every online call is
wrapped in try/except with automatic fallback logging so the user never
sees a raw API error.

Usage::

    from backend.core.mode_manager import get_mode_manager

    manager = get_mode_manager()
    manager.set_mode("online")
    client = manager.get_llm_client()
    result = client.chat(messages, tier="simple")

Thread Safety
-------------
The singleton accessor uses double-checked locking. The mode switch itself
is protected by a threading.Lock so concurrent requests mid-switch see a
consistent mode.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton state
# ---------------------------------------------------------------------------
_mode_manager_instance: Optional[ModeManager] = None
_mode_manager_lock = threading.Lock()


class ModeManager:
    """Runtime Online/Offline mode controller.

    Attributes:
        _mode: Current mode string — ``"online"`` or ``"offline"``.

    All cloud clients are lazily initialized. Clients that fail to initialize
    are recorded as ``None`` so the fallback chain is used automatically.
    """

    ONLINE = "online"
    OFFLINE = "offline"

    def __init__(self) -> None:
        # Read initial mode from NEXUS_MODE env var; default to OFFLINE for safety
        _env_mode = os.getenv("NEXUS_MODE", "offline").strip().lower()
        self._mode: str = _env_mode if _env_mode in (self.ONLINE, self.OFFLINE) else self.OFFLINE
        self._lock = threading.Lock()

        # Lazy-loaded cloud clients
        self._groq_client: Optional[Any] = None
        self._gemini_client: Optional[Any] = None
        self._openrouter_client: Optional[Any] = None
        self._cohere_client: Optional[Any] = None
        self._hf_client: Optional[Any] = None
        self._firecrawl_client: Optional[Any] = None
        self._jina_client: Optional[Any] = None

        # Init flags to prevent repeated failed attempts per process lifetime
        self._groq_init_attempted = False
        self._gemini_init_attempted = False
        self._openrouter_init_attempted = False
        self._cohere_init_attempted = False
        self._hf_init_attempted = False
        self._firecrawl_init_attempted = False
        self._jina_init_attempted = False

        # Ollama process we launched (None if started externally or not at all)
        self._ollama_process: Optional[subprocess.Popen] = None

        logger.info("[ModeManager] Initialized in OFFLINE mode")

    # ------------------------------------------------------------------
    # Mode control
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Switch between online and offline modes at runtime.

        Args:
            mode: ``"online"`` or ``"offline"`` (case-insensitive).

        Raises:
            ValueError: If *mode* is not one of the accepted values.
        """
        normalized = mode.strip().lower()
        if normalized not in (self.ONLINE, self.OFFLINE):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'online' or 'offline'.")
        with self._lock:
            previous = self._mode
            self._mode = normalized
        if previous != normalized:
            logger.info("[ModeManager] Mode switched: %s \u2192 %s", previous, normalized)
            # Manage Ollama in a daemon thread so HTTP response is not blocked
            if normalized == self.OFFLINE:
                threading.Thread(target=self._start_ollama_bg, daemon=True).start()
            else:
                threading.Thread(target=self._stop_ollama_bg, daemon=True).start()

    def get_mode(self) -> str:
        """Return the current mode (``"online"`` or ``"offline"``)."""
        return self._mode

    # ------------------------------------------------------------------
    # Ollama process lifecycle
    # ------------------------------------------------------------------

    def is_ollama_running(self) -> bool:
        """Return True if the Ollama HTTP server is responsive on localhost:11434."""
        try:
            with urllib.request.urlopen("http://localhost:11434/", timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def start_ollama(self) -> None:
        """Start the Ollama server if not already running.

        On Windows, opens a new console window so the user can see output.
        On other platforms, runs silently in the background.
        Waits up to 20 seconds for Ollama to become responsive.
        """
        if self.is_ollama_running():
            logger.info("[OllamaManager] Ollama already running \u2014 nothing to do")
            return

        logger.info("[OllamaManager] Starting Ollama server...")
        try:
            if platform.system() == "Windows":
                self._ollama_process = subprocess.Popen(
                    ["ollama", "serve"],
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                )
            else:
                self._ollama_process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except FileNotFoundError:
            logger.error("[OllamaManager] 'ollama' not found on PATH. Is Ollama installed?")
            return
        except Exception as exc:
            logger.error("[OllamaManager] Failed to start Ollama: %s", exc)
            return

        # Wait for Ollama to become responsive (up to 20 s)
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if self.is_ollama_running():
                logger.info(
                    "[OllamaManager] Ollama is up (PID %s)",
                    self._ollama_process.pid,
                )
                return
            time.sleep(0.5)

        logger.warning("[OllamaManager] Ollama started but not responsive within 20 s")

    def stop_ollama(self) -> None:
        """Stop the Ollama server.

        Terminates the process we launched first. If Ollama was started
        externally and is still running, uses a platform kill command.
        """
        if not self.is_ollama_running():
            logger.info("[OllamaManager] Ollama not running \u2014 nothing to stop")
            self._ollama_process = None
            return

        logger.info("[OllamaManager] Stopping Ollama server...")

        # Try the process we own first
        if self._ollama_process is not None:
            try:
                self._ollama_process.terminate()
                self._ollama_process.wait(timeout=5)
                logger.info(
                    "[OllamaManager] Ollama process terminated (PID %s)",
                    self._ollama_process.pid,
                )
            except Exception as exc:
                logger.warning("[OllamaManager] terminate() failed: %s", exc)
            finally:
                self._ollama_process = None

        # If still running (e.g. started externally), use platform kill
        if self.is_ollama_running():
            try:
                if platform.system() == "Windows":
                    subprocess.run(
                        ["taskkill", "/F", "/IM", "ollama.exe"],
                        capture_output=True, timeout=10,
                    )
                else:
                    subprocess.run(
                        ["pkill", "-f", "ollama serve"],
                        capture_output=True, timeout=10,
                    )
                logger.info("[OllamaManager] Ollama killed via system command")
            except Exception as exc:
                logger.warning("[OllamaManager] System kill failed: %s", exc)

    def _start_ollama_bg(self) -> None:
        """Background thread target — wraps start_ollama with error logging."""
        try:
            self.start_ollama()
        except Exception as exc:
            logger.error("[OllamaManager] Background start error: %s", exc)

    def _stop_ollama_bg(self) -> None:
        """Background thread target — wraps stop_ollama with error logging."""
        try:
            self.stop_ollama()
        except Exception as exc:
            logger.error("[OllamaManager] Background stop error: %s", exc)

    # ------------------------------------------------------------------
    # LLM client access (Groq \u2192 OpenRouter \u2192 Ollama)
    # ------------------------------------------------------------------

    def get_llm_client(self) -> Any:
        """Return the best available LLM client for the current mode.

        In OFFLINE mode, always returns the Ollama-backed client.
        In ONLINE mode, tries Groq → OpenRouter → Ollama (fallback).

        Returns:
            A client object with a ``.chat(messages, tier)`` / ``.generate()``
            compatible interface. In the worst case, returns ``None`` (callers
            must handle this gracefully).
        """
        if self._mode == self.OFFLINE:
            return None  # Callers use their existing Ollama path

        # --- Online path ---
        groq = self._get_groq()
        if groq is not None:
            return groq

        logger.warning("[ModeManager] Groq unavailable, trying OpenRouter fallback")
        openrouter = self._get_openrouter()
        if openrouter is not None:
            return openrouter

        logger.warning("[ModeManager] OpenRouter unavailable, falling back to Ollama")
        return None  # Signal to caller: use Ollama

    # ------------------------------------------------------------------
    # Embedding client access (Cohere → HuggingFace → local)
    # ------------------------------------------------------------------

    def get_embedding_client(self) -> Any:
        """Return the best available embedding client for the current mode.

        In OFFLINE mode, returns ``None`` (callers use local embeddings).
        In ONLINE mode, tries Cohere → HuggingFace → local (None).
        """
        if self._mode == self.OFFLINE:
            return None

        cohere = self._get_cohere()
        if cohere is not None:
            return cohere

        logger.warning("[ModeManager] Cohere unavailable, trying HuggingFace fallback")
        hf = self._get_huggingface()
        if hf is not None:
            return hf

        logger.warning("[ModeManager] HuggingFace unavailable, falling back to local embeddings")
        return None

    # ------------------------------------------------------------------
    # Scraper client access (Firecrawl → Jina → local)
    # ------------------------------------------------------------------

    def get_scraper_client(self) -> Any:
        """Return the best available web scraper client for the current mode.

        In OFFLINE mode, returns ``None`` (callers use their local scraper).
        In ONLINE mode, tries Firecrawl → Jina → local (None).
        """
        if self._mode == self.OFFLINE:
            return None

        firecrawl = self._get_firecrawl()
        if firecrawl is not None:
            return firecrawl

        logger.warning("[ModeManager] Firecrawl unavailable, trying Jina fallback")
        jina = self._get_jina()
        if jina is not None:
            return jina

        logger.warning("[ModeManager] Jina unavailable, falling back to local scraper")
        return None

    # ------------------------------------------------------------------
    # Private lazy-init helpers
    # ------------------------------------------------------------------

    def _get_groq(self):
        if self._groq_client is not None:
            return self._groq_client
        if self._groq_init_attempted:
            return None
        self._groq_init_attempted = True
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            logger.debug("[ModeManager] GROQ_API_KEY not set — skipping Groq")
            return None
        try:
            from backend.core.online_clients import GroqClient
            self._groq_client = GroqClient(api_key=api_key)
            logger.info("[ModeManager] GroqClient initialized")
            return self._groq_client
        except Exception as e:
            logger.warning("[ModeManager] GroqClient init failed: %s", e)
            return None

    def _get_openrouter(self):
        if self._openrouter_client is not None:
            return self._openrouter_client
        if self._openrouter_init_attempted:
            return None
        self._openrouter_init_attempted = True
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            logger.debug("[ModeManager] OPENROUTER_API_KEY not set — skipping OpenRouter")
            return None
        try:
            from backend.core.online_clients import OpenRouterClient
            self._openrouter_client = OpenRouterClient(api_key=api_key)
            logger.info("[ModeManager] OpenRouterClient initialized")
            return self._openrouter_client
        except Exception as e:
            logger.warning("[ModeManager] OpenRouterClient init failed: %s", e)
            return None

    def _get_gemini(self):
        if self._gemini_client is not None:
            return self._gemini_client
        if self._gemini_init_attempted:
            return None
        self._gemini_init_attempted = True
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            logger.debug("[ModeManager] GEMINI_API_KEY not set — skipping Gemini")
            return None
        try:
            from backend.core.online_clients import GeminiClient
            self._gemini_client = GeminiClient(api_key=api_key)
            logger.info("[ModeManager] GeminiClient initialized")
            return self._gemini_client
        except Exception as e:
            logger.warning("[ModeManager] GeminiClient init failed: %s", e)
            return None

    def _get_cohere(self):
        if self._cohere_client is not None:
            return self._cohere_client
        if self._cohere_init_attempted:
            return None
        self._cohere_init_attempted = True
        api_key = os.getenv("COHERE_API_KEY", "").strip()
        if not api_key:
            logger.debug("[ModeManager] COHERE_API_KEY not set — skipping Cohere")
            return None
        try:
            from backend.core.online_clients import CohereEmbedClient
            self._cohere_client = CohereEmbedClient(api_key=api_key)
            logger.info("[ModeManager] CohereEmbedClient initialized")
            return self._cohere_client
        except Exception as e:
            logger.warning("[ModeManager] CohereEmbedClient init failed: %s", e)
            return None

    def _get_huggingface(self):
        if self._hf_client is not None:
            return self._hf_client
        if self._hf_init_attempted:
            return None
        self._hf_init_attempted = True
        api_key = os.getenv("HUGGINGFACE_API_KEY", "").strip()
        if not api_key:
            logger.debug("[ModeManager] HUGGINGFACE_API_KEY not set — skipping HuggingFace")
            return None
        try:
            from backend.core.online_clients import HuggingFaceEmbedClient
            self._hf_client = HuggingFaceEmbedClient(api_key=api_key)
            logger.info("[ModeManager] HuggingFaceEmbedClient initialized")
            return self._hf_client
        except Exception as e:
            logger.warning("[ModeManager] HuggingFaceEmbedClient init failed: %s", e)
            return None

    def _get_firecrawl(self):
        if self._firecrawl_client is not None:
            return self._firecrawl_client
        if self._firecrawl_init_attempted:
            return None
        self._firecrawl_init_attempted = True
        api_key = os.getenv("FIRECRAWL_API_KEY", "").strip()
        if not api_key:
            logger.debug("[ModeManager] FIRECRAWL_API_KEY not set — skipping Firecrawl")
            return None
        try:
            from backend.core.online_clients import FirecrawlClient
            self._firecrawl_client = FirecrawlClient(api_key=api_key)
            logger.info("[ModeManager] FirecrawlClient initialized")
            return self._firecrawl_client
        except Exception as e:
            logger.warning("[ModeManager] FirecrawlClient init failed: %s", e)
            return None

    def _get_jina(self):
        if self._jina_client is not None:
            return self._jina_client
        if self._jina_init_attempted:
            return None
        self._jina_init_attempted = True
        try:
            from backend.core.online_clients import JinaClient
            self._jina_client = JinaClient()
            logger.info("[ModeManager] JinaClient initialized")
            return self._jina_client
        except Exception as e:
            logger.warning("[ModeManager] JinaClient init failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Reset (for testing / mode switches that should re-probe APIs)
    # ------------------------------------------------------------------

    def reset_clients(self) -> None:
        """Clear all cached clients so they are re-initialized on next access.

        Call this after switching to online mode to force fresh API key checks.
        """
        with self._lock:
            self._groq_client = None
            self._gemini_client = None
            self._openrouter_client = None
            self._cohere_client = None
            self._hf_client = None
            self._firecrawl_client = None
            self._jina_client = None
            self._groq_init_attempted = False
            self._gemini_init_attempted = False
            self._openrouter_init_attempted = False
            self._cohere_init_attempted = False
            self._hf_init_attempted = False
            self._firecrawl_init_attempted = False
            self._jina_init_attempted = False
        logger.debug("[ModeManager] All clients reset")


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

def get_mode_manager() -> ModeManager:
    """Return the global ModeManager singleton (thread-safe, double-checked).

    Returns:
        The single shared :class:`ModeManager` instance for this process.
    """
    global _mode_manager_instance
    if _mode_manager_instance is None:
        with _mode_manager_lock:
            if _mode_manager_instance is None:
                _mode_manager_instance = ModeManager()
    return _mode_manager_instance
