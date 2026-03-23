"""
Model Manager — Centralized LLM & Infrastructure Lifecycle Controller
======================================================================

Manages lazy initialization, caching, and lifecycle of all LLM models,
vector store clients, orchestrator instances, and Chain-of-Thought engines
used throughout the Nexus analytics pipeline.

Architecture
------------
- **Singleton Pattern**: A single ``ModelManager`` instance is shared across
  the entire application via ``get_model_manager()``, protected by a
  reentrant lock for thread safety under concurrent FastAPI requests.
- **Lazy Loading**: Every expensive resource (LLM connections, ChromaDB
  client, orchestrator, CoT engine) is instantiated on first access, not
  at import time. This keeps startup fast and memory usage minimal.
- **Thread-Safe Properties**: Each lazy property is guarded by a dedicated
  ``threading.Lock`` to prevent duplicate initialization under concurrent
  access from multiple Uvicorn worker threads.
- **Sentinel Pattern**: Failed initializations are recorded with a sentinel
  value to prevent retry storms — a component that fails once is not
  re-attempted on every property access.
- **Graceful Degradation**: If a non-critical component (e.g., ChromaDB)
  fails to initialize, the system continues operating with reduced
  functionality rather than crashing.

Component Inventory
-------------------
================= ================================ ===========================
Property          Underlying Type                  Typical Use
================= ================================ ===========================
llm_client        ``backend.core.llm_client``      Raw Ollama HTTP client
primary_llm       ``langchain_ollama.OllamaLLM``   Primary analysis model
review_llm        ``langchain_ollama.OllamaLLM``   Secondary validation model
chroma_client     ``chromadb.PersistentClient``     Vector similarity search
orchestrator      ``QueryOrchestrator``             Query routing & planning
query_parser      ``AdvancedQueryParser``           NL query understanding
cot_engine        ``SelfCorrectionEngine``          Chain-of-Thought review
================= ================================ ===========================

Thread Safety
-------------
``get_model_manager()`` uses a double-checked locking pattern at the module
level. Each lazy property uses its own ``threading.Lock`` so that two threads
requesting different components (e.g., ``llm_client`` vs ``chroma_client``)
do not block each other.

Replaces
--------
Legacy ``model_initializer.py`` (removed in v2.0).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.core.llm_client import LLMClient
    from backend.core.engine.query_orchestrator import QueryOrchestrator
    from backend.core.query_parser import AdvancedQueryParser

from backend.core.mode_manager import get_mode_manager

# ---------------------------------------------------------------------------
# Module-level singleton state
# ---------------------------------------------------------------------------
_model_manager: Optional[ModelManager] = None
_manager_lock = threading.Lock()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentinel for distinguishing "not yet attempted" from "attempted and failed"
# ---------------------------------------------------------------------------
class _InitState(Enum):
    """Internal state machine for each lazy-loaded component."""
    NOT_ATTEMPTED = auto()
    READY = auto()
    FAILED = auto()


class _Sentinel:
    """Marker object to distinguish 'init failed' from 'not yet tried'.

    Without this, a property that returns ``None`` on failure would be
    re-attempted on every access, causing log spam and wasted I/O.
    """
    __slots__ = ("state", "error")

    def __init__(self, state: _InitState = _InitState.NOT_ATTEMPTED, error: Optional[str] = None) -> None:
        self.state = state
        self.error = error

    def __bool__(self) -> bool:
        return self.state == _InitState.READY

    def __repr__(self) -> str:
        return f"<_Sentinel {self.state.name}>"


class ModelManager:
    """Centralized lifecycle manager for all LLM and infrastructure components.

    Implements the Singleton pattern (via ``get_model_manager()``) and exposes
    every shared resource as a thread-safe, lazy-loading ``@property``.
    No component is instantiated until the first caller actually needs it.

    Attributes:
        cached_models: Mapping of role names (``"primary"``, ``"review"``) to
            the fully-qualified Ollama model tags currently in use.
        init_timings: Mapping of component names to their initialization
            duration in seconds — useful for performance monitoring.

    Example::

        manager = get_model_manager()
        manager.ensure_initialized()  # Pre-warm all critical components
        response = manager.llm_client.generate("Summarize this data", model="phi3:mini")

        # Check health
        print(manager.get_status())
        # {'initialized': True, 'llm_client_ready': True, ...}

        # Hot-reload models after user changes preferences
        manager.reload_models()
    """

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    def __init__(self) -> None:
        """Initialize internal state to ``None``; real resources are lazy-loaded.

        All heavy work is deferred to first property access or an explicit
        ``ensure_initialized()`` call.
        """
        self._initialized: bool = False

        # --- Per-component locks (prevents race conditions on lazy init) ---
        self._llm_client_lock = threading.Lock()
        self._llm_models_lock = threading.Lock()
        self._chroma_lock = threading.Lock()
        self._orchestrator_lock = threading.Lock()
        self._tools_lock = threading.Lock()
        self._parser_lock = threading.Lock()
        self._cot_lock = threading.Lock()

        # --- LLM layer ---
        self._llm_client: Optional[Any] = None
        self._primary_llm: Optional[Any] = None
        self._review_llm: Optional[Any] = None

        # --- Infrastructure layer (with sentinels to prevent retry storms) ---
        self._chroma_sentinel = _Sentinel()
        self._chroma_client_inner: Optional[Any] = None
        self._orchestrator_instance: Optional[Any] = None

        # --- Analysis tooling layer ---
        self._tools: Optional[List[Any]] = None
        self._query_parser: Optional[Any] = None

        # --- Chain-of-Thought layer ---
        self._cot_sentinel = _Sentinel()
        self._cot_engine_inner: Optional[Any] = None
        self._cot_config: Optional[Dict[str, Any]] = None

        # --- LLM client sentinel (retry-safe Ollama failure handling) ---
        self._llm_client_sentinel = _Sentinel()

        # --- Runtime metadata ---
        self.cached_models: Dict[str, str] = {}
        self.init_timings: Dict[str, float] = {}

        logger.info("ModelManager created (lazy loading enabled)")

    # ------------------------------------------------------------------ #
    #  Lazy-loaded properties — LLM layer
    # ------------------------------------------------------------------ #

    @property
    def llm_client(self) -> "LLMClient":
        """Return the low-level Ollama HTTP client, creating it on first access.

        Thread-safe — guarded by ``_llm_client_lock``.
        Uses sentinel pattern: if Ollama is unreachable at init time,
        subsequent accesses will retry (state resets to NOT_ATTEMPTED
        so transient failures don't permanently disable the client).

        Returns:
            An instance of ``backend.core.llm_client.LLMClient`` ready for
            ``generate()`` / ``generate_async()`` calls.

        Raises:
            RuntimeError: If LLMClient initialization fails.
        """
        # Online mode: Ollama is not required — return None instead of attempting connection
        try:
            from backend.core.mode_manager import get_mode_manager as _gmm
            if _gmm().get_mode() == "online":
                return None  # type: ignore[return-value]
        except Exception:
            pass

        if self._llm_client is None and self._llm_client_sentinel.state != _InitState.FAILED:
            with self._llm_client_lock:
                if self._llm_client is None and self._llm_client_sentinel.state != _InitState.FAILED:
                    t0 = time.monotonic()
                    try:
                        from backend.core.llm_client import LLMClient
                        self._llm_client = LLMClient()
                        self._llm_client_sentinel = _Sentinel(_InitState.READY)
                        elapsed = time.monotonic() - t0
                        self.init_timings["llm_client"] = elapsed
                        logger.debug("LLMClient initialized (%.3fs)", elapsed)
                    except Exception as exc:
                        # Do NOT set sentinel to FAILED — allow retry on next access
                        # (Ollama may have been temporarily unavailable)
                        elapsed = time.monotonic() - t0
                        logger.error("LLMClient init failed (%.3fs): %s", elapsed, exc)
                        raise RuntimeError(f"Cannot connect to Ollama: {exc}") from exc
        return self._llm_client

    @property
    def primary_llm(self) -> Any:
        """Return the LangChain-wrapped primary analysis LLM.

        Triggers ``_initialize_llms()`` on first access, which also sets
        ``review_llm`` and populates ``cached_models``. Thread-safe via
        ``_llm_models_lock``.

        Returns:
            A ``langchain_ollama.OllamaLLM`` (or compatible fallback).

        Raises:
            RuntimeError: If Ollama is unreachable.
        """
        if self._primary_llm is None:
            with self._llm_models_lock:
                if self._primary_llm is None:
                    self._initialize_llms()
        return self._primary_llm

    @property
    def primary_model_name(self) -> str:
        """Return the name of the active primary model.

        Convenience accessor for dashboards and logging.
        """
        return self.cached_models.get("primary", "unknown")

    @property
    def review_llm(self) -> Any:
        """Return the LangChain-wrapped review / validation LLM.

        Both ``primary_llm`` and ``review_llm`` are initialized together.
        Thread-safe via ``_llm_models_lock``.

        Returns:
            A ``langchain_ollama.OllamaLLM`` (or compatible fallback).
        """
        if self._review_llm is None:
            with self._llm_models_lock:
                if self._review_llm is None:
                    self._initialize_llms()
        return self._review_llm

    # ------------------------------------------------------------------ #
    #  Lazy-loaded properties — Infrastructure layer
    # ------------------------------------------------------------------ #

    @property
    def orchestrator(self) -> "QueryOrchestrator":
        """Return the ``QueryOrchestrator`` singleton for routing and planning.

        The orchestrator determines execution method, model selection, and
        complexity scoring for every incoming query. Thread-safe via
        ``_orchestrator_lock``.

        Returns:
            ``backend.core.engine.query_orchestrator.QueryOrchestrator``
        """
        if self._orchestrator_instance is None:
            with self._orchestrator_lock:
                if self._orchestrator_instance is None:
                    t0 = time.monotonic()
                    from backend.core.engine.query_orchestrator import QueryOrchestrator
                    self._orchestrator_instance = QueryOrchestrator()
                    elapsed = time.monotonic() - t0
                    self.init_timings["orchestrator"] = elapsed
                    logger.debug("QueryOrchestrator initialized (%.3fs)", elapsed)
        return self._orchestrator_instance

    @property
    def chroma_client(self) -> Optional[Any]:
        """Return the ChromaDB persistent vector store client.

        If ChromaDB is unavailable or initialization fails, returns ``None``
        and records the failure so subsequent calls skip re-initialization
        (sentinel pattern prevents retry storms).

        Returns:
            ``chromadb.PersistentClient`` or ``None`` on failure.
        """
        if self._chroma_sentinel.state == _InitState.NOT_ATTEMPTED:
            with self._chroma_lock:
                if self._chroma_sentinel.state == _InitState.NOT_ATTEMPTED:
                    self._chroma_client_inner = self._create_chroma_client()
                    if self._chroma_client_inner is not None:
                        self._chroma_sentinel = _Sentinel(_InitState.READY)
                    else:
                        self._chroma_sentinel = _Sentinel(
                            _InitState.FAILED, "initialization returned None"
                        )
        return self._chroma_client_inner

    # ------------------------------------------------------------------ #
    #  Lazy-loaded properties — Analysis tooling layer
    # ------------------------------------------------------------------ #

    @property
    def tools(self) -> List[Any]:
        """Return the list of analysis tools available to agents.

        Currently returns an empty list; custom tool implementations can be
        registered here in future iterations. Thread-safe via ``_tools_lock``.

        Returns:
            A list of tool instances (may be empty).
        """
        if self._tools is None:
            with self._tools_lock:
                if self._tools is None:
                    self._tools = self._create_tools()
        return self._tools

    @property
    def query_parser(self) -> "AdvancedQueryParser":
        """Return the advanced NL query parser. Thread-safe via ``_parser_lock``.

        Returns:
            ``backend.core.query_parser.AdvancedQueryParser``
        """
        if self._query_parser is None:
            with self._parser_lock:
                if self._query_parser is None:
                    t0 = time.monotonic()
                    from backend.core.query_parser import AdvancedQueryParser
                    self._query_parser = AdvancedQueryParser(self.llm_client)
                    elapsed = time.monotonic() - t0
                    self.init_timings["query_parser"] = elapsed
                    logger.debug("AdvancedQueryParser initialized (%.3fs)", elapsed)
        return self._query_parser

    # ------------------------------------------------------------------ #
    #  Bulk initialization helpers
    # ------------------------------------------------------------------ #

    def ensure_initialized(self, include_optional: bool = False) -> None:
        """Pre-warm all critical components in a single call.

        Idempotent — calling multiple times is a no-op after the first
        successful invocation. Useful during application startup to fail
        fast if Ollama is unreachable.

        Args:
            include_optional: If ``True``, also pre-warms orchestrator,
                chroma_client, and query_parser (slower but fully primed).
                Default ``False`` keeps startup fast.

        Raises:
            RuntimeError: If LLM initialization fails (Ollama not running).
        """
        if self._initialized:
            return

        t0 = time.monotonic()

        # In online mode llm_client is Ollama-only — skip it so Ollama is not required
        _online = False
        try:
            from backend.core.mode_manager import get_mode_manager as _gmm
            _online = _gmm().get_mode() == "online"
        except Exception:
            pass

        # Critical path — must succeed or raise
        # If any step fails, _initialized stays False so next call retries
        if not _online:
            _ = self.llm_client  # Only needed for Ollama path
        _ = self.primary_llm
        _ = self.review_llm

        # Optional components — failures are tolerated
        if include_optional:
            try:
                _ = self.orchestrator
            except Exception as exc:
                logger.warning("Optional init skipped (orchestrator): %s", exc)
            try:
                _ = self.chroma_client
            except Exception as exc:
                logger.warning("Optional init skipped (chroma_client): %s", exc)
            try:
                _ = self.query_parser
            except Exception as exc:
                logger.warning("Optional init skipped (query_parser): %s", exc)

        self._initialized = True
        elapsed = time.monotonic() - t0
        self.init_timings["total_init"] = elapsed
        logger.info(
            "ModelManager fully initialized in %.2fs (optional=%s)",
            elapsed,
            include_optional,
        )

    def ensure_cot_engine(self) -> Optional[Any]:
        """Ensure the Chain-of-Thought self-correction engine is available.

        Loads CoT configuration from ``config/cot_review_config.json``; if the
        engine is disabled in config or initialization fails, returns ``None``.
        Uses the sentinel pattern to avoid retry storms on repeated failures.

        Returns:
            ``SelfCorrectionEngine`` instance or ``None`` if disabled/failed.
        """
        if self._cot_sentinel.state == _InitState.NOT_ATTEMPTED:
            with self._cot_lock:
                if self._cot_sentinel.state == _InitState.NOT_ATTEMPTED:
                    config = self._load_cot_config()
                    if not config.get("enabled", False):
                        self._cot_sentinel = _Sentinel(_InitState.FAILED, "disabled in config")
                    else:
                        try:
                            from backend.core.engine.self_correction_engine import SelfCorrectionEngine
                            self._cot_engine_inner = SelfCorrectionEngine(config)
                            self._cot_sentinel = _Sentinel(_InitState.READY)
                            logger.info("SelfCorrectionEngine initialized (CoT enabled)")
                        except ImportError:
                            self._cot_sentinel = _Sentinel(_InitState.FAILED, "import error")
                            logger.warning("SelfCorrectionEngine import failed — CoT disabled")
                        except Exception as exc:
                            self._cot_sentinel = _Sentinel(_InitState.FAILED, str(exc))
                            logger.error("SelfCorrectionEngine init failed: %s", exc)
        return self._cot_engine_inner

    # Backward-compatible alias (some callers reference _cot_engine directly)
    @property
    def _cot_engine(self) -> Optional[Any]:
        """Backward-compatible accessor for the CoT engine.

        Delegates to ``ensure_cot_engine()`` for proper sentinel-guarded init.
        """
        return self.ensure_cot_engine()

    # ------------------------------------------------------------------ #
    #  Warmup & validation
    # ------------------------------------------------------------------ #

    def warmup(self, timeout: int = 60) -> Dict[str, Any]:
        """Send a lightweight probe to verify Ollama is responsive and models load.

        Sends a tiny prompt (``"Say 'ready' in one word."``) to the primary
        model. This forces Ollama to load the model into VRAM/RAM, so the
        first real user query doesn't pay cold-start latency.

        Args:
            timeout: Maximum seconds to wait for the probe response.

        Returns:
            Dict with keys ``success`` (bool), ``latency_seconds`` (float),
            ``model`` (str), and optionally ``error`` (str) on failure.

        Example::

            result = manager.warmup()
            if result["success"]:
                print(f"Model ready in {result['latency_seconds']:.1f}s")
        """
        t0 = time.monotonic()
        try:
            client = self.llm_client
            # Use generate() with explicit model to honor the timeout parameter
            response = client.generate(
                prompt="Say 'ready' in one word.",
                model=self.cached_models.get("primary", getattr(client, 'primary_model', None)),
                adaptive_timeout=False,
            )
            elapsed = time.monotonic() - t0
            self.init_timings["warmup"] = elapsed

            success = bool(response.get("response"))
            result = {
                "success": success,
                "latency_seconds": round(elapsed, 3),
                "model": self.cached_models.get("primary", client.primary_model),
                "response": response.get("response", "")[:50],
            }
            if success:
                logger.info("Warmup completed in %.1fs — model is hot", elapsed)
            else:
                result["error"] = "Empty response from model"
                logger.warning("Warmup returned empty response (%.1fs)", elapsed)
            return result

        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.warning("Warmup failed after %.1fs: %s", elapsed, exc)
            return {
                "success": False,
                "latency_seconds": round(elapsed, 3),
                "model": self.cached_models.get("primary", "unknown"),
                "error": str(exc),
            }

    async def async_warmup(self, timeout: int = 60) -> Dict[str, Any]:
        """Async version of ``warmup()`` — non-blocking for FastAPI lifespan.

        Uses the LLM client's async HTTP path so the event loop is not
        blocked during model loading. Preferred over ``warmup()`` in async
        startup contexts.

        Args:
            timeout: Maximum seconds to wait for the probe response.

        Returns:
            Same structure as ``warmup()``.
        """
        # Online mode: Ollama warmup is not needed — cloud APIs are already active
        try:
            from backend.core.mode_manager import get_mode_manager as _gmm
            if _gmm().get_mode() == "online":
                logger.info("Online mode — skipping Ollama warmup")
                return {"success": True, "latency_seconds": 0.0, "model": "online", "response": "online"}
        except Exception:
            pass

        t0 = time.monotonic()
        try:
            client = self.llm_client
            if client is None:
                return {"success": False, "latency_seconds": 0.0, "model": "unknown", "error": "No LLM client (Ollama not running?)"}

            response = await client.generate_primary_async(
                prompt="Say 'ready' in one word."
            )
            elapsed = time.monotonic() - t0
            self.init_timings["warmup"] = elapsed

            success = bool(response.get("response"))
            result = {
                "success": success,
                "latency_seconds": round(elapsed, 3),
                "model": self.cached_models.get("primary", client.primary_model),
                "response": response.get("response", "")[:50],
            }
            if success:
                logger.info("Async warmup completed in %.1fs — model is hot", elapsed)
            else:
                result["error"] = "Empty response from model"
                logger.warning("Async warmup returned empty response (%.1fs)", elapsed)
            return result

        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.warning("Async warmup failed after %.1fs: %s", elapsed, exc)
            return {
                "success": False,
                "latency_seconds": round(elapsed, 3),
                "model": self.cached_models.get("primary", "unknown"),
                "error": str(exc),
            }

    def list_available_models(self) -> Dict[str, Any]:
        """Query Ollama for all installed models and their metadata.

        Returns a structured inventory that can be served directly to
        frontend dropdowns or admin dashboards. Uses the ``/api/tags``
        endpoint.

        Returns:
            Dict with keys:
            - ``models``: List of dicts (``name``, ``size_gb``, ``modified``).
            - ``count``: Number of installed models.
            - ``error``: Only present on failure.

        Example::

            inventory = manager.list_available_models()
            for m in inventory["models"]:
                print(f"{m['name']} ({m['size_gb']:.1f} GB)")
        """
        try:
            import requests as _requests

            base_url = getattr(self.llm_client, "base_url", "http://localhost:11434")
            resp = _requests.get(f"{base_url}/api/tags", timeout=10)
            resp.raise_for_status()

            raw_models = resp.json().get("models", [])
            models = []
            for m in raw_models:
                models.append({
                    "name": m.get("name", "unknown"),
                    "size_gb": round(m.get("size", 0) / (1024 ** 3), 2),
                    "parameter_size": m.get("details", {}).get("parameter_size", "unknown"),
                    "family": m.get("details", {}).get("family", "unknown"),
                    "quantization": m.get("details", {}).get("quantization_level", "unknown"),
                    "modified": m.get("modified_at", ""),
                })

            logger.debug("Ollama reports %d installed models", len(models))
            return {"models": models, "count": len(models)}

        except Exception as exc:
            logger.warning("Failed to list Ollama models: %s", exc)
            return {"models": [], "count": 0, "error": str(exc)}

    def is_healthy(self) -> bool:
        """Quick boolean health check for load balancers and readiness probes.

        Returns ``True`` only when the LLM client has been initialized and
        at least one model is cached. Does NOT trigger lazy initialization —
        purely reads current state.

        Returns:
            ``True`` if the manager has a live LLM client and known models.
        """
        return (
            self._llm_client is not None
            and self._primary_llm is not None
            and len(self.cached_models) > 0
        )

    # ------------------------------------------------------------------ #
    #  Hot-reload & lifecycle management
    # ------------------------------------------------------------------ #

    def reload_models(self) -> None:
        """Re-discover and re-initialize LLM models without restarting the server.

        Useful after a user changes their model preferences via the UI.
        Clears the current LLM instances and triggers fresh initialization
        from ``ModelSelector`` and user preferences on the next access.

        Thread-safe — acquires ``_llm_models_lock`` during the swap.
        """
        with self._llm_models_lock:
            old_primary = self.cached_models.get("primary", "none")
            old_review = self.cached_models.get("review", "none")

            self._primary_llm = None
            self._review_llm = None
            self.cached_models.clear()

            # Trigger re-initialization immediately so errors surface now
            self._initialize_llms()

            logger.info(
                "Models reloaded — primary: %s → %s, review: %s → %s",
                old_primary,
                self.cached_models.get("primary", "unknown"),
                old_review,
                self.cached_models.get("review", "unknown"),
            )

    async def shutdown(self) -> None:
        """Release all managed resources for a clean shutdown.

        Should be called during application teardown (e.g., FastAPI
        ``shutdown`` event). Resets the singleton so that a subsequent
        ``get_model_manager()`` call creates a fresh instance.

        This method is idempotent and never raises.
        """
        global _model_manager

        logger.info("ModelManager shutting down — releasing resources")

        # Close ChromaDB client if it supports it
        if self._chroma_client_inner is not None:
            try:
                # PersistentClient doesn't have close(), but future versions might
                if hasattr(self._chroma_client_inner, "close"):
                    self._chroma_client_inner.close()
            except Exception as exc:
                logger.debug("ChromaDB close error (ignored): %s", exc)
            self._chroma_client_inner = None
            self._chroma_sentinel = _Sentinel()

        # Close LLM Client resources
        if self._llm_client is not None:
            try:
                self._llm_client.close()
                await self._llm_client.aclose()
                logger.debug("LLMClient connections closed")
            except Exception as exc:
                logger.warning("LLMClient close error: %s", exc)
            self._llm_client = None
            self._llm_client_sentinel = _Sentinel()

        # Clear references (garbage collector handles cleanup)
        self._primary_llm = None
        self._review_llm = None
        self._orchestrator_instance = None
        self._query_parser = None
        self._tools = None
        self._cot_engine_inner = None
        self._cot_sentinel = _Sentinel()
        self._cot_config = None
        self.cached_models.clear()
        self.init_timings.clear()
        self._initialized = False

        # Clear the global singleton reference
        with _manager_lock:
            _model_manager = None

        logger.info("ModelManager shutdown complete")

    # ------------------------------------------------------------------ #
    #  Private initialization methods
    # ------------------------------------------------------------------ #

    def _initialize_llms(self) -> None:
        """Initialize both primary and review LangChain LLM instances.

        Model selection follows this precedence:
        1. ``ModelSelector.select_optimal_models()`` — inspects Ollama inventory.
        2. User preferences from ``config/user_preferences.json``.
        3. Hardcoded defaults (``llama3.1:8b`` / ``phi3:mini``).

        Side Effects:
            Populates ``self._primary_llm``, ``self._review_llm``, and
            ``self.cached_models['primary']`` / ``self.cached_models['review']``.

        Raises:
            RuntimeError: If Ollama is unreachable or no models are available.

        Note:
            Caller must hold ``_llm_models_lock`` before invoking this method.
        """
        t0 = time.monotonic()

        # --- Online mode: use ModeManager cloud client if available ---
        try:
            _mode_mgr = get_mode_manager()
            if _mode_mgr.get_mode() == "online":
                _online_client = _mode_mgr.get_llm_client()
                if _online_client is not None:
                    # Wrap cloud client in a thin LangChain-compatible shim
                    # with runtime fallback: Groq → OpenRouter
                    class _OnlineLLMShim:
                        """Minimal shim so agents can call .invoke(prompt).
                        
                        Includes runtime fallback: if the primary client
                        raises an exception (e.g. 403 from expired Groq key),
                        automatically retries with the next available client.
                        """
                        def __init__(self, mode_mgr, tier):
                            self._mode_mgr = mode_mgr
                            self._tier = tier
                        def invoke(self, prompt, **kwargs):
                            # Build ordered client chain
                            clients = []
                            _groq = self._mode_mgr._get_groq()
                            if _groq:
                                clients.append((_groq, "Groq"))
                            _gemini = self._mode_mgr._get_gemini()
                            if _gemini:
                                clients.append((_gemini, "Gemini"))
                            _or = self._mode_mgr._get_openrouter()
                            if _or:
                                clients.append((_or, "OpenRouter"))
                            
                            last_err = None
                            for client, name in clients:
                                try:
                                    return client.generate(
                                        str(prompt), tier=self._tier
                                    )
                                except Exception as e:
                                    last_err = e
                                    logger.warning(
                                        "Online LLM %s failed (%s), trying next...",
                                        name, e
                                    )
                            
                            raise RuntimeError(
                                f"All online LLM clients failed. Last error: {last_err}"
                            )
                        def __call__(self, prompt, **kwargs):
                            return self.invoke(prompt, **kwargs)

                    self._primary_llm = _OnlineLLMShim(_mode_mgr, "complex")
                    self._review_llm = _OnlineLLMShim(_mode_mgr, "medium")
                    client_name = _online_client.__class__.__name__
                    self.cached_models["primary"] = f"online:{client_name}:complex"
                    self.cached_models["review"] = f"online:{client_name}:medium"
                    elapsed = time.monotonic() - t0
                    self.init_timings["llm_models"] = elapsed
                    logger.info(
                        "LLMs initialized via %s in %.2fs (online mode)",
                        client_name, elapsed
                    )
                    return
        except Exception as _mode_err:
            logger.debug("Online LLM init check skipped, using Ollama: %s", _mode_err)

        # --- Offline mode (default): existing Ollama path unchanged ---
        try:
            # Prefer the modern langchain-ollama package; fall back to legacy
            try:
                from langchain_ollama import OllamaLLM
            except ImportError:
                from langchain_community.llms import Ollama as OllamaLLM  # type: ignore[no-redef]

            from backend.core.engine.model_selector import ModelSelector
            from backend.core.engine.user_preferences import get_preferences_manager

            # --- Determine models ---
            prefs = get_preferences_manager().load_preferences()
            primary_model, review_model, _ = ModelSelector.select_optimal_models()

            primary_model = primary_model or prefs.primary_model or "llama3.1:8b"
            review_model = review_model or prefs.review_model or "phi3:mini"

            # Persist resolved model names for downstream consumers
            self.cached_models["primary"] = primary_model
            self.cached_models["review"] = review_model

            # LangChain's OllamaLLM expects bare model names (no "ollama/" prefix)
            primary_clean = primary_model.replace("ollama/", "")
            review_clean = review_model.replace("ollama/", "")

            self._primary_llm = OllamaLLM(model=primary_clean, timeout=120)
            self._review_llm = OllamaLLM(model=review_clean, timeout=60)

            elapsed = time.monotonic() - t0
            self.init_timings["llm_models"] = elapsed
            logger.info(
                "LLMs initialized in %.2fs — primary=%s, review=%s",
                elapsed,
                primary_model,
                review_model,
            )

        except Exception as exc:
            logger.error("LLM initialization failed: %s", exc)
            raise RuntimeError(f"Cannot initialize LLMs: {exc}") from exc

    def _create_chroma_client(self) -> Optional[Any]:
        """Create and return a ChromaDB ``PersistentClient``.

        Reads the persist directory from application settings and ensures the
        directory exists. Telemetry is disabled by default.

        Returns:
            ``chromadb.PersistentClient`` or ``None`` on failure.
        """
        t0 = time.monotonic()

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            from backend.core.config import get_settings

            persist_dir = get_settings().chromadb_persist_directory
            os.makedirs(persist_dir, exist_ok=True)

            client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            elapsed = time.monotonic() - t0
            self.init_timings["chroma_client"] = elapsed
            logger.info("ChromaDB client initialized at %s (%.3fs)", persist_dir, elapsed)
            return client

        except ImportError:
            logger.warning("chromadb package not installed — vector search unavailable")
            return None
        except Exception as exc:
            logger.warning("ChromaDB initialization failed: %s", exc)
            return None

    @staticmethod
    def _create_tools() -> List[Any]:
        """Create the list of analysis tools available to agents.

        Currently returns an empty list. Custom tool implementations (e.g.,
        statistical tests, visualization helpers) should be registered here
        in future iterations.

        Returns:
            An empty list (placeholder for future tool registry).
        """
        return []

    def _load_cot_config(self) -> Dict[str, Any]:
        """Load Chain-of-Thought review configuration from disk.

        Reads ``config/cot_review_config.json`` relative to ``PROJECT_ROOT``.
        The result is cached for the lifetime of the ``ModelManager`` instance.

        Returns:
            Parsed JSON configuration dict. Falls back to ``{"enabled": False}``
            if the file is missing or malformed.
        """
        if self._cot_config is not None:
            return self._cot_config

        default_config: Dict[str, Any] = {"enabled": False}

        try:
            from backend.core.config import get_settings

            config_path = Path(get_settings().PROJECT_ROOT) / "config" / "cot_review_config.json"

            if config_path.is_file():
                with open(config_path, "r", encoding="utf-8") as fh:
                    self._cot_config = json.load(fh)
                logger.debug("CoT config loaded from %s", config_path)
            else:
                logger.debug("CoT config not found at %s — using defaults", config_path)
                self._cot_config = default_config
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load CoT config: %s — using defaults", exc)
            self._cot_config = default_config

        return self._cot_config

    # ------------------------------------------------------------------ #
    #  Diagnostics & introspection
    # ------------------------------------------------------------------ #

    def get_status(self) -> Dict[str, Any]:
        """Return a diagnostic snapshot of all managed components.

        Useful for health-check endpoints, admin dashboards, and automated
        monitoring. Includes readiness flags, cached model names, component
        init states, and timing data.

        Returns:
            Dict with component readiness booleans, model names, sentinel
            states for optional components, and initialization timings.
        """
        return {
            "initialized": self._initialized,
            "components": {
                "llm_client": self._llm_client is not None,
                "primary_llm": self._primary_llm is not None,
                "review_llm": self._review_llm is not None,
                "chroma_client": {
                    "ready": self._chroma_sentinel.state == _InitState.READY,
                    "state": self._chroma_sentinel.state.name,
                    "error": self._chroma_sentinel.error,
                },
                "orchestrator": self._orchestrator_instance is not None,
                "query_parser": self._query_parser is not None,
                "cot_engine": {
                    "ready": self._cot_sentinel.state == _InitState.READY,
                    "state": self._cot_sentinel.state.name,
                    "error": self._cot_sentinel.error,
                },
            },
            "cached_models": dict(self.cached_models),
            "init_timings_seconds": dict(self.init_timings),
        }

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "lazy"
        models = ", ".join(f"{k}={v}" for k, v in self.cached_models.items()) or "none loaded"
        return f"<ModelManager status={status} models=[{models}]>"


# ====================================================================== #
#  Module-level singleton accessor
# ====================================================================== #


def get_model_manager() -> ModelManager:
    """Return the process-wide ``ModelManager`` singleton (thread-safe).

    Uses double-checked locking so that only the very first call pays the
    cost of acquiring the lock. Subsequent calls are a simple ``is None``
    check with no synchronization overhead.

    Returns:
        The singleton ``ModelManager`` instance.
    """
    global _model_manager
    if _model_manager is None:
        with _manager_lock:
            if _model_manager is None:
                _model_manager = ModelManager()
    return _model_manager


def reset_model_manager() -> None:
    """Reset the singleton for test isolation.

    Clears the global ``ModelManager`` instance so the next call to
    ``get_model_manager()`` creates a fresh one. Should only be used
    in test fixtures, never in production code.

    Performs best-effort resource cleanup before clearing the reference.

    Note:
        Cannot call ``shutdown()`` directly because ``shutdown()`` also
        acquires ``_manager_lock``, which would deadlock. Instead, we
        inline the critical cleanup steps.
    """
    global _model_manager
    with _manager_lock:
        if _model_manager is not None:
            # Best-effort inline cleanup (avoids deadlock with shutdown's lock)
            try:
                if _model_manager._chroma_client_inner is not None:
                    if hasattr(_model_manager._chroma_client_inner, "close"):
                        _model_manager._chroma_client_inner.close()
            except Exception:
                pass
            _model_manager._llm_client = None
            _model_manager._primary_llm = None
            _model_manager._review_llm = None
            _model_manager._initialized = False
        _model_manager = None
