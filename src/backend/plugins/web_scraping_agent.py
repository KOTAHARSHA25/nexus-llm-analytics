"""Web Scraping Agent Plugin — Nexus LLM Analytics
==================================================

Implements web content retrieval via a Firecrawl → Jina → local fallback
chain in ONLINE mode, and the built-in local scraper in OFFLINE mode.

The agent is registered automatically by the AgentRegistry auto-discovery
scan because it resides in the ``plugins/`` directory and extends
``BasePluginAgent``.

Offline mode uses ``urllib`` + a lightweight HTML stripper so there are
zero new dependencies for the default configuration.

Online mode (when ModeManager.get_mode() == "online"):
    1st → Firecrawl API (clean markdown, full JS rendering)
    2nd → Jina Reader  (https://r.jina.ai/<url>, no key required)
    3rd → Local scraper (offline path, always available)
"""

from __future__ import annotations

import logging
import re
import urllib.request
import urllib.error
import urllib.parse
from typing import Any, Dict, List, Optional

from backend.core.plugin_system import BasePluginAgent, AgentMetadata, AgentCapability
from backend.core.mode_manager import get_mode_manager

logger = logging.getLogger(__name__)


def _local_scrape(url: str, timeout: int = 15) -> str:
    """Minimal local HTTP scraper — fetches URL and strips HTML tags.

    Falls back gracefully on any network error.
    """
    try:
        headers = {"User-Agent": "NexusLLM-Bot/2.0"}
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        # Strip HTML tags (good enough for plain text extraction)
        text = re.sub(r"<[^>]+>", " ", raw)
        # Collapse whitespace
        text = re.sub(r"\s{2,}", "\n", text).strip()
        return text[:50_000]  # Cap at 50k chars to stay within context window
    except Exception as e:
        logger.warning("[WebScrapingAgent] Local scrape failed for %s: %s", url, e)
        return ""


class WebScrapingAgent(BasePluginAgent):
    """Web scraping plugin with online/offline mode support.

    In OFFLINE mode, uses the built-in local scraper (urllib + HTML strip).
    In ONLINE mode, chains Firecrawl → Jina → local scraper as fallbacks.
    """

    @classmethod
    def get_metadata(cls) -> AgentMetadata:
        return AgentMetadata(
            name="WebScrapingAgent",
            version="2.0.0",
            description=(
                "Retrieves and cleans web content. "
                "Online mode: Firecrawl → Jina → local. "
                "Offline mode: local only."
            ),
            author="Nexus Team",
            capabilities=[AgentCapability.WEB_SCRAPING],
            file_types=[],
            dependencies=[],
            min_ram_mb=128,
            max_timeout_seconds=60,
            priority=60,
        )

    def initialize(self) -> bool:
        """Initialize the agent — always succeeds (no heavy resources required)."""
        self.initialized = True
        return True

    # ------------------------------------------------------------------
    # Main scraping entry-point
    # ------------------------------------------------------------------

    def scrape_url(self, url: str) -> str:
        """Scrape a URL and return clean text/markdown content.

        Respects the current ModeManager mode:
        - ONLINE  → Firecrawl → Jina → local fallback
        - OFFLINE → local scraper only

        Args:
            url: The URL to scrape.

        Returns:
            Scraped content as a string. Empty string on total failure.
        """
        try:
            mode_mgr = get_mode_manager()
            if mode_mgr.get_mode() == "online":
                return self._scrape_online(url)
        except Exception as e:
            logger.debug("[WebScrapingAgent] Mode check failed, using local: %s", e)

        return _local_scrape(url)

    def _scrape_online(self, url: str) -> str:
        """Try Firecrawl → Jina → local in sequence."""
        mode_mgr = get_mode_manager()

        # 1st: Firecrawl
        try:
            scraper = mode_mgr.get_scraper_client()
            if scraper is not None and scraper.__class__.__name__ == "FirecrawlClient":
                content = scraper.scrape(url)
                if content:
                    logger.debug("[WebScrapingAgent] Firecrawl succeeded for %s", url)
                    return content
        except Exception as e:
            logger.warning(
                "[ModeManager] FirecrawlClient failed, falling back to Jina. Error: %s", e
            )

        # 2nd: Jina
        try:
            from backend.core.online_clients import JinaClient
            jina = JinaClient()
            content = jina.scrape(url)
            if content:
                logger.debug("[WebScrapingAgent] Jina succeeded for %s", url)
                return content
        except Exception as e:
            logger.warning(
                "[ModeManager] JinaClient failed, falling back to local scraper. Error: %s", e
            )

        # 3rd: local
        logger.info("[WebScrapingAgent] Using local scraper fallback for %s", url)
        return _local_scrape(url)

    # ------------------------------------------------------------------
    # BasePluginAgent interface
    # ------------------------------------------------------------------

    def execute(self, query: str, data: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """Execute a web-scraping query.

        Args:
            query: A URL to scrape, or natural-language query containing a URL.
            data:  Unused for web scraping.
            **kwargs: Optional ``url`` key to override URL extraction from query.

        Returns:
            Dict with ``result``, ``status``, and ``metadata`` keys.
        """
        # Extract URL from kwargs or from the query string itself
        url: Optional[str] = kwargs.get("url") or _extract_url(query)

        if not url:
            return {
                "result": "No URL found in the query. Please provide a URL to scrape.",
                "status": "error",
                "metadata": {"agent": "WebScrapingAgent", "error": "no_url"},
            }

        try:
            content = self.scrape_url(url)
            if not content:
                return {
                    "result": "Analysis temporarily unavailable. Please try again.",
                    "status": "error",
                    "metadata": {"agent": "WebScrapingAgent", "url": url, "error": "empty_response"},
                }

            # Hand content to LLM if a question was provided alongside the URL
            answer = content
            if query and url not in query.strip():
                # There's a question beyond just the URL
                answer = self._summarize_with_llm(query, content)

            return {
                "result": answer,
                "status": "success",
                "metadata": {
                    "agent": "WebScrapingAgent",
                    "url": url,
                    "content_length": len(content),
                    "mode": get_mode_manager().get_mode(),
                },
            }
        except Exception as e:
            logger.error("[WebScrapingAgent] execute() failed for %s: %s", url, e)
            return {
                "result": "Analysis temporarily unavailable. Please try again.",
                "status": "error",
                "metadata": {"agent": "WebScrapingAgent", "url": url, "error": str(e)},
            }

    def _summarize_with_llm(self, query: str, content: str) -> str:
        """Ask the active LLM to answer *query* using *content* as context."""
        try:
            from backend.agents.model_manager import get_model_manager
            llm = get_model_manager().primary_llm
            prompt = (
                f"Based on the following web content, answer this question: {query}\n\n"
                f"Web content:\n{content[:8000]}\n\nAnswer:"
            )
            return str(llm.invoke(prompt))
        except Exception as e:
            logger.warning("[WebScrapingAgent] LLM summarization failed: %s", e)
            return content[:5000]

    def reflective_execute(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Reflective execution wrapper — delegates to execute()."""
        return self.execute(query, **(context or {}))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _extract_url(text: str) -> Optional[str]:
    """Extract the first HTTP/HTTPS URL from *text*, or return None."""
    match = re.search(r"https?://[^\s\"'<>]+", text)
    return match.group(0) if match else None
