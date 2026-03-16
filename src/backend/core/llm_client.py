"""
LLM Client — Nexus LLM Analytics v2.0
=====================================

Synchronous and asynchronous communication with the local Ollama LLM
server.  Supports circuit-breaker protection, adaptive timeouts based
on model size and available RAM, and real-time token streaming.

Enterprise v2.0 Additions
-------------------------
* **LLMClientPool** — Connection-pool wrapper that maintains multiple
  :class:`LLMClient` instances for concurrent request throughput.
* **RequestTracer** — Lightweight span tracer that records prompt
  size, response size, latency, and model for every LLM call.
* ``get_llm_client()`` — Thread-safe singleton accessor.

Backward Compatibility
----------------------
:class:`LLMClient` retains its full v1.x interface including
``generate``, ``generate_primary``, ``generate_review``,
``generate_async``, ``stream_generate``, etc.

.. versionchanged:: 2.0
   Added pool, tracer, and singleton accessor.

Author: Nexus Analytics Research Team
Date: February 2026
"""

# Handles communication with local Ollama LLM server

import requests
import httpx
import asyncio
import logging
import json
import threading
import time as _time
from typing import Dict, Any, List, Optional, Union, AsyncGenerator

class LLMClient:
    """Client for communicating with a local Ollama LLM server.

    Provides both synchronous (:meth:`generate`) and async
    (:meth:`generate_async`, :meth:`stream_generate`) interfaces with
    built-in circuit-breaker protection and RAM-aware adaptive
    timeouts.

    Args:
        base_url: Ollama HTTP endpoint (default ``http://localhost:11434``).
        primary_model: Model used for generation.  ``None`` triggers
            automatic selection via :class:`ModelSelector`.
        review_model: Model used for critic / review steps.

    Example::

        client = LLMClient()
        result = client.generate("Explain gradient descent in 3 lines.")
        print(result["response"])

    .. versionchanged:: 2.0
       Added enterprise pool and tracer wrappers.
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434", 
        primary_model: Optional[str] = None, 
        review_model: Optional[str] = None
    ) -> None:
        from backend.core.engine.model_selector import ModelSelector
        
        self.base_url = base_url
        
        # Use intelligent model selection if models not provided
        if primary_model is None or review_model is None:
            selected_primary, selected_review, _ = ModelSelector.select_optimal_models()
            # Remove 'ollama/' prefix for API calls
            self.primary_model = selected_primary.replace("ollama/", "") if primary_model is None else primary_model
            self.review_model = selected_review.replace("ollama/", "") if review_model is None else review_model
        else:
            self.primary_model = primary_model
            self.review_model = review_model
            
        # [OPTIMIZATION 4.1] Connection pooling
        self._session = requests.Session()
        self._async_client = httpx.AsyncClient(timeout=300.0, limits=httpx.Limits(max_keepalive_connections=10, max_connections=20))
        # Cache for timeouts to avoid recalculating every request
        self._timeout_cache = {}
        self._last_timeout_update = 0
            
        logging.debug(f"LLMClient initialized - Primary: {self.primary_model}, Review: {self.review_model}")

    def generate(
        self, 
        prompt: str, 
        model: Optional[str] = None, 
        system: Optional[str] = None,
        adaptive_timeout: bool = True
    ) -> Dict[str, Any]:
        from backend.infra.circuit_breaker import get_circuit_breaker, CircuitBreakerConfig
        
        if model is None:
            model = self.primary_model
        
        # Check if model is available before making call
        if "model_not_available" in model:
            return {
                "model": model,
                "prompt": prompt,
                "response": "Model not available. Please install a compatible model.",
                "error": "No suitable models are installed. Please run: ollama pull tinyllama (for low memory) or ollama pull phi3:mini (for better performance)",
                "user_action_required": True,
                "suggestions": [
                    "Install tinyllama for low memory systems: ollama pull tinyllama",
                    "Install phi3:mini for better performance: ollama pull phi3:mini", 
                    "Install llama3.1:8b for best results (requires 8GB+ RAM): ollama pull llama3.1:8b"
                ]
            }
        
        # Use circuit breaker for LLM calls
        cb_config = CircuitBreakerConfig(
            failure_threshold=2,  # Open after 2 failures
            recovery_timeout=30.0,  # Try again after 30 seconds
            success_threshold=1,   # Close after 1 success
            timeout=self._calculate_adaptive_timeout(model) if adaptive_timeout else 300
        )
        
        circuit_breaker = get_circuit_breaker(f"llm_{model}", cb_config)
        
        def _make_llm_call():
            # Adaptive timeout based on model and system resources
            timeout = cb_config.timeout
            
            url = f"{self.base_url}/api/generate"
            payload = {"model": model, "prompt": prompt, "stream": False}
            if system:
                payload["system"] = system
            
            # [ADDED] Log input for visibility (redact PII - only log metadata)
            logging.info(f"\n{'='*20} [LLM INPUT] {'='*20}\nModel: {model}\nPrompt Length: {len(prompt)} chars\n{'='*52}")
            
            # [OPTIMIZATION 4.1] Use persistent session
            response = self._session.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            
            data = response.json()
            if "response" in data:
                resp_text = data["response"].strip()
                # [ADDED] Log output metadata (redact content to avoid PII leaks)
                logging.info(f"\n{'='*20} [LLM OUTPUT] {'='*20}\nResponse Length: {len(resp_text)} chars\n{'='*53}")
                return {"model": model, "prompt": prompt, "response": resp_text, "success": True}
            else:
                raise Exception("Empty response from LLM")
        
        # Execute with circuit breaker protection
        result = circuit_breaker.call(_make_llm_call)
        
        # Handle fallback responses
        if result.get("fallback_used"):
            logging.warning(f"Using fallback response for model {model}")
            return {
                "model": model, 
                "prompt": prompt, 
                "response": result.get("result", "Service unavailable"), 
                "error": result.get("error"),
                "fallback": True
            }
        
        return result
    
    def _calculate_adaptive_timeout(self, model: str) -> int:
        """Calculate timeout based on model requirements and system resources."""
        # [OPTIMIZATION 4.1] Cache timeout calculation for 60 seconds
        current_time = _time.time()
        if model in self._timeout_cache and (current_time - self._last_timeout_update < 60):
            return self._timeout_cache[model]

        try:
            from backend.core.engine.model_selector import ModelSelector
            import os
            
            # Get system memory info
            memory_info = ModelSelector.get_system_memory()
            available_ram = memory_info["available_gb"]
            
            # Base timeouts by model type
            base_timeouts: Dict[str, int] = {
                "llama3.1:8b": 600,    # 10 minutes for large model
                "phi3:mini": 300,      # 5 minutes for medium model  
                "tinyllama": 120,      # 2 minutes for small model
                "nomic-embed-text": 60 # 1 minute for embedding
            }
            
            # Clean model name
            clean_model = model.replace("ollama/", "")
            base_timeout = base_timeouts.get(clean_model, 300)
            
            # Check if using swap (low available RAM for model requirements)
            model_requirements = ModelSelector.MODEL_REQUIREMENTS.get(clean_model, {})
            required_ram = model_requirements.get("min_ram_gb", 2.0)
            
            # If using swap memory, increase timeout significantly
            from backend.core.config import get_settings
            settings = get_settings()
            allow_swap = settings.allow_swap_usage
            if allow_swap and available_ram < required_ram:
                # Using swap - increase timeout by 3x
                adaptive_timeout = base_timeout * 3
                logging.warning(f"🐌 Using swap memory - Extended timeout to {adaptive_timeout}s for {clean_model}")
            elif available_ram < required_ram + 1.0:  # Close to memory limit
                # Tight memory - increase timeout by 1.5x
                adaptive_timeout = int(base_timeout * 1.5)
                logging.warning(f"⚠️ Low memory - Extended timeout to {adaptive_timeout}s for {clean_model}")
            else:
                # Normal operation
                adaptive_timeout = base_timeout
            
            # Update cache
            self._timeout_cache[model] = adaptive_timeout
            self._last_timeout_update = current_time
                
            return adaptive_timeout
            
        except Exception as e:
            logging.warning(f"Failed to calculate adaptive timeout: {e}")
            return 300  # Default fallback

    def generate_primary(self, prompt: str) -> Dict[str, Any]:
        """Generate response using the primary model."""
        return self.generate(prompt, model=self.primary_model)

    def generate_review(self, prompt: str) -> Dict[str, Any]:
        """Generate response using the review model."""
        return self.generate(prompt, model=self.review_model)

    async def generate_async(
        self, 
        prompt: str, 
        model: Optional[str] = None, 
        system: Optional[str] = None,
        adaptive_timeout: bool = True
    ) -> Dict[str, Any]:
        """
        Async version of generate for non-blocking LLM calls.
        Use this in async endpoints for better throughput.
        """
        from backend.infra.circuit_breaker import get_circuit_breaker, CircuitBreakerConfig
        
        if model is None:
            model = self.primary_model
        
        # Check if model is available
        if "model_not_available" in model:
            return {
                "model": model,
                "prompt": prompt,
                "response": "Model not available. Please install a compatible model.",
                "error": "No suitable models are installed.",
                "user_action_required": True,
                "suggestions": [
                    "Install tinyllama for low memory systems: ollama pull tinyllama",
                    "Install phi3:mini for better performance: ollama pull phi3:mini", 
                    "Install llama3.1:8b for best results (requires 8GB+ RAM): ollama pull llama3.1:8b"
                ]
            }
        
        # Use circuit breaker for LLM calls
        cb_config = CircuitBreakerConfig(
            failure_threshold=2,  # Open after 2 failures
            recovery_timeout=30.0,  # Try again after 30 seconds
            success_threshold=1,   # Close after 1 success
            timeout=self._calculate_adaptive_timeout(model) if adaptive_timeout else 300
        )
        
        circuit_breaker = get_circuit_breaker(f"llm_{model}", cb_config)
        
        # Async wrapper for the circuit breaker (CircuitBreaker currently supports synchronous calls, 
        # so we wrap the specific async logic or use it for state check/fallback only)
        # Note: Ideally CircuitBreaker should support async calls, but for now we'll check state 
        # and manually update it or wrap in a sync function if the library supports it.
        # Since our recreated CircuitBreaker is simple, we will expand it or use it carefully.
        
        # Simple Async Implementation matching the synchronous pattern:
        # Check circuit breaker state before making the call
        if circuit_breaker.state.value == "OPEN":
             return {
                "fallback_used": True,
                "error": "Circuit is OPEN due to repeated failures",
                "result": "[!] Service temporarily unavailable. Please try again later.",
                "success": False
            }

        # Calculate timeout
        timeout = cb_config.timeout
        
        url = f"{self.base_url}/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        
        try:
            # [OPTIMIZATION 4.1] Use persistent async client
            # We use a per-request timeout here because the client's timeout is global/connect-only
            response = await self._async_client.post(url, json=payload, timeout=httpx.Timeout(timeout))
            response.raise_for_status()
            
            data = response.json()
            if "response" in data:
                circuit_breaker._handle_success()
                return {
                    "model": model, 
                    "prompt": prompt, 
                    "response": data["response"].strip(), 
                    "success": True
                }
            else:
                raise Exception("Empty response from LLM")
                    
        except httpx.TimeoutException:
            logging.warning(f"Async LLM call timed out after {timeout}s for model {model}")
            return {
                "model": model,
                "prompt": prompt,
                "response": "",
                "error": f"Request timed out after {timeout}s",
                "timeout": True
            }
        except Exception as e:
            logging.error(f"Async LLM call failed: {e}")
            return {
                "model": model,
                "prompt": prompt,
                "response": "",
                "error": str(e)
            }

    async def generate_primary_async(self, prompt: str) -> Dict[str, Any]:
        """Async generate using the primary model."""
        return await self.generate_async(prompt, model=self.primary_model)

    async def generate_review_async(self, prompt: str) -> Dict[str, Any]:
        """Async generate using the review model."""
        return await self.generate_async(prompt, model=self.review_model)

    async def stream_generate(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        system: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from Ollama in real-time.
        Yields individual tokens as they're generated by the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            model: Model to use (defaults to primary_model)
            system: Optional system prompt
            
        Yields:
            Individual tokens from the LLM response
        """
        if model is None:
            model = self.primary_model
            
        # Check if model is available
        if "model_not_available" in model:
            yield "[Error: Model not available. Please install a compatible model.]"
            return
        
        # Circuit breaker check for streaming
        from backend.infra.circuit_breaker import get_circuit_breaker, CircuitBreakerConfig
        cb_config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=30.0,
            success_threshold=1,
            timeout=self._calculate_adaptive_timeout(model)
        )
        circuit_breaker = get_circuit_breaker(f"llm_{model}", cb_config)
        
        if circuit_breaker.state.value == "OPEN":
            yield "[Error: Service temporarily unavailable due to repeated failures. Please try again in 30 seconds.]"
            return
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model, 
            "prompt": prompt, 
            "stream": True
        }
        if system:
            payload["system"] = system
        
        logging.info(f"\n{'='*20} [LLM STREAM START] {'='*20}\nModel: {model}\nPrompt Preview: {prompt[:200]}...\n{'='*52}")
        
        try:
            # Use longer timeout for streaming (up to 5 minutes)
            timeout = httpx.Timeout(300.0, connect=10.0)
            # [OPTIMIZATION 4.1] Use persistent async client
            async with self._async_client.stream("POST", url, json=payload, timeout=timeout) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            
                            # Ollama streams with this format:
                            # {"model": "...", "response": "token", "done": false}
                            if "response" in data:
                                token = data["response"]
                                if token:  # Only yield non-empty tokens
                                    yield token
                            
                            # Check if streaming is complete
                            if data.get("done", False):
                                logging.info("[LLM STREAM] Streaming complete")
                                break
                                
                        except json.JSONDecodeError:
                            # Skip malformed JSON lines
                            continue
                        except Exception as e:
                            logging.warning(f"Error parsing stream line: {e}")
                            continue
                                
        except httpx.TimeoutException:
            logging.error(f"Stream timeout for model {model}")
            circuit_breaker._handle_failure(Exception(f"Stream timeout for {model}"))
            yield "\n\n[Error: Request timed out. The model may be too slow for your system.]"
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error during streaming: {e}")
            circuit_breaker._handle_failure(e)
            yield f"\n\n[Error: HTTP {e.response.status_code} - {e}]"
        except Exception as e:
            logging.error(f"Streaming failed: {e}", exc_info=True)
            circuit_breaker._handle_failure(e)
            yield f"\n\n[Error: Streaming failed - {str(e)}]"

    def close(self) -> None:
        """Close synchronous HTTP resources (requests.Session)."""
        if hasattr(self, '_session') and self._session:
            try:
                self._session.close()
            except Exception:
                pass

    async def aclose(self) -> None:
        """Close asynchronous HTTP resources (httpx.AsyncClient)."""
        if hasattr(self, '_async_client') and self._async_client:
            try:
                await self._async_client.aclose()
            except Exception:
                pass


# ============================================================================
# Enterprise v2.0 — Pool, Tracer & Singleton
# ============================================================================


class RequestTracer:
    """Lightweight LLM request span tracer.

    Records prompt size, response size, latency, and model for every
    call.  Useful for cost monitoring and performance dashboards.

    Example::

        tracer = RequestTracer()
        tracer.record(model="phi3:mini", prompt_len=120, response_len=340, latency_ms=1500)
        print(tracer.summary())

    .. versionadded:: 2.0
    """

    def __init__(self, max_history: int = 5000) -> None:
        self._max = max_history
        self._records: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def record(
        self,
        model: str,
        prompt_len: int,
        response_len: int,
        latency_ms: float,
        success: bool = True,
    ) -> None:
        """Append a request span.

        Args:
            model: Model identifier.
            prompt_len: Character count of the prompt.
            response_len: Character count of the response.
            latency_ms: Wall-clock time in milliseconds.
            success: Whether the call succeeded.
        """
        entry = {
            "ts": _time.time(),
            "model": model,
            "prompt_len": prompt_len,
            "response_len": response_len,
            "latency_ms": latency_ms,
            "success": success,
        }
        with self._lock:
            self._records.append(entry)
            if len(self._records) > self._max:
                self._records = self._records[-self._max:]

    def summary(self) -> Dict[str, Any]:
        """Return aggregated trace summary.

        Returns:
            Dict with ``total_calls``, ``avg_latency_ms``,
            ``success_rate``, and ``by_model`` breakdown.
        """
        with self._lock:
            if not self._records:
                return {"total_calls": 0}
            total = len(self._records)
            ok = sum(1 for r in self._records if r["success"])
            avg_lat = sum(r["latency_ms"] for r in self._records) / total
            by_model: Dict[str, int] = {}
            for r in self._records:
                by_model[r["model"]] = by_model.get(r["model"], 0) + 1
            return {
                "total_calls": total,
                "success_rate": ok / total * 100,
                "avg_latency_ms": round(avg_lat, 1),
                "by_model": by_model,
            }


class LLMClientPool:
    """Connection-pool wrapper for concurrent LLM throughput.

    Maintains a fixed number of :class:`LLMClient` instances and
    round-robins requests across them.

    Args:
        pool_size: Number of client instances.
        base_url: Ollama endpoint passed to each client.

    Example::

        pool = LLMClientPool(pool_size=3)
        result = pool.generate("Summarise the data.")

    .. versionadded:: 2.0
    """

    def __init__(self, pool_size: int = 2, base_url: str = "http://localhost:11434") -> None:
        self._clients: List[LLMClient] = []
        for _ in range(pool_size):
            self._clients.append(LLMClient(base_url=base_url))
        self._idx = 0
        self._lock = threading.Lock()

    def _next(self) -> LLMClient:
        with self._lock:
            client = self._clients[self._idx % len(self._clients)]
            self._idx += 1
            return client

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate using the next available pooled client.

        Args:
            prompt: The user prompt.
            **kwargs: Forwarded to :meth:`LLMClient.generate`.

        Returns:
            LLM response dict.
        """
        return self._next().generate(prompt, **kwargs)

    async def generate_async(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Async generate using the next available pooled client.

        Args:
            prompt: The user prompt.
            **kwargs: Forwarded to :meth:`LLMClient.generate_async`.

        Returns:
            LLM response dict.
        """
        return await self._next().generate_async(prompt, **kwargs)

    @property
    def pool_size(self) -> int:
        """Return the number of pooled clients."""
        return len(self._clients)


    def close(self):
        """Close synchronous resources."""
        if hasattr(self, '_session') and self._session:
            self._session.close()

    async def aclose(self):
        """Close asynchronous resources."""
        if hasattr(self, '_async_client') and self._async_client:
            await self._async_client.aclose()

# ---------------------------------------------------------------------------
# Thread-safe singleton accessor (v2.0)
# ---------------------------------------------------------------------------

_llm_client_singleton: Optional[LLMClient] = None
_llm_client_lock = threading.Lock()


def get_llm_client(base_url: str = "http://localhost:11434") -> LLMClient:
    """Return the global :class:`LLMClient` singleton.

    Thread-safe with double-checked locking.

    Args:
        base_url: Ollama API endpoint.

    Returns:
        The shared ``LLMClient`` instance.

    .. versionadded:: 2.0
    """
    global _llm_client_singleton
    if _llm_client_singleton is None:
        with _llm_client_lock:
            if _llm_client_singleton is None:
                _llm_client_singleton = LLMClient(base_url=base_url)
    return _llm_client_singleton

