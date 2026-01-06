# Handles communication with local Ollama LLM server

import requests
import httpx
import asyncio
import logging
from typing import Dict, Any, Optional, Union

class LLMClient:
    """Client for communicating with local Ollama LLM server."""
    
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
            
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            
            data = response.json()
            if "response" in data:
                return {"model": model, "prompt": prompt, "response": data["response"].strip(), "success": True}
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
                
            return adaptive_timeout
            
        except Exception as e:
            print(f"Failed to calculate adaptive timeout: {e}")
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
        if circuit_breaker.state == "OPEN": # using string access or import constant if available
             # But we can access the property directly
             pass

        # To keep it robust using the provided class:
        # We'll check state first
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
            circuit_breaker._total_calls += 1
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                response = await client.post(url, json=payload)
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
            circuit_breaker._handle_failure(Exception(f"Timeout after {timeout}s"))
            logging.warning(f"Async LLM call timed out after {timeout}s for model {model}")
            return {
                "model": model,
                "prompt": prompt,
                "response": "",
                "error": f"Request timed out after {timeout}s",
                "timeout": True
            }
        except Exception as e:
            circuit_breaker._handle_failure(e)
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
