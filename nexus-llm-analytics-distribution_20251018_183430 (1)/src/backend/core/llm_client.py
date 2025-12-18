# Handles communication with local Ollama LLM server

import requests
import logging

class LLMClient:
    """Client for communicating with local Ollama LLM server."""
    def __init__(self, base_url="http://localhost:11434", primary_model=None, review_model=None):
        from .model_selector import ModelSelector
        
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
            
        logging.info(f"🤖 LLMClient initialized - Primary: {self.primary_model}, Review: {self.review_model}")

    def generate(self, prompt, model=None, adaptive_timeout=True):
        from .circuit_breaker import get_circuit_breaker, CircuitBreakerConfig
        
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
            payload = {"model": model, "prompt": prompt}
            
            response = requests.post(url, json=payload, timeout=timeout, stream=True)
            response.raise_for_status()
            output = ""
            for line in response.iter_lines():
                if line:
                    try:
                        import json
                        data = json.loads(line.decode("utf-8"))
                        if isinstance(data, dict) and "response" in data:
                            output += data["response"]
                    except Exception:
                        logging.debug("Operation failed (non-critical) - continuing")
            
            if not output.strip():
                raise Exception("Empty response from LLM")
                
            return {"model": model, "prompt": prompt, "response": output.strip(), "success": True}
        
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
    
    def _calculate_adaptive_timeout(self, model):
        """Calculate timeout based on model requirements and system resources"""
        try:
            from .model_selector import ModelSelector
            import os
            
            # Get system memory info
            memory_info = ModelSelector.get_system_memory()
            available_ram = memory_info["available_gb"]
            
            # Base timeouts by model type
            base_timeouts = {
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
            allow_swap = os.getenv("ALLOW_SWAP_USAGE", "false").lower() == "true"
            if allow_swap and available_ram < required_ram:
                # Using swap - increase timeout by 3x
                adaptive_timeout = base_timeout * 3
                print(f"🐌 Using swap memory - Extended timeout to {adaptive_timeout}s for {clean_model}")
            elif available_ram < required_ram + 1.0:  # Close to memory limit
                # Tight memory - increase timeout by 1.5x
                adaptive_timeout = int(base_timeout * 1.5)
                print(f"⚠️ Low memory - Extended timeout to {adaptive_timeout}s for {clean_model}")
            else:
                # Normal operation
                adaptive_timeout = base_timeout
                
            return adaptive_timeout
            
        except Exception as e:
            print(f"Failed to calculate adaptive timeout: {e}")
            return 300  # Default fallback

    def generate_primary(self, prompt):
        return self.generate(prompt, model=self.primary_model)

    def generate_review(self, prompt):
        return self.generate(prompt, model=self.review_model)
