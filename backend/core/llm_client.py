# Handles communication with local Ollama LLM server

import requests

class LLMClient:
    """Client for communicating with local Ollama LLM server."""
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.primary_model = "llama3.1:8b"
        self.review_model = "phi3:mini"

    def generate(self, prompt, model=None):
        if model is None:
            model = self.primary_model
        url = f"{self.base_url}/api/generate"
        payload = {"model": model, "prompt": prompt}
        try:
            response = requests.post(url, json=payload, timeout=300, stream=True)
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
                        pass
            return {"model": model, "prompt": prompt, "response": output.strip()}
        except Exception as e:
            return {"error": str(e)}

    def generate_primary(self, prompt):
        return self.generate(prompt, model=self.primary_model)

    def generate_review(self, prompt):
        return self.generate(prompt, model=self.review_model)
