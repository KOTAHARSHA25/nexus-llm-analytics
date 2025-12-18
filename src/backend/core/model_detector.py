"""
Dynamic Model Detector for Intelligent Routing

Automatically detects available Ollama models and maps them to routing tiers.
No need to download specific models - works with whatever the user has!

Author: Research Team
Date: November 9, 2025
"""

import subprocess
import logging
from typing import Dict, List, Tuple, Optional
import re


class ModelDetector:
    """Detects available Ollama models and categorizes them by size/capability"""
    
    # Model size categories (in GB, approximate)
    TINY_MODELS = ['tinyllama', 'phi', 'qwen2.5:0.5b', 'qwen:0.5b', 'gemma:2b']
    SMALL_MODELS = ['phi3', 'qwen2.5:3b', 'qwen:4b', 'gemma:7b', 'mistral:7b']
    MEDIUM_MODELS = ['llama3', 'llama3.1', 'qwen2.5:7b', 'mixtral:8x7b']
    LARGE_MODELS = ['llama3:70b', 'qwen2.5:14b', 'qwen2.5:32b']
    
    def __init__(self):
        self.available_models = []
        self.tier_mapping = {}
        
    def detect_models(self) -> List[str]:
        """
        Detect all available Ollama models on the system
        
        Returns:
            List of model names
        """
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        # Extract model name (first column)
                        model_name = line.split()[0]
                        
                        # Skip embedding models
                        if 'embed' in model_name.lower() or 'embedding' in model_name.lower():
                            logging.debug(f"Skipping embedding model: {model_name}")
                            continue
                        
                        models.append(model_name)
                
                self.available_models = models
                logging.debug(f"Detected {len(models)} available models: {models}")
                return models
            else:
                logging.error(f"Failed to detect models: {result.stderr}")
                return []
                
        except Exception as e:
            logging.error(f"Error detecting models: {e}")
            return []
    
    def categorize_model(self, model_name: str) -> str:
        """
        Categorize a model by size/capability
        
        Args:
            model_name: Model name (e.g., 'phi3:mini', 'llama3.1:8b')
            
        Returns:
            Category: 'tiny', 'small', 'medium', or 'large'
        """
        model_lower = model_name.lower()
        
        # Check size indicators in model name
        if any(keyword in model_lower for keyword in ['0.5b', 'tiny', 'nano']):
            return 'tiny'
        elif any(keyword in model_lower for keyword in ['3b', '4b', 'mini', '2b']):
            return 'small'
        elif any(keyword in model_lower for keyword in ['7b', '8b', '13b']):
            return 'medium'
        elif any(keyword in model_lower for keyword in ['14b', '32b', '70b', 'large']):
            return 'large'
        
        # Check against known model families
        for keyword in self.TINY_MODELS:
            if keyword in model_lower:
                return 'tiny'
        
        for keyword in self.SMALL_MODELS:
            if keyword in model_lower:
                return 'small'
        
        for keyword in self.MEDIUM_MODELS:
            if keyword in model_lower:
                return 'medium'
        
        for keyword in self.LARGE_MODELS:
            if keyword in model_lower:
                return 'large'
        
        # Default to medium if unknown
        return 'medium'
    
    def map_to_tiers(self) -> Dict[str, Optional[str]]:
        """
        Map detected models to routing tiers (fast, balanced, full_power)
        
        Strategy:
        - Fast: Use smallest available model (tiny/small)
        - Balanced: Use medium model or largest small model
        - Full Power: Use largest available model
        
        Returns:
            Dict with 'fast', 'balanced', 'full_power' keys
        """
        if not self.available_models:
            self.detect_models()
        
        if not self.available_models:
            logging.warning("[!] No models detected! Routing will use defaults.")
            return {
                'fast': None,
                'balanced': None,
                'full_power': None
            }
        
        # Categorize all models
        categorized = {
            'tiny': [],
            'small': [],
            'medium': [],
            'large': []
        }
        
        for model in self.available_models:
            category = self.categorize_model(model)
            categorized[category].append(model)
        
        # Select best model for each tier
        fast_model = None
        balanced_model = None
        full_power_model = None
        
        # FAST TIER: Prefer tiny > small > medium
        if categorized['tiny']:
            fast_model = categorized['tiny'][0]
        elif categorized['small']:
            fast_model = categorized['small'][0]
        elif categorized['medium']:
            fast_model = categorized['medium'][0]
        elif categorized['large']:
            fast_model = categorized['large'][0]
        
        # BALANCED TIER: Prefer small > medium > tiny
        if categorized['small']:
            balanced_model = categorized['small'][0]
        elif categorized['medium']:
            balanced_model = categorized['medium'][0]
        elif categorized['tiny']:
            balanced_model = categorized['tiny'][0]
        elif categorized['large']:
            balanced_model = categorized['large'][0]
        
        # FULL POWER TIER: Prefer large > medium > small
        if categorized['large']:
            full_power_model = categorized['large'][0]
        elif categorized['medium']:
            full_power_model = categorized['medium'][0]
        elif categorized['small']:
            full_power_model = categorized['small'][0]
        elif categorized['tiny']:
            full_power_model = categorized['tiny'][0]
        
        self.tier_mapping = {
            'fast': fast_model,
            'balanced': balanced_model,
            'full_power': full_power_model
        }
        
        # Log the mapping
        logging.debug("[MODEL DETECTOR] Tier Mapping:")
        logging.debug(f"  FAST: {fast_model}")
        logging.debug(f"  BALANCED: {balanced_model}")
        logging.debug(f"  FULL_POWER: {full_power_model}")
        
        return self.tier_mapping
    
    def get_tier_models(self) -> Dict[str, Optional[str]]:
        """
        Get the current tier mapping (detects if not already done)
        
        Returns:
            Dict with model names for each tier
        """
        if not self.tier_mapping:
            self.map_to_tiers()
        
        return self.tier_mapping


# Singleton instance
_model_detector = None

def get_model_detector() -> ModelDetector:
    """Get or create the global ModelDetector instance"""
    global _model_detector
    if _model_detector is None:
        _model_detector = ModelDetector()
    return _model_detector


# Convenience function
def detect_and_map_models() -> Dict[str, Optional[str]]:
    """
    Detect available models and return tier mapping
    
    Returns:
        Dict with 'fast', 'balanced', 'full_power' keys
    """
    detector = get_model_detector()
    return detector.get_tier_models()


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("DYNAMIC MODEL DETECTOR - TEST")
    print("=" * 80)
    
    detector = ModelDetector()
    
    # Detect models
    print("\n[>] Detecting available models...")
    models = detector.detect_models()
    print(f"Found {len(models)} models:")
    for model in models:
        category = detector.categorize_model(model)
        print(f"  • {model} → {category.upper()}")
    
    # Map to tiers
    print("\n[>] Mapping to routing tiers...")
    tier_mapping = detector.map_to_tiers()
    print(f"\nTier Assignments:")
    print(f"  FAST TIER:       {tier_mapping['fast']}")
    print(f"  BALANCED TIER:   {tier_mapping['balanced']}")
    print(f"  FULL POWER TIER: {tier_mapping['full_power']}")
    
    print("\n" + "=" * 80)
    print("[OK] Model detection complete!")
    print("=" * 80)
