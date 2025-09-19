import json
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel
from datetime import datetime

class UserPreferences(BaseModel):
    """User preferences model with validation"""
    primary_model: str = "llama3.1:8b"
    review_model: str = "phi3:mini"
    embedding_model: str = "nomic-embed-text"
    auto_model_selection: bool = True
    allow_swap_usage: bool = True
    memory_buffer_gb: float = 0.5
    preferred_performance: str = "balanced"  # "high", "balanced", "efficient"
    first_time_setup_complete: bool = False
    last_updated: str = ""
    model_test_results: Dict[str, Any] = {}

class UserPreferencesManager:
    """Manages user preferences with persistent storage"""
    
    def __init__(self, config_dir: str = None):
        if config_dir is None:
            # Store in project root config directory
            config_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "config")
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.preferences_file = self.config_dir / "user_preferences.json"
        self._preferences: Optional[UserPreferences] = None
    
    def load_preferences(self) -> UserPreferences:
        """Load user preferences from file or create defaults"""
        if self._preferences is not None:
            return self._preferences
        
        try:
            if self.preferences_file.exists():
                with open(self.preferences_file, 'r') as f:
                    data = json.load(f)
                    self._preferences = UserPreferences(**data)
                    logging.info("✅ User preferences loaded successfully")
            else:
                # Create default preferences
                self._preferences = UserPreferences()
                self.save_preferences()
                logging.info("📝 Created default user preferences")
        except Exception as e:
            logging.warning(f"⚠️ Error loading preferences, using defaults: {e}")
            self._preferences = UserPreferences()
        
        return self._preferences
    
    def save_preferences(self, preferences: UserPreferences = None) -> bool:
        """Save user preferences to file"""
        try:
            if preferences is not None:
                self._preferences = preferences
            
            if self._preferences is None:
                return False
            
            # Update last modified timestamp
            self._preferences.last_updated = datetime.now().isoformat()
            
            # Save to file
            with open(self.preferences_file, 'w') as f:
                json.dump(self._preferences.model_dump(), f, indent=2)
            
            logging.info("💾 User preferences saved successfully")
            return True
        except Exception as e:
            logging.error(f"❌ Error saving preferences: {e}")
            return False
    
    def update_preferences(self, **kwargs) -> bool:
        """Update specific preference fields"""
        try:
            preferences = self.load_preferences()
            
            # Update fields
            for key, value in kwargs.items():
                if hasattr(preferences, key):
                    setattr(preferences, key, value)
                else:
                    logging.warning(f"Unknown preference field: {key}")
            
            return self.save_preferences(preferences)
        except Exception as e:
            logging.error(f"❌ Error updating preferences: {e}")
            return False
    
    def get_model_config(self) -> Dict[str, str]:
        """Get current model configuration"""
        preferences = self.load_preferences()
        return {
            "primary_model": preferences.primary_model,
            "review_model": preferences.review_model,
            "embedding_model": preferences.embedding_model,
            "auto_selection": preferences.auto_model_selection,
            "allow_swap": preferences.allow_swap_usage
        }
    
    def set_model_config(self, primary: str, review: str, embedding: str = None, auto_selection: bool = None) -> bool:
        """Set model configuration"""
        update_data = {
            "primary_model": primary,
            "review_model": review
        }
        
        if embedding is not None:
            update_data["embedding_model"] = embedding
        
        if auto_selection is not None:
            update_data["auto_model_selection"] = auto_selection
        
        return self.update_preferences(**update_data)
    
    def mark_setup_complete(self) -> bool:
        """Mark first-time setup as complete"""
        return self.update_preferences(first_time_setup_complete=True)
    
    def is_first_time_user(self) -> bool:
        """Check if this is a first-time user"""
        preferences = self.load_preferences()
        return not preferences.first_time_setup_complete
    
    def save_model_test_result(self, model_name: str, success: bool, response_time: float = None, error: str = None) -> bool:
        """Save model test results"""
        preferences = self.load_preferences()
        
        test_result = {
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "response_time": response_time,
            "error": error
        }
        
        preferences.model_test_results[model_name] = test_result
        return self.save_preferences(preferences)
    
    def save_test_result(self, test_data: Dict[str, Any]) -> bool:
        """Save general test results"""
        preferences = self.load_preferences()
        
        # Add timestamp if not provided
        if "timestamp" not in test_data:
            test_data["timestamp"] = datetime.now().isoformat()
        
        # Store under model name or use "startup_test" as default
        model_name = test_data.get("model", "startup_test")
        preferences.model_test_results[model_name] = test_data
        
        return self.save_preferences(preferences)
    
    def get_model_test_results(self) -> Dict[str, Any]:
        """Get model test results"""
        preferences = self.load_preferences()
        return preferences.model_test_results
    
    def reset_preferences(self) -> bool:
        """Reset to default preferences"""
        try:
            self._preferences = UserPreferences()
            return self.save_preferences()
        except Exception as e:
            logging.error(f"❌ Error resetting preferences: {e}")
            return False
    
    def export_preferences(self) -> Dict[str, Any]:
        """Export preferences for backup or sharing"""
        preferences = self.load_preferences()
        return preferences.model_dump()
    
    def import_preferences(self, data: Dict[str, Any]) -> bool:
        """Import preferences from backup"""
        try:
            preferences = UserPreferences(**data)
            return self.save_preferences(preferences)
        except Exception as e:
            logging.error(f"❌ Error importing preferences: {e}")
            return False

# Global instance
_preferences_manager = None

def get_preferences_manager() -> UserPreferencesManager:
    """Get global preferences manager instance"""
    global _preferences_manager
    if _preferences_manager is None:
        _preferences_manager = UserPreferencesManager()
    return _preferences_manager