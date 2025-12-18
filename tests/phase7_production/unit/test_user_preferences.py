"""
Unit Tests for User Preferences Manager
Production-grade tests for user settings and preferences
"""
import pytest
import sys
from pathlib import Path
import json
import tempfile
import os
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from backend.core.user_preferences import UserPreferencesManager, UserPreferences


class TestUserPreferencesManager:
    """Test suite for UserPreferencesManager"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def manager(self, temp_config_dir):
        """Create preferences manager"""
        return UserPreferencesManager(config_dir=temp_config_dir)
    
    # ===== BASIC READ/WRITE TESTS =====
    
    def test_load_default_preferences(self, manager):
        """Should load default preferences on first use"""
        prefs = manager.load_preferences()
        
        assert isinstance(prefs, UserPreferences)
        assert prefs.primary_model == "llama3.1:8b"
        assert prefs.auto_model_selection == True
    
    def test_save_and_load_preferences(self, manager):
        """Should save and reload preferences"""
        # Modify and save
        prefs = manager.load_preferences()
        prefs.primary_model = "qwen2.5:7b"
        manager.save_preferences(prefs)
        
        # Reload
        manager._preferences = None  # Clear cache
        reloaded = manager.load_preferences()
        assert reloaded.primary_model == "qwen2.5:7b"
    
    def test_preferences_file_created(self, temp_config_dir, manager):
        """Preferences file should be created on first save"""
        manager.load_preferences()
        
        prefs_file = Path(temp_config_dir) / "user_preferences.json"
        assert prefs_file.exists()
    
    # ===== UPDATE TESTS =====
    
    def test_update_single_preference(self, manager):
        """Should update single preference field"""
        result = manager.update_preferences(primary_model="phi3:mini")
        
        assert result == True
        prefs = manager.load_preferences()
        assert prefs.primary_model == "phi3:mini"
    
    def test_update_multiple_preferences(self, manager):
        """Should update multiple preference fields"""
        result = manager.update_preferences(
            primary_model="qwen2.5:7b",
            review_model="gemma2:2b",
            auto_model_selection=False
        )
        
        assert result == True
        prefs = manager.load_preferences()
        assert prefs.primary_model == "qwen2.5:7b"
        assert prefs.review_model == "gemma2:2b"
        assert prefs.auto_model_selection == False
    
    # ===== MODEL CONFIG TESTS =====
    
    def test_get_model_config(self, manager):
        """Should get current model configuration"""
        config = manager.get_model_config()
        
        assert isinstance(config, dict)
        assert "primary_model" in config
        assert "review_model" in config
        assert "embedding_model" in config
    
    def test_set_model_config(self, manager):
        """Should set model configuration"""
        result = manager.set_model_config(
            primary="qwen2.5:7b",
            review="gemma2:2b",
            embedding="nomic-embed-text"
        )
        
        assert result == True
        config = manager.get_model_config()
        assert config["primary_model"] == "qwen2.5:7b"
        assert config["review_model"] == "gemma2:2b"
    
    # ===== ROUTING PREFERENCE TESTS =====
    
    def test_intelligent_routing_off_by_default(self, manager):
        """Intelligent routing should be OFF by default (respects user choice)"""
        prefs = manager.load_preferences()
        assert prefs.enable_intelligent_routing == False
    
    def test_enable_intelligent_routing(self, manager):
        """Should be able to enable intelligent routing"""
        result = manager.update_preferences(enable_intelligent_routing=True)
        
        assert result == True
        prefs = manager.load_preferences()
        assert prefs.enable_intelligent_routing == True
    
    # ===== FIRST-TIME USER TESTS =====
    
    def test_is_first_time_user(self, manager):
        """Should detect first-time users"""
        assert manager.is_first_time_user() == True
    
    def test_mark_setup_complete(self, manager):
        """Should mark setup as complete"""
        result = manager.mark_setup_complete()
        
        assert result == True
        assert manager.is_first_time_user() == False
    
    # ===== MODEL TEST RESULTS TESTS =====
    
    def test_save_model_test_result(self, manager):
        """Should save model test results"""
        result = manager.save_model_test_result(
            model_name="llama3.1:8b",
            success=True,
            response_time=1.5
        )
        
        assert result == True
        prefs = manager.load_preferences()
        assert "llama3.1:8b" in prefs.model_test_results
        assert prefs.model_test_results["llama3.1:8b"]["success"] == True
    
    def test_save_test_result_with_dict(self, manager):
        """Should save test results from dict"""
        test_data = {
            "model": "qwen2.5:7b",
            "success": True,
            "response_time": 2.1,
            "memory_used": "8GB"
        }
        
        result = manager.save_test_result(test_data)
        
        assert result == True
        prefs = manager.load_preferences()
        assert "qwen2.5:7b" in prefs.model_test_results
    
    # ===== PERSISTENCE TESTS =====
    
    def test_preferences_persist_across_instances(self, temp_config_dir):
        """Preferences should persist across manager instances"""
        # First manager
        manager1 = UserPreferencesManager(config_dir=temp_config_dir)
        manager1.update_preferences(primary_model="qwen2.5:7b")
        
        # Second manager
        manager2 = UserPreferencesManager(config_dir=temp_config_dir)
        prefs = manager2.load_preferences()
        assert prefs.primary_model == "qwen2.5:7b"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
