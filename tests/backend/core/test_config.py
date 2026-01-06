import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.backend.core.config import Settings, get_settings, apply_environment_preset, validate_config

def test_settings_defaults():
    # Ensure env vars don't interfere with defaults for this test
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings()
        assert settings.app_name == "Nexus LLM Analytics"
        # Host may be set by .env file, just ensure it's a valid string
        assert isinstance(settings.host, str)
        assert len(settings.host) > 0
        assert settings.port == 8000
        assert isinstance(settings.cors_allowed_origins, list)
        assert "http://localhost:3000" in settings.cors_allowed_origins

def test_validators():
    with patch.dict(os.environ, {}, clear=True):
        # Test CORS validator
        s = Settings(cors_allowed_origins="a.com, b.com")
        assert "a.com" in s.cors_allowed_origins
        assert "b.com" in s.cors_allowed_origins
        
        # Test file extensions
        s = Settings(allowed_file_extensions="TXT, PDF ")
        assert "txt" in s.allowed_file_extensions
        assert "pdf" in s.allowed_file_extensions
        
        # Test log level
        s = Settings(log_level="debug")
        assert s.log_level == "DEBUG"
        
        with pytest.raises(ValueError):
            Settings(log_level="INVALID")

def test_paths():
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings()
        # Mock mkdir to avoid actual OS operations
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            assert isinstance(settings.get_upload_path(), Path)
            assert isinstance(settings.get_reports_path(), Path)
            mock_mkdir.assert_called()

def test_safe_dict():
    with patch.dict(os.environ, {"OLLAMA_API_KEY": "secret123"}, clear=True):
        settings = Settings()
        safe = settings.get_safe_dict()
        assert safe["ollama_api_key"] == "***MASKED***"
        assert safe["app_name"] == "Nexus LLM Analytics"

def test_setup_logging():
    settings = Settings()
    with patch('logging.getLogger') as mock_get_logger:
        mock_root = MagicMock()
        mock_get_logger.return_value = mock_root
        
        settings.setup_logging()
        
        mock_root.handlers.clear.assert_called()
        mock_root.addHandler.assert_called()

def test_environment_validation():
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings()
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            
            # Mock psutil
            with patch('psutil.virtual_memory') as mock_mem:
                mock_mem.return_value.available = 100 * 1024**3 # 100GB
                
                warnings = settings.validate_environment()
                assert len(warnings) == 0
                
                # Test low memory
                mock_mem.return_value.available = 1 * 1024**3
                warnings = settings.validate_environment()
                assert len(warnings) > 0
                assert "Low memory" in warnings[0]

def test_apply_env_preset():
    apply_environment_preset("production")
    s = get_settings()
    assert s.debug is False
    assert s.workers == 4
    
    apply_environment_preset("development")
    assert s.debug is True # Should switch back if mutable singleton
