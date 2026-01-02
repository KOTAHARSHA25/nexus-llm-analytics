import pytest
import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Add src to path so we can import backend modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.main import app
from backend.core.llm_client import LLMClient

@pytest.fixture(scope="module")
def client():
    """FastAPI Test Client"""
    with TestClient(app) as c:
        yield c

@pytest.fixture
def mock_llm_response():
    """Mock LLM response to control output"""
    with patch("backend.core.llm_client.LLMClient.generate") as mock_gen:
        mock_gen.return_value = {"response": "This is a mock analysis result."}
        yield mock_gen


@pytest.fixture(autouse=True)
def mock_cot_config():
    """Disable CoT for tests to ensure deterministic execution path"""
    with patch("backend.plugins.data_analyst_agent.DataAnalystAgent._load_cot_config") as mock_config:
        mock_config.return_value = {'enabled': False}
        yield mock_config

@pytest.fixture(scope="session", autouse=True)
def mock_settings(tmp_path_factory):
    """Override settings to use temp directories for tests"""
    from backend.core.config import settings
    
    # Create temp dirs
    upload_dir = tmp_path_factory.mktemp("uploads")
    settings.upload_directory = str(upload_dir)
    
    # Ensure DataPathResolver picks this up (it caches, so we might need to reset it)
    from backend.utils.data_utils import DataPathResolver
    DataPathResolver._uploads_dir = None 
    
    return settings

@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a temporary CSV file for testing"""
    # Create it locally first
    d = tmp_path / "data"
    d.mkdir()
    p = d / "test_analysis_data.csv"
    p.write_text("category,value,region\nA,10,North\nB,20,South\nC,30,East")
    return p

@pytest.fixture(autouse=True)
def mock_model_selector():
    """Mock ModelSelector to avoid real OLLAMA calls"""
    with patch("backend.core.model_selector.ModelSelector.select_optimal_models") as mock_sel:
        mock_sel.return_value = ("phi3:mini", "phi3:mini", "nomic-embed-text")
        yield mock_sel

@pytest.fixture(scope="session", autouse=True)
def mock_ollama_llm():
    """Mock LLM initialization to prevent any network calls"""
    with patch("backend.agents.model_initializer.ModelInitializer._initialize_llms", autospec=True) as mock_init:
        # Mock the side effects of initialization
        def side_effect(self):
            # Create mocks for LLMs
            primary = MagicMock()
            primary.model = "phi3:mini"
            primary.bind.return_value = primary
            
            review = MagicMock()
            
            # Set attributes on the instance
            self._primary_llm = primary
            self._review_llm = review
            self.cached_models = {'primary': 'phi3:mini', 'review': 'phi3:mini'}
            
        mock_init.side_effect = side_effect
        yield mock_init

@pytest.fixture
def sample_text_file(tmp_path):
    """Create a temporary Text file for testing"""
    d = tmp_path / "docs"
    d.mkdir()
    p = d / "test_doc.txt"
    p.write_text("This is a sample document for testing RAG capabilities.")
    return p
