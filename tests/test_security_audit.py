
import pytest
import os
import sys
import json
import logging
from unittest.mock import MagicMock, patch
from backend.core.security.sandbox import EnhancedSandbox
from backend.core.security.security_guards import SecurityGuards
from backend.core.engine.query_orchestrator import QueryOrchestrator
from backend.services.analysis_service import AnalysisService
# from backend.utils.data_utils import load_data

# Configure logging for audit tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SecurityAudit")

class TestSecurityAudit:
    """
    Phase 12: Security & Privacy Audit.
    Aggressively probes the system for vulnerabilities.
    """

    @pytest.fixture
    def sandbox(self):
        return EnhancedSandbox()

    @pytest.fixture
    def orchestrator(self):
        # Mock dependencies to isolate the orchestrator logic
        mock_selector = MagicMock()
        mock_selector.get_model.return_value = "phi3:mini"
        
        mock_discovery = MagicMock()
        mock_discovery.discover_models_sync.return_value = []
        
        with patch('backend.core.engine.model_selector.ModelSelector', return_value=mock_selector):
            orchestrator = QueryOrchestrator()
            return orchestrator

    # =========================================================================
    # SECTION A: SANDBOX BREACH ATTEMPTS
    # =========================================================================
    
    def test_sandbox_rce_block(self, sandbox):
        """Attempt Remote Code Execution (RCE) via os/subprocess."""
        logger.info("Attempting RCE inside sandbox...")
        
        # Attack 1: Classic os.system
        code_os = "import os; os.system('echo hacked')"
        result_os = sandbox.execute(code_os)
        assert "error" in result_os
        err = str(result_os["error"])
        assert "Import of 'os' is not allowed" in err or "AST validation failed" in err or "Dangerous import" in err

        # Attack 2: Subprocess
        code_sub = "import subprocess; subprocess.run(['ls'])"
        result_sub = sandbox.execute(code_sub)
        assert "error" in result_sub
        err = str(result_sub["error"])
        assert "Import of 'subprocess' is not allowed" in err or "AST validation failed" in err

    def test_sandbox_fs_access_block(self, sandbox):
        """Attempt to read sensitive files outside allowed scope."""
        logger.info("Attempting File System Traversal...")
        
        # Attack: Read /etc/passwd or Windows equivalent
        code_read = "open('C:/Windows/win.ini', 'r').read()"
        result_read = sandbox.execute(code_read)
        assert "error" in result_read
        err = str(result_read["error"])
        assert "Access to 'open' is not allowed" in err or "AST validation failed" in err or "Dangerous function call" in err

    def test_sandbox_network_block(self, sandbox):
        """Attempt external network connections."""
        logger.info("Attempting Network Access...")
        
        # Attack 1: Socket
        code_sock = "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.connect(('google.com', 80))"
        result_sock = sandbox.execute(code_sock)
        assert "error" in result_sock
        err = str(result_sock["error"])
        assert "Import of 'socket' is not allowed" in err or "AST validation failed" in err

        # Attack 2: Requests
        code_req = "import requests; requests.get('http://google.com')"
        result_req = sandbox.execute(code_req)
        assert "error" in result_req
        err = str(result_req["error"])
        assert "Import of 'requests' is not allowed" in err or "AST validation failed" in err

    def test_sandbox_import_bypass(self, sandbox):
        """Attempt to bypass import restrictions using __import__."""
        logger.info("Attempting Import Bypass...")
        
        # Attack: __import__ magic
        code_imp = "__import__('os').system('echo hacked')"
        result_imp = sandbox.execute(code_imp)
        assert "error" in result_imp
        err = str(result_imp["error"])
        assert "Import of 'os' is not allowed" in err or "AST validation failed" in err or "Dangerous function call" in err

    # ... (skipping B section)

    def test_sql_injection_defense(self):
        """Verify standard SQL injection patterns are blocked or sanitized."""
        # Note: Nexus uses SQLAlchemy / Pandas, which creates parameterized queries or 
        # object-based filtering, inherently reducing SQLi risk.
        # We verify that 'pandas.read_sql' is BLOCKED in Sandbox.
        
        sb = EnhancedSandbox()
        code = "import pandas as pd; pd.read_sql('DROP TABLE users', 'sqlite://')"
        result = sb.execute(code)
        
        assert "error" in result
        err = str(result["error"])
        # It might fail at import or execution, both are valid blocks
        assert "not allowed" in err or "not found" in err or "AttributeError" in err

    # =========================================================================
    # SECTION D: PRIVACY COMPLIANCE
    # =========================================================================

    def test_no_unauthorized_telemetry(self, sandbox):
        """Confirm no 'telemetry' or 'analytics' modules are allowed."""
        logger.info("Verifying Telemetry Blocking...")
        
        telemetry_mods = ['posthog', 'mixpanel', 'sentry_sdk', 'segment']
        
        for mod in telemetry_mods:
            code = f"import {mod}"
            result = sandbox.execute(code)
            # Should fail either due to 'not found' (ideal) or 'import not allowed'
            # If it succeeds, valid = True, which is a fail.
            if "error" not in result:
                 # Check if it was actually imported (might just be no-op if logic allowed it)
                 # But EnhancedSandbox whitelists imports, so this should fail hard.
                 pass 
            
            # The error message typically says "Import of 'X' is not allowed" or syntax error
            assert "error" in result

    # =========================================================================
    # SECTION E: ERROR HANDLING SECURITY
    # =========================================================================
    
    def test_error_sanitization(self, sandbox):
        """Ensure stack traces don't leak internal paths in the Sandbox return."""
        logger.info("Testing Error Sanitization...")
        
        # Force an error
        code = "1 / 0"
        result = sandbox.execute(code)
        
        assert "error" in result
        error_msg = result["error"]
        
        # Should contain "ZeroDivisionError"
        assert "division by zero" in error_msg or "ZeroDivisionError" in error_msg
        
        # Should NOT ensure full stack trace with local paths
        # (Sandbox catches exception and returns formatted string)
        # We manually verify it doesn't dump the whole stack trace object into the JSON output
        assert "c:\\Users" not in error_msg # Windows path leakage check
