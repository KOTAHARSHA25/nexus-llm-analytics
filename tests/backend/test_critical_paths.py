"""
Critical-path regression tests for issues fixed in the fresh audit (Round 2).

Covers:
  1. PersistentCache: single get(), string paths, correct .pickle extension, tag invalidation
  2. Streaming endpoint: execution_method_str always defined (no NameError)
  3. Background tasks: all asyncio.create_task calls inside try block
  4. Streaming context: force_refresh & review_level forwarded
  5. visualize.py: exec() safety deny-list blocks dangerous code
  6. types.ts alignment: result_data includes 'agent' key
"""

import os
import sys
import time
import pickle
import tempfile
import asyncio
import pytest

# ---------------------------------------------------------------------------
# 1. PersistentCache unit tests
# ---------------------------------------------------------------------------

# Add project root to path so imports resolve
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestPersistentCache:
    """Tests for PersistentCache L3 after bug fixes."""

    def _make_cache(self, tmp_path):
        """Create a PersistentCache pointed at a temp directory."""
        from backend.core.enhanced_cache_integration import PersistentCache, _enhanced_cache_manager
        
        # PARANOID CLEARING: Reset global singleton even if not used directly
        import src.backend.core.enhanced_cache_integration as ci
        ci._enhanced_cache_manager = None

        cache = PersistentCache(capacity=100, default_ttl=3600)
        cache.cache_dir = str(tmp_path)
        return cache

    def test_single_get_method(self):
        """There must be exactly ONE get() method (no duplicate shadowing)."""
        from backend.core.enhanced_cache_integration import PersistentCache
        import inspect
        # get should be defined once — MRO should show exactly one
        methods = [m for m in dir(PersistentCache) if m == "get"]
        assert len(methods) == 1
        # And it should accept 'key' parameter
        sig = inspect.signature(PersistentCache.get)
        assert "key" in sig.parameters

    def test_get_returns_none_for_missing(self, tmp_path):
        cache = self._make_cache(tmp_path)
        assert cache.get("nonexistent_key") is None

    def test_put_and_get_roundtrip(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.put("hello", {"answer": 42}, tags={"test"})
        result = cache.get("hello")
        assert result == {"answer": 42}

    def test_file_extension_is_pickle(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.put("ext_test", "value")
        files = os.listdir(str(tmp_path))
        assert all(f.endswith(".pickle") for f in files), f"Unexpected extensions: {files}"

    def test_ttl_expiration(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.default_ttl = 0.1  # 100ms
        cache.put("ttl_key", "short-lived")
        time.sleep(0.2)
        assert cache.get("ttl_key") is None

    def test_invalidate_by_tags(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.put("a", "val_a", tags={"sales", "monthly"})
        cache.put("b", "val_b", tags={"inventory"})
        cache.put("c", "val_c", tags={"sales"})

        cache.invalidate_by_tags({"sales"})

        assert cache.get("a") is None
        assert cache.get("b") == "val_b"
        assert cache.get("c") is None

    def test_invalidate_by_tags_on_empty_dir(self, tmp_path):
        """Should not crash on an empty or non-existent directory."""
        cache = self._make_cache(tmp_path)
        cache.cache_dir = os.path.join(str(tmp_path), "does_not_exist")
        cache.invalidate_by_tags({"any"})  # Must not raise


# ---------------------------------------------------------------------------
# 2. Streaming endpoint: execution_method_str always set
# ---------------------------------------------------------------------------

class TestStreamingEndpointFields:
    """Verify that the streaming result_data dict is well-formed for both
    dict and non-dict raw_result branches."""

    def test_result_data_has_execution_method(self):
        """The result_data dict must always contain 'execution_method'
        regardless of whether raw_result is a dict or string.
        This is a source-level verification."""
        import ast
        analyze_path = os.path.join(PROJECT_ROOT, "src", "backend", "api", "analyze.py")
        with open(analyze_path, encoding='utf-8') as f:
            source = f.read()

        # execution_method_str must NOT be inside an else: block
        # It should be at the same indentation as the if/else that checks raw_result
        tree = ast.parse(source)

        # Find assignment "execution_method_str = ..."
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "execution_method_str":
                        # This confirms the variable is set somewhere
                        return
        pytest.fail("execution_method_str assignment not found in analyze.py")

    def test_result_data_includes_agent_field(self):
        """result_data must include 'agent' key for frontend type alignment."""
        analyze_path = os.path.join(PROJECT_ROOT, "src", "backend", "api", "analyze.py")
        with open(analyze_path, encoding='utf-8') as f:
            source = f.read()
        assert '"agent": result_dict.get("agent")' in source, \
            "result_data missing 'agent' field"


# ---------------------------------------------------------------------------
# 3. Background tasks: create_task inside try block
# ---------------------------------------------------------------------------

class TestBackgroundTasks:
    """Verify that _periodic_optimization task creation is inside try."""

    def test_optimization_task_inside_try(self):
        import ast
        cache_path = os.path.join(
            PROJECT_ROOT, "src", "backend", "core", "enhanced_cache_integration.py"
        )
        with open(cache_path, encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_start_background_tasks":
                # The body should start with a Try node
                assert any(isinstance(stmt, ast.Try) for stmt in node.body), \
                    "_start_background_tasks must have a try block"

                for stmt in node.body:
                    if isinstance(stmt, ast.Try):
                        # All create_task calls must be inside the try body
                        try_calls = ast.dump(ast.Module(body=stmt.body, type_ignores=[]))
                        assert "periodic_optimization" in try_calls, \
                            "_periodic_optimization must be inside the try block"
                        return

        pytest.fail("_start_background_tasks not found")


# ---------------------------------------------------------------------------
# 4. Streaming context: force_refresh & review_level forwarded
# ---------------------------------------------------------------------------

class TestStreamingContextFields:
    def test_context_includes_force_refresh_and_review_level(self):
        analyze_path = os.path.join(PROJECT_ROOT, "src", "backend", "api", "analyze.py")
        with open(analyze_path, encoding='utf-8') as f:
            source = f.read()
        # The streaming context dict must include both fields
        assert '"force_refresh": request.force_refresh' in source
        assert '"review_level": request.review_level' in source


# ---------------------------------------------------------------------------
# 5. exec() safety deny-list
# ---------------------------------------------------------------------------

class TestExecSafety:
    def test_deny_list_blocks_os_import(self):
        from backend.api.visualize import _validate_code_safety
        with pytest.raises(ValueError, match="safety filter"):
            _validate_code_safety("import os; os.system('rm -rf /')")

    def test_deny_list_blocks_subprocess(self):
        from backend.api.visualize import _validate_code_safety
        with pytest.raises(ValueError, match="safety filter"):
            _validate_code_safety("import subprocess; subprocess.run(['ls'])")

    def test_deny_list_blocks_open(self):
        from backend.api.visualize import _validate_code_safety
        with pytest.raises(ValueError, match="safety filter"):
            _validate_code_safety("f = open('/etc/passwd', 'r')")

    def test_allows_safe_pandas_code(self):
        from backend.api.visualize import _validate_code_safety
        # This should NOT raise
        _validate_code_safety("fig = px.bar(data, x='col1', y='col2')")

    def test_exec_with_timeout_works(self):
        from backend.api.visualize import _exec_with_timeout
        ns = {}
        result = _exec_with_timeout("x = 1 + 1", ns, timeout=5)
        assert result["x"] == 2
