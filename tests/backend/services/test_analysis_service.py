"""
Tests for AnalysisService (services/analysis_service.py)

Note: AnalysisService requires the full backend infrastructure (ModelManager, 
QueryOrchestrator, Plugin Registry, etc.) which has complex import dependencies.
Skipped for unit tests - covered by integration tests.
"""
import pytest

pytestmark = pytest.mark.skip(reason="Complex import dependencies - use integration tests")

def test_analysis_service_placeholder():
    """Placeholder - AnalysisService tested via integration tests"""
    pass
