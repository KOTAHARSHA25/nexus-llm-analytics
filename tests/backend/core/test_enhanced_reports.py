import sys
import pytest
from unittest.mock import MagicMock, patch

# Mocking reportlab globally at module level is DANGEROUS because it persists across tests.
# Since reportlab is installed, we rely on the installed version.
# If mocking is needed, it should be done inside fixtures or contexts.

# (Removed global sys.modules override)

# Now import the module under test
from backend.core.enhanced_reports import (
    PDFReportGenerator, ReportTemplate, ColorPalette
)

@pytest.fixture
def generator():
    return PDFReportGenerator()

def test_init(generator):
    assert generator.template is not None
    assert generator.styles is not None

def test_generate_report(generator):
    results = [
        {
            "query": "Test Query",
            "success": True,
            "result": "Some result text.",
            "execution_time": 1.5,
            "chart_data": {"visualization": {"figure_json": "{}"}}
        }
    ]
    
    with patch('backend.core.enhanced_reports.SimpleDocTemplate') as mock_doc:
        mock_build = mock_doc.return_value.build
        path = generator.generate_report(results, "test.pdf")
        
        assert "test.pdf" in path
        mock_build.assert_called_once()

def test_sections(generator):
    results = [{"success": True}]
    
    # Exec Summary
    summary = generator._create_executive_summary(results)
    assert len(summary) > 0
    
    # Stats
    stats = generator._create_statistical_summary(results)
    assert len(stats) > 0
    
    # Quality
    quality = generator._create_data_quality_section(results)
    assert len(quality) > 0

def test_chart_save_failure(generator):
    # Ensure it doesn't crash if chart fails
    with patch('backend.core.enhanced_reports.json.loads', side_effect=Exception("Fail")):
        path = generator._save_chart("bad json", 1)
        assert path is None
