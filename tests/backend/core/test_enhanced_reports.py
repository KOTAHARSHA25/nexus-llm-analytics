import sys
import pytest
from unittest.mock import MagicMock, patch

# Mock reportlab and openpyxl BEFORE importing enhanced_reports
class MockCanvas:
    def __init__(self, *args, **kwargs): pass
    def save(self): pass
    def showPage(self): pass
    def saveState(self): pass
    def restoreState(self): pass
    def setStrokeColor(self, *args): pass
    def setLineWidth(self, *args): pass
    def line(self, *args): pass
    def setFont(self, *args): pass
    def setFillColor(self, *args): pass
    def drawString(self, *args): pass
    def drawRightString(self, *args): pass
    def drawCentredString(self, *args): pass

mock_rl = MagicMock()
mock_rl.pdfgen.canvas.Canvas = MockCanvas
mock_rl.lib.pagesizes.A4 = (595, 841)
mock_rl.lib.pagesizes.letter = (612, 792)
mock_rl.lib.colors.HexColor = lambda x: x
mock_rl.lib.colors.white = "white"
mock_rl.lib.colors.black = "black"
mock_rl.lib.colors.grey = "grey"
mock_rl.lib.units.inch = 72
mock_rl.lib.enums.TA_CENTER = 1
mock_rl.lib.enums.TA_LEFT = 0
mock_rl.lib.enums.TA_RIGHT = 2
mock_rl.lib.enums.TA_JUSTIFY = 4
mock_rl.platypus.SimpleDocTemplate = MagicMock()
mock_rl.lib.styles.getSampleStyleSheet = lambda: MagicMock()
mock_rl.lib.styles.ParagraphStyle = MagicMock()

sys.modules['reportlab'] = mock_rl
sys.modules['reportlab.lib'] = mock_rl.lib
sys.modules['reportlab.lib.pagesizes'] = mock_rl.lib.pagesizes
sys.modules['reportlab.lib.units'] = mock_rl.lib.units
sys.modules['reportlab.lib.enums'] = mock_rl.lib.enums
sys.modules['reportlab.lib.colors'] = mock_rl.lib.colors
sys.modules['reportlab.lib.styles'] = mock_rl.lib.styles
sys.modules['reportlab.platypus'] = mock_rl.platypus
sys.modules['reportlab.pdfgen'] = mock_rl.pdfgen
sys.modules['reportlab.pdfgen.canvas'] = mock_rl.pdfgen.canvas
sys.modules['openpyxl'] = MagicMock()
sys.modules['openpyxl.styles'] = MagicMock()
sys.modules['openpyxl.chart'] = MagicMock()

# Now import the module under test
from src.backend.core.enhanced_reports import (
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
    
    with patch('src.backend.core.enhanced_reports.SimpleDocTemplate') as mock_doc:
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
    with patch('src.backend.core.enhanced_reports.json.loads', side_effect=Exception("Fail")):
        path = generator._save_chart("bad json", 1)
        assert path is None
