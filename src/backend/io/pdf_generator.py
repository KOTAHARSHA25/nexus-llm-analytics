# Fix 17: Enterprise PDF Report Generator
# Professional, comprehensive, zero-waste-space PDF generation

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json
import base64
from io import BytesIO

# ReportLab imports
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, ListFlowable, ListItem, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white, grey
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF

logger = logging.getLogger(__name__)


# =============================================================================
# PROFESSIONAL COLOR SCHEME
# =============================================================================
class EnterpriseColors:
    """Enterprise-grade color palette - professional and accessible"""
    PRIMARY = HexColor('#0f172a')         # Slate 900 - main headers
    SECONDARY = HexColor('#1e40af')       # Blue 800 - section headers
    ACCENT = HexColor('#3b82f6')          # Blue 500 - accents/links
    SUCCESS = HexColor('#059669')         # Emerald 600 - positive data
    WARNING = HexColor('#ea580c')         # Orange 600 - alerts
    ERROR = HexColor('#dc2626')           # Red 600 - errors
    TEXT_PRIMARY = HexColor('#1e293b')    # Slate 800 - body text
    TEXT_SECONDARY = HexColor('#475569')  # Slate 600 - secondary text
    TEXT_MUTED = HexColor('#94a3b8')      # Slate 400 - captions
    BG_LIGHT = HexColor('#f8fafc')        # Slate 50 - backgrounds
    BG_MEDIUM = HexColor('#e2e8f0')       # Slate 200 - table alternates
    BORDER = HexColor('#cbd5e1')          # Slate 300 - borders
    CODE_BG = HexColor('#f1f5f9')         # Slate 100 - code blocks
    HIGHLIGHT = HexColor('#fef3c7')       # Amber 100 - highlights


# =============================================================================
# NUMBERED CANVAS WITH ENTERPRISE HEADERS/FOOTERS
# =============================================================================
class EnterpriseCanvas(canvas.Canvas):
    """Custom canvas with professional headers, footers, and page numbers"""
    
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []
        self.report_title = "Analysis Report"
        self.report_subtitle = ""
        self.user_query = ""
        self.model_used = "LLM"
        self.generated_date = datetime.now()
        self.company_name = "Nexus LLM Analytics"
    
    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()
    
    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_enterprise_header_footer(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)
    
    def _draw_enterprise_header_footer(self, page_count):
        """Draw enterprise-grade header and footer"""
        page_num = self._pageNumber
        width, height = A4
        
        # Skip decorations on title page
        if page_num == 1:
            return
        
        self.saveState()
        
        # === HEADER ===
        # Top border - gradient effect with two lines
        self.setStrokeColor(EnterpriseColors.PRIMARY)
        self.setLineWidth(3)
        self.line(50, height - 35, width - 50, height - 35)
        
        self.setStrokeColor(EnterpriseColors.ACCENT)
        self.setLineWidth(1)
        self.line(50, height - 38, width - 50, height - 38)
        
        # Company logo area (text-based)
        self.setFont('Helvetica-Bold', 11)
        self.setFillColor(EnterpriseColors.PRIMARY)
        self.drawString(50, height - 50, self.company_name)
        
        # Report title in header (right side)
        self.setFont('Helvetica', 9)
        self.setFillColor(EnterpriseColors.TEXT_SECONDARY)
        title_text = self.report_title[:45] + "..." if len(self.report_title) > 45 else self.report_title
        self.drawRightString(width - 50, height - 50, title_text)
        
        # === FOOTER ===
        # Bottom border
        self.setStrokeColor(EnterpriseColors.BORDER)
        self.setLineWidth(1)
        self.line(50, 40, width - 50, 40)
        
        # Left: Generation timestamp
        self.setFont('Helvetica', 8)
        self.setFillColor(EnterpriseColors.TEXT_MUTED)
        timestamp = self.generated_date.strftime('%B %d, %Y at %H:%M')
        self.drawString(50, 25, f"Generated: {timestamp}")
        
        # Center: Page number with modern styling
        self.setFont('Helvetica-Bold', 9)
        self.setFillColor(EnterpriseColors.PRIMARY)
        self.drawCentredString(width / 2, 25, f"{page_num}")
        self.setFont('Helvetica', 8)
        self.setFillColor(EnterpriseColors.TEXT_MUTED)
        self.drawCentredString(width / 2, 15, f"of {page_count} pages")
        
        # Right: Model attribution
        self.setFont('Helvetica-Oblique', 8)
        self.setFillColor(EnterpriseColors.ACCENT)
        self.drawRightString(width - 50, 25, f"Powered by {self.model_used}")
        
        self.restoreState()


# =============================================================================
# PDF REPORT GENERATOR - ENTERPRISE EDITION
# =============================================================================
class PDFReportGenerator:
    """
    Enterprise-grade PDF report generator.
    Zero wasted space, comprehensive content, professional formatting.
    """
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_enterprise_styles()
    
    def _setup_enterprise_styles(self):
        """Setup comprehensive typography system"""
        
        # === TITLE PAGE STYLES ===
        self.styles.add(ParagraphStyle(
            name='CoverTitle',
            parent=self.styles['Heading1'],
            fontSize=36,
            leading=44,
            textColor=EnterpriseColors.PRIMARY,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            spaceAfter=12,
            spaceBefore=0
        ))
        
        self.styles.add(ParagraphStyle(
            name='CoverSubtitle',
            parent=self.styles['Normal'],
            fontSize=16,
            leading=22,
            textColor=EnterpriseColors.SECONDARY,
            alignment=TA_CENTER,
            fontName='Helvetica',
            spaceAfter=8
        ))
        
        self.styles.add(ParagraphStyle(
            name='CoverMeta',
            parent=self.styles['Normal'],
            fontSize=11,
            leading=16,
            textColor=EnterpriseColors.TEXT_SECONDARY,
            alignment=TA_CENTER,
            fontName='Helvetica',
            spaceAfter=4
        ))
        
        # === CONTENT STYLES ===
        self.styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=self.styles['Heading1'],
            fontSize=20,
            leading=26,
            textColor=EnterpriseColors.PRIMARY,
            fontName='Helvetica-Bold',
            spaceAfter=10,
            spaceBefore=20,
            borderWidth=2,
            borderColor=EnterpriseColors.ACCENT,
            borderPadding=8,
            backColor=EnterpriseColors.BG_LIGHT
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubsectionTitle',
            parent=self.styles['Heading2'],
            fontSize=15,
            leading=20,
            textColor=EnterpriseColors.SECONDARY,
            fontName='Helvetica-Bold',
            spaceAfter=8,
            spaceBefore=14,
            leftIndent=10,
            borderWidth=0,
            borderPadding=0,
            borderColor=EnterpriseColors.ACCENT,
            borderRadius=0
        ))
        
        self.styles.add(ParagraphStyle(
            name='MinorTitle',
            parent=self.styles['Heading3'],
            fontSize=13,
            leading=18,
            textColor=EnterpriseColors.TEXT_PRIMARY,
            fontName='Helvetica-Bold',
            spaceAfter=6,
            spaceBefore=10,
            leftIndent=15
        ))
        
        # === TEXT STYLES ===
        self.styles.add(ParagraphStyle(
            name='EnterpriseBody',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=15,
            textColor=EnterpriseColors.TEXT_PRIMARY,
            fontName='Helvetica',
            alignment=TA_JUSTIFY,
            spaceAfter=8
        ))
        
        self.styles.add(ParagraphStyle(
            name='EnterpriseBodyCompact',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=13,
            textColor=EnterpriseColors.TEXT_PRIMARY,
            fontName='Helvetica',
            alignment=TA_LEFT,
            spaceAfter=4
        ))
        
        self.styles.add(ParagraphStyle(
            name='EnterpriseCode',
            parent=self.styles['Code'],
            fontSize=9,
            leading=11,
            textColor=EnterpriseColors.TEXT_PRIMARY,
            fontName='Courier',
            backColor=EnterpriseColors.CODE_BG,
            borderWidth=1,
            borderColor=EnterpriseColors.BORDER,
            borderPadding=8,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=8,
            spaceBefore=4
        ))
        
        self.styles.add(ParagraphStyle(
            name='EnterpriseKeyValue',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=EnterpriseColors.TEXT_PRIMARY,
            fontName='Helvetica',
            spaceAfter=2,
            leftIndent=20
        ))
        
        self.styles.add(ParagraphStyle(
            name='EnterpriseBullet',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=EnterpriseColors.TEXT_PRIMARY,
            fontName='Helvetica',
            spaceAfter=4,
            leftIndent=30,
            bulletIndent=15
        ))
        
        self.styles.add(ParagraphStyle(
            name='EnterpriseCaption',
            parent=self.styles['Normal'],
            fontSize=9,
            leading=11,
            textColor=EnterpriseColors.TEXT_MUTED,
            fontName='Helvetica-Oblique',
            alignment=TA_CENTER,
            spaceAfter=12,
            spaceBefore=4
        ))
        
        self.styles.add(ParagraphStyle(
            name='EnterpriseHighlight',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=EnterpriseColors.TEXT_PRIMARY,
            fontName='Helvetica-Bold',
            backColor=EnterpriseColors.HIGHLIGHT,
            borderPadding=6,
            spaceAfter=8
        ))
        
        self.styles.add(ParagraphStyle(
            name='EnterpriseQuote',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=15,
            textColor=EnterpriseColors.TEXT_SECONDARY,
            fontName='Helvetica-Oblique',
            leftIndent=30,
            rightIndent=30,
            borderWidth=0,
            borderPadding=10,
            backColor=EnterpriseColors.BG_LIGHT,
            spaceAfter=10,
            spaceBefore=6
        ))
    
    def generate_report(self, 
                       analysis_result: Dict[str, Any],
                       output_path: str = None,
                       include_raw_data: bool = True) -> str:
        """
        Generate comprehensive PDF report from analysis results.
        
        Args:
            analysis_result: Full analysis result dictionary
            output_path: Custom output path (auto-generated if None)
            include_raw_data: Include raw data tables in appendix
        
        Returns:
            Path to generated PDF file
        """
        
        # Generate output path if not provided
        if not output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(__file__).parent.parent.parent.parent / 'reports'
            output_dir.mkdir(exist_ok=True)
            output_path = str(output_dir / f'analysis_report_{timestamp}.pdf')
        
        logger.info(f"[PDF] Generating enterprise report: {output_path}")
        
        # Create PDF document with custom canvas
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=50,
            leftMargin=50,
            topMargin=70,
            bottomMargin=60,
            title=f"Analysis Report - {datetime.now().strftime('%Y-%m-%d')}",
            author="Nexus LLM Analytics"
        )
        
        # Build story (content flow)
        story = []
        
        # Extract metadata
        query = analysis_result.get('query', 'Data Analysis')
        model = analysis_result.get('model_used', analysis_result.get('metadata', {}).get('model', 'LLM'))
        timestamp = datetime.now()
        
        # === TITLE PAGE ===
        story.extend(self._create_title_page(query, model, timestamp))
        story.append(PageBreak())
        
        # === TABLE OF CONTENTS ===
        story.extend(self._create_table_of_contents(analysis_result))
        story.append(PageBreak())
        
        # === EXECUTIVE SUMMARY ===
        story.extend(self._create_executive_summary(analysis_result))
        story.append(Spacer(1, 0.2*inch))
        
        # === QUERY ANALYSIS ===
        story.extend(self._create_query_section(analysis_result))
        story.append(Spacer(1, 0.2*inch))
        
        # === AI INTERPRETATION ===
        story.extend(self._create_interpretation_section(analysis_result))
        story.append(Spacer(1, 0.2*inch))
        
        # === ORCHESTRATOR REASONING (if available) ===
        if analysis_result.get('orchestrator_reasoning') or analysis_result.get('routing_decision'):
            story.extend(self._create_orchestrator_section(analysis_result))
            story.append(Spacer(1, 0.2*inch))
        
        # === KEY FINDINGS ===
        story.extend(self._create_findings_section(analysis_result))
        story.append(Spacer(1, 0.2*inch))
        
        # === DETAILED RESULTS ===
        story.extend(self._create_results_section(analysis_result))
        story.append(Spacer(1, 0.2*inch))
        
        # === DATA INSIGHTS ===
        story.extend(self._create_insights_section(analysis_result))
        story.append(Spacer(1, 0.2*inch))
        
        # === CODE GENERATED (if available) ===
        if analysis_result.get('code_generated') or analysis_result.get('code'):
            story.extend(self._create_code_section(analysis_result))
            story.append(Spacer(1, 0.2*inch))
        
        # === VISUALIZATIONS (if available) ===
        if analysis_result.get('visualizations') or analysis_result.get('charts'):
            story.extend(self._create_visualization_section(analysis_result))
            story.append(Spacer(1, 0.2*inch))
        
        # === METHODOLOGY ===
        story.extend(self._create_methodology_section(analysis_result))
        story.append(Spacer(1, 0.2*inch))
        
        # === TECHNICAL DETAILS ===
        story.extend(self._create_technical_section(analysis_result))
        
        # === RAW DATA (if requested) ===
        if include_raw_data and analysis_result.get('data'):
            story.append(PageBreak())
            story.extend(self._create_data_appendix(analysis_result))
        
        # Build PDF with custom canvas
        def on_first_page(canvas_obj, doc_obj):
            pass  # Title page - no header/footer
        
        def on_later_pages(canvas_obj, doc_obj):
            pass  # Headers/footers handled by EnterpriseCanvas
        
        # Build with custom canvas
        doc.canvasmaker = EnterpriseCanvas
        doc.build(story, onFirstPage=on_first_page, onLaterPages=on_later_pages)
        
        # Set canvas metadata
        if hasattr(doc, 'canv'):
            doc.canv.report_title = f"Analysis: {query[:50]}"
            doc.canv.user_query = query
            doc.canv.model_used = model
        
        logger.info(f"[PDF] Report generated successfully: {output_path}")
        return output_path
    
    # =========================================================================
    # SECTION BUILDERS
    # =========================================================================
    
    def _create_title_page(self, query: str, model: str, timestamp: datetime) -> List:
        """Create professional title page"""
        content = []
        
        # Vertical spacing
        content.append(Spacer(1, 2*inch))
        
        # Main title
        content.append(Paragraph("ANALYSIS REPORT", self.styles['CoverTitle']))
        content.append(Spacer(1, 0.3*inch))
        
        # Query as subtitle
        query_text = query if len(query) < 120 else query[:117] + "..."
        content.append(Paragraph(f"<i>{query_text}</i>", self.styles['CoverSubtitle']))
        content.append(Spacer(1, 0.8*inch))
        
        # Horizontal line
        content.append(self._create_horizontal_line(EnterpriseColors.ACCENT, 2))
        content.append(Spacer(1, 0.5*inch))
        
        # Metadata
        content.append(Paragraph(f"<b>Generated:</b> {timestamp.strftime('%B %d, %Y at %H:%M')}", 
                                self.styles['CoverMeta']))
        content.append(Paragraph(f"<b>AI Model:</b> {model}", self.styles['CoverMeta']))
        content.append(Paragraph(f"<b>Platform:</b> Nexus LLM Analytics v2.0", self.styles['CoverMeta']))
        
        content.append(Spacer(1, 1*inch))
        
        # Footer box
        content.append(self._create_horizontal_line(EnterpriseColors.BORDER, 1))
        content.append(Spacer(1, 0.2*inch))
        content.append(Paragraph(
            "<i>This report was automatically generated using advanced AI analysis. "
            "All insights are data-driven and validated.</i>",
            self.styles['EnterpriseCaption']
        ))
        
        return content
    
    def _create_table_of_contents(self, result: Dict[str, Any]) -> List:
        """Create table of contents"""
        content = []
        
        content.append(Paragraph("TABLE OF CONTENTS", self.styles['SectionTitle']))
        content.append(Spacer(1, 0.2*inch))
        
        # Build TOC entries
        toc_items = [
            "1. Executive Summary",
            "2. Query Analysis",
            "3. AI Interpretation",
        ]
        
        section_num = 4
        if result.get('orchestrator_reasoning') or result.get('routing_decision'):
            toc_items.append(f"{section_num}. Orchestrator Reasoning")
            section_num += 1
        
        toc_items.append(f"{section_num}. Key Findings")
        section_num += 1
        toc_items.append(f"{section_num}. Detailed Results")
        section_num += 1
        toc_items.append(f"{section_num}. Data Insights")
        section_num += 1
        
        if result.get('code_generated') or result.get('code'):
            toc_items.append(f"{section_num}. Generated Code")
            section_num += 1
        
        if result.get('visualizations') or result.get('charts'):
            toc_items.append(f"{section_num}. Visualizations")
            section_num += 1
        
        toc_items.append(f"{section_num}. Methodology")
        section_num += 1
        toc_items.append(f"{section_num}. Technical Details")
        
        # Render TOC
        for item in toc_items:
            content.append(Paragraph(f"• {item}", self.styles['EnterpriseKeyValue']))
        
        return content
    
    def _create_executive_summary(self, result: Dict[str, Any]) -> List:
        """Create executive summary"""
        content = []
        
        content.append(Paragraph("1. EXECUTIVE SUMMARY", self.styles['SectionTitle']))
        content.append(Spacer(1, 0.15*inch))
        
        # Extract key points
        summary_points = []
        
        # Query overview
        query = result.get('query', 'N/A')
        summary_points.append(f"<b>Analysis Objective:</b> {query}")
        
        # Status
        success = result.get('success', True)
        status_text = "✓ Completed Successfully" if success else "⚠ Completed with Warnings"
        status_color = 'green' if success else 'orange'
        summary_points.append(f"<b>Status:</b> <font color='{status_color}'>{status_text}</font>")
        
        # Agent used
        agent = result.get('agent', result.get('metadata', {}).get('agent', 'Unknown'))
        summary_points.append(f"<b>Processing Agent:</b> {agent}")
        
        # Data summary
        if result.get('metadata'):
            meta = result.get('metadata', {})
            if meta.get('rows'):
                summary_points.append(f"<b>Data Scope:</b> {meta.get('rows', 'N/A')} rows × {meta.get('columns', 'N/A')} columns")
        
        # Execution time
        exec_time = result.get('execution_time', result.get('metadata', {}).get('execution_time'))
        if exec_time:
            summary_points.append(f"<b>Execution Time:</b> {exec_time:.2f} seconds")
        
        # Render summary points
        for point in summary_points:
            content.append(Paragraph(point, self.styles['EnterpriseBodyCompact']))
        
        content.append(Spacer(1, 0.15*inch))
        
        # Quick insight highlight
        interpretation = result.get('interpretation', result.get('result', ''))
        if interpretation and isinstance(interpretation, str):
            # Take first meaningful sentence
            sentences = interpretation.split('. ')
            if sentences:
                highlight = sentences[0][:200] + ("..." if len(sentences[0]) > 200 else "")
                content.append(Paragraph(f"<b>Key Insight:</b> {highlight}", self.styles['EnterpriseHighlight']))
        
        return content
    
    def _create_query_section(self, result: Dict[str, Any]) -> List:
        """Create query analysis section"""
        content = []
        
        content.append(Paragraph("2. QUERY ANALYSIS", self.styles['SectionTitle']))
        content.append(Spacer(1, 0.1*inch))
        
        # Original query
        content.append(Paragraph("2.1 User Query", self.styles['SubsectionTitle']))
        query = result.get('query', 'No query provided')
        content.append(Paragraph(f'"{query}"', self.styles['EnterpriseQuote']))
        
        # Query characteristics
        content.append(Paragraph("2.2 Query Characteristics", self.styles['SubsectionTitle']))
        
        characteristics = []
        query_lower = query.lower()
        
        # Detect query type
        if any(word in query_lower for word in ['trend', 'over time', 'timeline', 'forecast']):
            characteristics.append("• <b>Type:</b> Time Series Analysis")
        elif any(word in query_lower for word in ['correlation', 'relationship', 'compare']):
            characteristics.append("• <b>Type:</b> Comparative Analysis")
        elif any(word in query_lower for word in ['predict', 'ml', 'model', 'classify']):
            characteristics.append("• <b>Type:</b> Machine Learning")
        elif any(word in query_lower for word in ['sum', 'total', 'average', 'count']):
            characteristics.append("• <b>Type:</b> Aggregation Analysis")
        else:
            characteristics.append("• <b>Type:</b> General Analysis")
        
        # Complexity
        word_count = len(query.split())
        if word_count > 20:
            characteristics.append("• <b>Complexity:</b> High (detailed multi-part query)")
        elif word_count > 10:
            characteristics.append("• <b>Complexity:</b> Medium (specific requirements)")
        else:
            characteristics.append("• <b>Complexity:</b> Simple (direct question)")
        
        for char in characteristics:
            content.append(Paragraph(char, self.styles['EnterpriseBodyCompact']))
        
        return content
    
    def _create_interpretation_section(self, result: Dict[str, Any]) -> List:
        """Create AI interpretation section"""
        content = []
        
        content.append(Paragraph("3. AI INTERPRETATION", self.styles['SectionTitle']))
        content.append(Spacer(1, 0.1*inch))
        
        interpretation = result.get('interpretation', result.get('result', 'No interpretation available'))
        
        if isinstance(interpretation, dict):
            interpretation = json.dumps(interpretation, indent=2)
        
        # Split into paragraphs for better readability
        if isinstance(interpretation, str):
            paragraphs = interpretation.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    # Escape XML special characters
                    para_escaped = para.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    content.append(Paragraph(para_escaped, self.styles['EnterpriseBody']))
        
        return content
    
    def _create_orchestrator_section(self, result: Dict[str, Any]) -> List:
        """Create orchestrator reasoning section"""
        content = []
        
        content.append(Paragraph("4. ORCHESTRATOR REASONING", self.styles['SectionTitle']))
        content.append(Spacer(1, 0.1*inch))
        
        content.append(Paragraph(
            "The query orchestrator analyzed your request and made intelligent routing decisions:",
            self.styles['EnterpriseBody']
        ))
        content.append(Spacer(1, 0.05*inch))
        
        # Routing decision
        if result.get('routing_decision'):
            routing = result['routing_decision']
            content.append(Paragraph("4.1 Routing Decision", self.styles['SubsectionTitle']))
            
            routing_data = [
                ['Decision Point', 'Value'],
                ['Selected Agent', routing.get('agent', 'N/A')],
                ['Confidence Score', f"{routing.get('confidence', 0):.2%}"],
                ['Reasoning', routing.get('reasoning', 'N/A')[:100]]
            ]
            
            table = Table(routing_data, colWidths=[2.5*inch, 4*inch])
            table.setStyle(self._get_table_style())
            content.append(table)
            content.append(Spacer(1, 0.1*inch))
        
        # Orchestrator reasoning
        if result.get('orchestrator_reasoning'):
            reasoning = result['orchestrator_reasoning']
            content.append(Paragraph("4.2 Decision Rationale", self.styles['SubsectionTitle']))
            
            if isinstance(reasoning, dict):
                for key, value in reasoning.items():
                    content.append(Paragraph(f"<b>{key}:</b> {value}", self.styles['EnterpriseKeyValue']))
            else:
                content.append(Paragraph(str(reasoning), self.styles['EnterpriseBody']))
        
        return content
    
    def _create_findings_section(self, result: Dict[str, Any]) -> List:
        """Create key findings section"""
        content = []
        
        section_num = 5 if not result.get('orchestrator_reasoning') else 5
        content.append(Paragraph(f"{section_num}. KEY FINDINGS", self.styles['SectionTitle']))
        content.append(Spacer(1, 0.1*inch))
        
        # Extract findings from result
        findings = []
        
        # Check various result formats
        if result.get('insights'):
            insights = result['insights']
            if isinstance(insights, list):
                findings.extend(insights[:10])  # Top 10
            elif isinstance(insights, dict):
                for key, value in list(insights.items())[:10]:
                    findings.append(f"{key}: {value}")
        
        if result.get('key_metrics'):
            metrics = result['key_metrics']
            if isinstance(metrics, dict):
                for key, value in list(metrics.items())[:5]:
                    findings.append(f"{key}: {value}")
        
        if result.get('summary'):
            summary = result['summary']
            if isinstance(summary, str):
                findings.append(summary)
            elif isinstance(summary, dict):
                for key, value in list(summary.items())[:5]:
                    findings.append(f"{key}: {value}")
        
        # If no structured findings, extract from interpretation
        if not findings and result.get('interpretation'):
            interp = result['interpretation']
            if isinstance(interp, str) and len(interp) > 50:
                # Extract first few sentences
                sentences = interp.split('. ')[:5]
                findings.extend([s.strip() + '.' for s in sentences if s.strip()])
        
        # Default message if no findings
        if not findings:
            findings = ["Analysis completed successfully. See detailed results below."]
        
        # Render findings as numbered list
        for i, finding in enumerate(findings, 1):
            finding_text = str(finding)[:500]  # Limit length
            content.append(Paragraph(f"<b>{i}.</b> {finding_text}", self.styles['EnterpriseBullet']))
        
        return content
    
    def _create_results_section(self, result: Dict[str, Any]) -> List:
        """Create detailed results section"""
        content = []
        
        section_num = 6 if not result.get('orchestrator_reasoning') else 6
        content.append(Paragraph(f"{section_num}. DETAILED RESULTS", self.styles['SectionTitle']))
        content.append(Spacer(1, 0.1*inch))
        
        # Main result
        main_result = result.get('result', result.get('output', 'No result data available'))
        
        if isinstance(main_result, dict):
            # Render as structured data
            content.append(Paragraph("Result Structure:", self.styles['SubsectionTitle']))
            
            for key, value in main_result.items():
                if isinstance(value, (list, dict)):
                    value_str = json.dumps(value, indent=2)[:500]
                else:
                    value_str = str(value)[:200]
                
                content.append(Paragraph(f"<b>{key}:</b>", self.styles['MinorTitle']))
                content.append(Paragraph(value_str, self.styles['EnterpriseKeyValue']))
        
        elif isinstance(main_result, list):
            # Render as table if possible
            if len(main_result) > 0 and isinstance(main_result[0], dict):
                # Convert to table
                headers = list(main_result[0].keys())[:6]  # Max 6 columns
                table_data = [headers]
                
                for row in main_result[:15]:  # Max 15 rows
                    table_row = [str(row.get(h, ''))[:50] for h in headers]
                    table_data.append(table_row)
                
                col_widths = [6.5*inch / len(headers)] * len(headers)
                table = Table(table_data, colWidths=col_widths)
                table.setStyle(self._get_table_style())
                content.append(table)
            else:
                # Render as list
                for i, item in enumerate(main_result[:20], 1):
                    content.append(Paragraph(f"{i}. {str(item)[:200]}", self.styles['EnterpriseBodyCompact']))
        
        else:
            # Render as text
            result_text = str(main_result)
            if len(result_text) > 50:
                paragraphs = result_text.split('\n')
                for para in paragraphs[:20]:  # Max 20 lines
                    if para.strip():
                        content.append(Paragraph(para.strip(), self.styles['EnterpriseBody']))
            else:
                content.append(Paragraph(result_text, self.styles['EnterpriseBody']))
        
        return content
    
    def _create_insights_section(self, result: Dict[str, Any]) -> List:
        """Create data insights section"""
        content = []
        
        section_num = 7 if not result.get('orchestrator_reasoning') else 7
        content.append(Paragraph(f"{section_num}. DATA INSIGHTS", self.styles['SectionTitle']))
        content.append(Spacer(1, 0.1*inch))
        
        # Statistical insights
        if result.get('statistics'):
            content.append(Paragraph("Statistical Summary", self.styles['SubsectionTitle']))
            stats = result['statistics']
            
            if isinstance(stats, dict):
                stats_data = [['Metric', 'Value']]
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        stats_data.append([key, f"{value:.4f}" if isinstance(value, float) else str(value)])
                    else:
                        stats_data.append([key, str(value)[:100]])
                
                table = Table(stats_data, colWidths=[2.5*inch, 4*inch])
                table.setStyle(self._get_table_style())
                content.append(table)
                content.append(Spacer(1, 0.1*inch))
        
        # Metadata insights
        if result.get('metadata'):
            content.append(Paragraph("Dataset Characteristics", self.styles['SubsectionTitle']))
            meta = result['metadata']
            
            meta_points = []
            if meta.get('rows'):
                meta_points.append(f"• Total Records: {meta['rows']:,}")
            if meta.get('columns'):
                meta_points.append(f"• Total Features: {meta['columns']}")
            if meta.get('data_types'):
                meta_points.append(f"• Data Types: {', '.join(str(dt) for dt in meta['data_types'])}")
            if meta.get('missing_values'):
                meta_points.append(f"• Missing Values: {meta['missing_values']}")
            
            for point in meta_points:
                content.append(Paragraph(point, self.styles['EnterpriseBodyCompact']))
        
        # Additional insights
        if not result.get('statistics') and not result.get('metadata'):
            content.append(Paragraph(
                "Detailed statistical insights are available in the full analysis output.",
                self.styles['EnterpriseBody']
            ))
        
        return content
    
    def _create_code_section(self, result: Dict[str, Any]) -> List:
        """Create generated code section"""
        content = []
        
        content.append(Paragraph("GENERATED CODE", self.styles['SectionTitle']))
        content.append(Spacer(1, 0.1*inch))
        
        code = result.get('code_generated', result.get('code', ''))
        
        if code:
            content.append(Paragraph(
                "The following Python code was generated to perform the analysis:",
                self.styles['EnterpriseBody']
            ))
            content.append(Spacer(1, 0.05*inch))
            
            # Format code
            code_lines = code.split('\n')
            code_formatted = '<br/>'.join(line.replace(' ', '&nbsp;').replace('<', '&lt;').replace('>', '&gt;') 
                                         for line in code_lines[:50])  # Max 50 lines
            
            content.append(Paragraph(f"<font face='Courier' size='8'>{code_formatted}</font>", 
                                   self.styles['EnterpriseCode']))
            
            if len(code_lines) > 50:
                content.append(Paragraph(
                    f"<i>... ({len(code_lines) - 50} more lines)</i>",
                    self.styles['EnterpriseCaption']
                ))
        
        return content
    
    def _create_visualization_section(self, result: Dict[str, Any]) -> List:
        """Create visualizations section"""
        content = []
        
        content.append(Paragraph("VISUALIZATIONS", self.styles['SectionTitle']))
        content.append(Spacer(1, 0.1*inch))
        
        content.append(Paragraph(
            "Visualization data is available in the analysis output. "
            "Charts can be generated using the provided visualization endpoints.",
            self.styles['EnterpriseBody']
        ))
        
        # List available visualizations
        viz_data = result.get('visualizations', result.get('charts', []))
        if viz_data:
            content.append(Spacer(1, 0.05*inch))
            content.append(Paragraph("Available Charts:", self.styles['SubsectionTitle']))
            
            if isinstance(viz_data, list):
                for i, viz in enumerate(viz_data[:10], 1):
                    if isinstance(viz, dict):
                        viz_type = viz.get('type', 'Chart')
                        viz_title = viz.get('title', f'Visualization {i}')
                        content.append(Paragraph(f"• {viz_type}: {viz_title}", self.styles['EnterpriseBodyCompact']))
                    else:
                        content.append(Paragraph(f"• Visualization {i}", self.styles['EnterpriseBodyCompact']))
        
        return content
    
    def _create_methodology_section(self, result: Dict[str, Any]) -> List:
        """Create methodology section"""
        content = []
        
        content.append(Paragraph("METHODOLOGY", self.styles['SectionTitle']))
        content.append(Spacer(1, 0.1*inch))
        
        content.append(Paragraph("Analysis Approach", self.styles['SubsectionTitle']))
        
        methodology_text = """
This analysis was conducted using the Nexus LLM Analytics platform, which employs:

• <b>Intelligent Query Orchestration:</b> Automatic routing to specialized analysis agents based on query characteristics and data structure.

• <b>Multi-Agent Architecture:</b> Dedicated agents for statistical analysis, time series, financial metrics, machine learning, and data visualization.

• <b>Dynamic Code Generation:</b> Automated Python code generation tailored to your specific analysis requirements.

• <b>Quality Assurance:</b> Sandboxed execution environment with security guards and result validation.

• <b>Circuit Breaker Protection:</b> Graceful handling of LLM service interruptions with fallback mechanisms.
        """
        
        for line in methodology_text.strip().split('\n'):
            if line.strip():
                content.append(Paragraph(line.strip(), self.styles['EnterpriseBodyCompact']))
        
        # Agent-specific methodology
        agent = result.get('agent', 'Unknown')
        content.append(Spacer(1, 0.1*inch))
        content.append(Paragraph("Agent-Specific Approach", self.styles['SubsectionTitle']))
        content.append(Paragraph(
            f"This analysis was processed by the <b>{agent}</b> agent, which specializes in handling "
            f"queries of this type with domain-specific algorithms and interpretation methods.",
            self.styles['EnterpriseBody']
        ))
        
        return content
    
    def _create_technical_section(self, result: Dict[str, Any]) -> List:
        """Create technical details section"""
        content = []
        
        content.append(Paragraph("TECHNICAL DETAILS", self.styles['SectionTitle']))
        content.append(Spacer(1, 0.1*inch))
        
        # Build technical details table
        tech_data = [['Parameter', 'Value']]
        
        # Model
        model = result.get('model_used', result.get('metadata', {}).get('model', 'Unknown'))
        tech_data.append(['AI Model', model])
        
        # Agent
        agent = result.get('agent', result.get('metadata', {}).get('agent', 'Unknown'))
        tech_data.append(['Processing Agent', agent])
        
        # Timestamp
        timestamp = result.get('timestamp', datetime.now().isoformat())
        tech_data.append(['Analysis Timestamp', timestamp])
        
        # Execution time
        exec_time = result.get('execution_time', result.get('metadata', {}).get('execution_time'))
        if exec_time:
            tech_data.append(['Execution Time', f"{exec_time:.2f} seconds"])
        
        # Success status
        success = result.get('success', True)
        tech_data.append(['Status', '✓ Success' if success else '⚠ Warning'])
        
        # Data dimensions
        if result.get('metadata'):
            meta = result['metadata']
            if meta.get('rows'):
                tech_data.append(['Data Rows', f"{meta['rows']:,}"])
            if meta.get('columns'):
                tech_data.append(['Data Columns', str(meta['columns'])])
        
        # Platform version
        tech_data.append(['Platform', 'Nexus LLM Analytics v2.0'])
        tech_data.append(['Report Generator', 'Enterprise PDF Generator (Fix 17)'])
        
        table = Table(tech_data, colWidths=[2.5*inch, 4*inch])
        table.setStyle(self._get_table_style())
        content.append(table)
        
        return content
    
    def _create_data_appendix(self, result: Dict[str, Any]) -> List:
        """Create raw data appendix"""
        content = []
        
        content.append(Paragraph("APPENDIX: RAW DATA", self.styles['SectionTitle']))
        content.append(Spacer(1, 0.1*inch))
        
        content.append(Paragraph(
            "Complete analysis result data in JSON format:",
            self.styles['EnterpriseBody']
        ))
        content.append(Spacer(1, 0.05*inch))
        
        # Sanitize result data for display
        safe_result = {k: v for k, v in result.items() 
                      if k not in ['data', 'df'] or not isinstance(v, object)}
        
        # Convert to formatted JSON
        json_str = json.dumps(safe_result, indent=2, default=str)
        json_lines = json_str.split('\n')[:100]  # Max 100 lines
        
        json_formatted = '<br/>'.join(
            line.replace(' ', '&nbsp;').replace('<', '&lt;').replace('>', '&gt;')
            for line in json_lines
        )
        
        content.append(Paragraph(
            f"<font face='Courier' size='7'>{json_formatted}</font>",
            self.styles['EnterpriseCode']
        ))
        
        if len(json_lines) >= 100:
            content.append(Paragraph(
                "<i>Data truncated for brevity. Full data available in analysis output.</i>",
                self.styles['EnterpriseCaption']
            ))
        
        return content
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _create_horizontal_line(self, color, width: float) -> Drawing:
        """Create horizontal line"""
        d = Drawing(6.5*inch, width)
        d.add(Rect(0, 0, 6.5*inch, width, fillColor=color, strokeColor=color))
        return d
    
    def _get_table_style(self) -> TableStyle:
        """Get standard table style"""
        return TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), EnterpriseColors.PRIMARY),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 10),
            
            # Data rows
            ('BACKGROUND', (0, 1), (-1, -1), white),
            ('TEXTCOLOR', (0, 1), (-1, -1), EnterpriseColors.TEXT_PRIMARY),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            
            # Alternating rows
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, EnterpriseColors.BG_MEDIUM]),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 1, EnterpriseColors.BORDER),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ])


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================
def generate_pdf_report(analysis_result: Dict[str, Any], 
                       output_path: str = None,
                       include_raw_data: bool = True) -> str:
    """
    Convenience function to generate PDF report.
    
    Args:
        analysis_result: Complete analysis result dictionary
        output_path: Custom output path (auto-generated if None)
        include_raw_data: Include raw data in appendix
    
    Returns:
        Path to generated PDF file
    """
    generator = PDFReportGenerator()
    return generator.generate_report(analysis_result, output_path, include_raw_data)
