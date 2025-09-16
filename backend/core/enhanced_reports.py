# Enhanced Report Generation System
# Supports multiple formats: PDF, Excel, PowerPoint with professional templates

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from io import BytesIO
import pandas as pd

# PDF Generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white, grey
from reportlab.lib.units import inch
from reportlab.lib import colors

# Excel Generation
try:
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from openpyxl.chart import BarChart, LineChart, PieChart, Reference
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    logging.warning("openpyxl not available - Excel reports disabled")

class ReportTemplate:
    """Base class for report templates"""
    
    def __init__(self, title: str = "Nexus Analytics Report"):
        self.title = title
        self.created_at = datetime.now()
        self.company_name = "Nexus LLM Analytics"
        self.logo_path = None  # Can be set to include logo
    
    def get_header_style(self):
        """Get header styling"""
        return {
            'font_size': 24,
            'font_color': '#2563eb',
            'font_family': 'Helvetica-Bold'
        }
    
    def get_subheader_style(self):
        """Get subheader styling"""
        return {
            'font_size': 16,
            'font_color': '#374151',
            'font_family': 'Helvetica-Bold'
        }

class PDFReportGenerator:
    """Generate professional PDF reports"""
    
    def __init__(self, template: ReportTemplate = None):
        self.template = template or ReportTemplate()
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom styles for the PDF"""
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=HexColor('#2563eb'),
            alignment=1  # Center
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=HexColor('#374151'),
            borderWidth=0,
            borderColor=HexColor('#e5e7eb'),
            borderPadding=5
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='Subtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=HexColor('#6b7280'),
            alignment=1,
            spaceAfter=20
        ))
        
        # Code style
        self.styles.add(ParagraphStyle(
            name='CodeBlock',
            parent=self.styles['Code'],
            fontSize=10,
            backgroundColor=HexColor('#f3f4f6'),
            borderWidth=1,
            borderColor=HexColor('#d1d5db'),
            borderPadding=10
        ))
    
    def generate_report(self, analysis_results: List[Dict[str, Any]], 
                       output_path: str = None) -> str:
        """Generate a comprehensive PDF report"""
        
        if not output_path:
            output_path = f"nexus_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        story = []
        
        # Title page
        story.extend(self._create_title_page())
        story.append(PageBreak())
        
        # Executive Summary
        story.extend(self._create_executive_summary(analysis_results))
        story.append(PageBreak())
        
        # Table of Contents
        story.extend(self._create_table_of_contents(analysis_results))
        story.append(PageBreak())
        
        # Analysis sections
        for i, result in enumerate(analysis_results, 1):
            story.extend(self._create_analysis_section(result, i))
            if i < len(analysis_results):
                story.append(PageBreak())
        
        # Methodology section
        story.append(PageBreak())
        story.extend(self._create_methodology_section())
        
        # Appendix
        story.append(PageBreak())
        story.extend(self._create_appendix(analysis_results))
        
        # Build PDF
        try:
            doc.build(story)
            logging.info(f"PDF report generated successfully: {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"PDF generation failed: {e}")
            raise e
    
    def _create_title_page(self) -> List:
        """Create the title page"""
        elements = []
        
        # Add logo if available
        if self.template.logo_path and os.path.exists(self.template.logo_path):
            try:
                logo = Image(self.template.logo_path, width=2*inch, height=1*inch)
                logo.hAlign = 'CENTER'
                elements.append(logo)
                elements.append(Spacer(1, 0.5*inch))
            except:
                pass
        
        # Company name
        elements.append(Paragraph(self.template.company_name, self.styles['CustomTitle']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Report title
        elements.append(Paragraph(self.template.title, self.styles['CustomTitle']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Subtitle
        subtitle = f"Generated on {self.template.created_at.strftime('%B %d, %Y at %I:%M %p')}"
        elements.append(Paragraph(subtitle, self.styles['Subtitle']))
        
        elements.append(Spacer(1, 2*inch))
        
        # Add some professional text
        intro_text = """
        This report contains comprehensive data analysis results generated by our 
        AI-powered analytics platform. The insights presented here are based on 
        advanced machine learning algorithms and statistical analysis techniques.
        """
        elements.append(Paragraph(intro_text, self.styles['Normal']))
        
        return elements
    
    def _create_executive_summary(self, results: List[Dict[str, Any]]) -> List:
        """Create executive summary section"""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['CustomTitle']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Generate summary based on results
        total_analyses = len(results)
        successful_analyses = sum(1 for r in results if r.get('success', True))
        
        summary_text = f"""
        This report presents the results of {total_analyses} data analysis operations 
        performed using our advanced AI analytics platform. {successful_analyses} of these 
        analyses completed successfully, providing valuable insights into the data patterns 
        and characteristics.
        
        <b>Key Highlights:</b>
        • Multi-agent AI system utilized for comprehensive analysis
        • Secure sandboxed execution environment
        • Interactive visualizations and statistical summaries
        • Professional reporting with actionable insights
        """
        
        elements.append(Paragraph(summary_text, self.styles['Normal']))
        elements.append(Spacer(1, 0.5*inch))
        
        # Key metrics table if available
        if results:
            metrics_data = [['Metric', 'Value']]
            metrics_data.append(['Total Analyses', str(total_analyses)])
            metrics_data.append(['Success Rate', f"{(successful_analyses/total_analyses)*100:.1f}%"])
            
            # Add more metrics based on results
            for result in results:
                if result.get('execution_time'):
                    metrics_data.append(['Avg. Execution Time', f"{result['execution_time']:.2f}s"])
                    break
            
            metrics_table = Table(metrics_data, colWidths=[2*inch, 2*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2563eb')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8fafc')),
                ('GRID', (0, 0), (-1, -1), 1, HexColor('#e2e8f0'))
            ]))
            
            elements.append(metrics_table)
        
        return elements
    
    def _create_table_of_contents(self, results: List[Dict[str, Any]]) -> List:
        """Create table of contents"""
        elements = []
        
        elements.append(Paragraph("Table of Contents", self.styles['CustomTitle']))
        elements.append(Spacer(1, 0.3*inch))
        
        toc_data = [
            ['Section', 'Page'],
            ['Executive Summary', '2'],
            ['Table of Contents', '3']
        ]
        
        page_num = 4
        for i, result in enumerate(results, 1):
            title = f"Analysis {i}: {result.get('query', 'Data Analysis')[:50]}"
            toc_data.append([title, str(page_num)])
            page_num += 1
        
        toc_data.extend([
            ['Methodology', str(page_num)],
            ['Appendix', str(page_num + 1)]
        ])
        
        toc_table = Table(toc_data, colWidths=[5*inch, 1*inch])
        toc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2563eb')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#e2e8f0'))
        ]))
        
        elements.append(toc_table)
        
        return elements
    
    def _create_analysis_section(self, result: Dict[str, Any], section_num: int) -> List:
        """Create individual analysis section"""
        elements = []
        
        # Section title
        query_title = result.get('query', f'Analysis {section_num}')
        elements.append(Paragraph(f"Analysis {section_num}: {query_title}", 
                                self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Analysis details
        if result.get('filename'):
            elements.append(Paragraph(f"<b>Data Source:</b> {result['filename']}", 
                                    self.styles['Normal']))
            elements.append(Spacer(1, 0.1*inch))
        
        if result.get('type'):
            elements.append(Paragraph(f"<b>Analysis Type:</b> {result['type']}", 
                                    self.styles['Normal']))
            elements.append(Spacer(1, 0.1*inch))
        
        # Results
        elements.append(Paragraph("<b>Results:</b>", self.styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))
        
        result_text = str(result.get('result', 'No results available'))
        if len(result_text) > 1000:
            result_text = result_text[:1000] + "... (truncated)"
        
        elements.append(Paragraph(result_text, self.styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Code section
        if result.get('code') or result.get('generated_code'):
            elements.append(Paragraph("<b>Generated Code:</b>", self.styles['Normal']))
            elements.append(Spacer(1, 0.1*inch))
            
            code = result.get('code') or result.get('generated_code')
            elements.append(Paragraph(f"<pre>{code}</pre>", self.styles['CodeBlock']))
            elements.append(Spacer(1, 0.2*inch))
        
        # Explanation
        if result.get('explanation'):
            elements.append(Paragraph("<b>Explanation:</b>", self.styles['Normal']))
            elements.append(Spacer(1, 0.1*inch))
            elements.append(Paragraph(result['explanation'], self.styles['Normal']))
        
        return elements
    
    def _create_methodology_section(self) -> List:
        """Create methodology section"""
        elements = []
        
        elements.append(Paragraph("Methodology", self.styles['CustomTitle']))
        elements.append(Spacer(1, 0.3*inch))
        
        methodology_text = """
        <b>AI-Powered Multi-Agent Analysis</b><br/>
        Our analytics platform employs a sophisticated multi-agent AI system that combines 
        multiple specialized agents to provide comprehensive data analysis:
        
        <b>1. Data Agent:</b> Responsible for data loading, cleaning, and basic statistical operations.
        
        <b>2. Analysis Agent:</b> Performs complex analytical operations and generates insights.
        
        <b>3. Review Agent:</b> Validates code and results for accuracy and security.
        
        <b>4. Visualization Agent:</b> Creates interactive charts and graphs.
        
        <b>5. Report Agent:</b> Compiles results into professional reports.
        
        <b>Security & Privacy:</b><br/>
        All code execution occurs in a secure sandboxed environment using RestrictedPython 
        to prevent unauthorized system access. Data processing is performed locally to 
        ensure privacy and security.
        
        <b>Quality Assurance:</b><br/>
        Each analysis undergoes multiple validation steps including code review, 
        result verification, and statistical accuracy checks.
        """
        
        elements.append(Paragraph(methodology_text, self.styles['Normal']))
        
        return elements
    
    def _create_appendix(self, results: List[Dict[str, Any]]) -> List:
        """Create appendix with technical details"""
        elements = []
        
        elements.append(Paragraph("Appendix", self.styles['CustomTitle']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Technical specifications
        tech_specs = """
        <b>Technical Specifications:</b><br/>
        • Platform: Nexus LLM Analytics<br/>
        • AI Models: Llama 3.1 8B, Phi-3 Mini<br/>
        • Data Processing: Pandas, NumPy<br/>
        • Visualization: Plotly, Matplotlib<br/>
        • Security: RestrictedPython Sandbox<br/>
        • Report Generation: ReportLab PDF<br/>
        """
        
        elements.append(Paragraph(tech_specs, self.styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Raw results (summary)
        elements.append(Paragraph("<b>Raw Analysis Results (Summary):</b>", self.styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))
        
        for i, result in enumerate(results, 1):
            summary = {
                'analysis_id': i,
                'query': result.get('query', 'N/A'),
                'filename': result.get('filename', 'N/A'),
                'success': result.get('success', True),
                'execution_time': result.get('execution_time', 'N/A')
            }
            
            elements.append(Paragraph(f"<b>Analysis {i}:</b> {json.dumps(summary, indent=2)}", 
                                    self.styles['CodeBlock']))
            elements.append(Spacer(1, 0.1*inch))
        
        return elements

class ExcelReportGenerator:
    """Generate professional Excel reports with charts"""
    
    def __init__(self, template: ReportTemplate = None):
        if not EXCEL_AVAILABLE:
            raise ImportError("openpyxl is required for Excel report generation")
        
        self.template = template or ReportTemplate()
    
    def generate_report(self, analysis_results: List[Dict[str, Any]], 
                       output_path: str = None) -> str:
        """Generate comprehensive Excel report"""
        
        if not output_path:
            output_path = f"nexus_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        workbook = openpyxl.Workbook()
        
        # Remove default sheet
        workbook.remove(workbook.active)
        
        # Create sheets
        self._create_summary_sheet(workbook, analysis_results)
        self._create_results_sheets(workbook, analysis_results)
        self._create_data_sheet(workbook, analysis_results)
        
        # Save workbook
        try:
            workbook.save(output_path)
            logging.info(f"Excel report generated successfully: {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"Excel generation failed: {e}")
            raise e
    
    def _create_summary_sheet(self, workbook, results: List[Dict[str, Any]]):
        """Create summary sheet"""
        ws = workbook.create_sheet("Executive Summary", 0)
        
        # Title
        ws['A1'] = self.template.title
        ws['A1'].font = Font(size=20, bold=True, color='2563EB')
        ws.merge_cells('A1:E1')
        
        # Date
        ws['A2'] = f"Generated: {self.template.created_at.strftime('%Y-%m-%d %H:%M')}"
        ws['A2'].font = Font(size=12, color='6B7280')
        
        # Summary statistics
        row = 4
        ws[f'A{row}'] = "Summary Statistics"
        ws[f'A{row}'].font = Font(size=14, bold=True)
        
        row += 2
        stats_data = [
            ["Total Analyses", len(results)],
            ["Successful Analyses", sum(1 for r in results if r.get('success', True))],
            ["Success Rate", f"{(sum(1 for r in results if r.get('success', True))/len(results))*100:.1f}%"]
        ]
        
        for stat_name, stat_value in stats_data:
            ws[f'A{row}'] = stat_name
            ws[f'B{row}'] = stat_value
            ws[f'A{row}'].font = Font(bold=True)
            row += 1
        
        # Auto-fit columns
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            ws.column_dimensions[column_letter].width = max_length + 2
    
    def _create_results_sheets(self, workbook, results: List[Dict[str, Any]]):
        """Create individual result sheets"""
        for i, result in enumerate(results, 1):
            sheet_name = f"Analysis_{i}"
            ws = workbook.create_sheet(sheet_name)
            
            # Header
            ws['A1'] = f"Analysis {i}: {result.get('query', 'Data Analysis')[:30]}"
            ws['A1'].font = Font(size=16, bold=True)
            
            row = 3
            # Result details
            details = [
                ("Query", result.get('query', 'N/A')),
                ("Filename", result.get('filename', 'N/A')),
                ("Type", result.get('type', 'N/A')),
                ("Success", result.get('success', True)),
                ("Execution Time", f"{result.get('execution_time', 0):.2f}s")
            ]
            
            for label, value in details:
                ws[f'A{row}'] = label
                ws[f'B{row}'] = str(value)
                ws[f'A{row}'].font = Font(bold=True)
                row += 1
            
            # Results
            row += 2
            ws[f'A{row}'] = "Results"
            ws[f'A{row}'].font = Font(size=14, bold=True)
            row += 1
            
            result_text = str(result.get('result', 'No results available'))
            if len(result_text) > 32767:  # Excel cell limit
                result_text = result_text[:32767]
            
            ws[f'A{row}'] = result_text
            ws[f'A{row}'].alignment = Alignment(wrap_text=True)
            
            # Auto-fit columns
            for column in ws.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)  # Cap at 50
                ws.column_dimensions[column_letter].width = adjusted_width
    
    def _create_data_sheet(self, workbook, results: List[Dict[str, Any]]):
        """Create raw data sheet"""
        ws = workbook.create_sheet("Raw Data")
        
        # Headers
        headers = ["Analysis_ID", "Query", "Filename", "Type", "Success", "Execution_Time", "Result_Summary"]
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="2563EB", end_color="2563EB", fill_type="solid")
            cell.font = Font(bold=True, color="FFFFFF")
        
        # Data rows
        for i, result in enumerate(results, 1):
            row_data = [
                i,
                result.get('query', 'N/A'),
                result.get('filename', 'N/A'),
                result.get('type', 'N/A'),
                result.get('success', True),
                result.get('execution_time', 0),
                str(result.get('result', 'N/A'))[:100] + "..." if len(str(result.get('result', ''))) > 100 else str(result.get('result', 'N/A'))
            ]
            
            for col, value in enumerate(row_data, 1):
                ws.cell(row=i+1, column=col, value=value)
        
        # Format table
        table_range = f"A1:{chr(64+len(headers))}{len(results)+1}"
        tab = openpyxl.worksheet.table.Table(
            displayName="AnalysisResults",
            ref=table_range
        )
        ws.add_table(tab)

class EnhancedReportManager:
    """Manage multiple report formats and templates"""
    
    def __init__(self):
        self.pdf_generator = PDFReportGenerator()
        self.excel_generator = ExcelReportGenerator() if EXCEL_AVAILABLE else None
    
    def generate_report(self, analysis_results: List[Dict[str, Any]], 
                       format_type: str = "pdf", 
                       template: ReportTemplate = None,
                       output_path: str = None) -> str:
        """
        Generate report in specified format
        
        Args:
            analysis_results: List of analysis results
            format_type: "pdf", "excel", or "both"
            template: Custom report template
            output_path: Custom output path
            
        Returns:
            Path to generated report(s)
        """
        
        if template:
            self.pdf_generator.template = template
            if self.excel_generator:
                self.excel_generator.template = template
        
        generated_files = []
        
        try:
            if format_type.lower() in ["pdf", "both"]:
                pdf_path = output_path or f"nexus_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf_file = self.pdf_generator.generate_report(analysis_results, pdf_path)
                generated_files.append(pdf_file)
            
            if format_type.lower() in ["excel", "both"] and self.excel_generator:
                excel_path = output_path or f"nexus_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                if format_type.lower() == "both":
                    excel_path = excel_path.replace('.pdf', '.xlsx')
                excel_file = self.excel_generator.generate_report(analysis_results, excel_path)
                generated_files.append(excel_file)
            
            logging.info(f"Report generation completed: {generated_files}")
            return generated_files[0] if len(generated_files) == 1 else generated_files
            
        except Exception as e:
            logging.error(f"Report generation failed: {e}")
            raise e