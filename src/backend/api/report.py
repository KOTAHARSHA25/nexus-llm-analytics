"""Report API — Professional Report Generation & Download Endpoints
=================================================================

Generates enterprise-grade PDF, Excel, and CSV reports from analysis
results. Supports both the enhanced multi-result report manager and
the single-result enterprise PDF generator.

Endpoints
---------
``POST /``
    Generate a multi-result report (PDF / Excel / CSV).
``POST /pdf``
    Generate an enterprise PDF from a single analysis result.
``GET  /download-report``
    Download the most recent (or named) report file.
``GET  /download-log``
    Download the application log file.
``GET  /download-audit``
    Download the JSONL audit log.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.core.config import settings
from backend.core.enhanced_reports import EnhancedReportManager, ReportTemplate
from backend.io.pdf_generator import PDFReportGenerator, generate_pdf_report

logger = logging.getLogger(__name__)
router = APIRouter()

class ReportGenerationRequest(BaseModel):
    """Request payload for multi-result report generation.

    Attributes:
        results:              List of analysis result dicts to include.
        format_type:          Output format (``pdf``, ``excel``, ``csv``, or ``both``).
        title:                Custom report title (optional).
        include_methodology:  Append a methodology section.
        include_raw_data:     Append raw data in an appendix.
        include_charts:       Embed visualizations.
    """
    results: List[Dict[str, Any]] = Field(..., description="List of analysis results to include in report")
    format_type: str = Field("pdf", description="Report format: 'pdf', 'excel', 'csv', or 'both'")
    title: Optional[str] = Field(None, description="Custom report title")
    include_methodology: bool = Field(True, description="Include methodology section")
    include_raw_data: bool = Field(True, description="Include raw data in appendix")
    include_charts: bool = Field(True, description="Include visualizations")

# Initialize report manager
report_manager = EnhancedReportManager()


@router.get('/download-log', response_model=None)
def download_log() -> Response:
    """Download the application log file as a plain-text attachment.

    Returns:
        ``Response`` with the log content, or a 404 if no log exists.
    """
    log_path = settings.get_log_path()
    
    if not log_path or not log_path.exists():
        return Response(content='Log file not found', media_type='text/plain', status_code=404)
        
    with open(log_path, 'rb') as f:
        content = f.read()
    return Response(content=content, media_type='text/plain', headers={'Content-Disposition': f'attachment; filename="{log_path.name}"'})

@router.get('/download-audit', response_model=None)
def download_audit() -> Response:
    """Download the JSONL audit log as an attachment.

    Verifies the resolved path stays within the audit directory to
    prevent path-traversal attacks.

    Returns:
        ``Response`` with the audit log, 404 if missing, or 403 if
        the path resolves outside the allowed directory.
    """
    # Use centralized config for audit path
    audit_dir = settings.PROJECT_ROOT / "data" / "audit"
    audit_path = audit_dir / "audit_log.jsonl"
    if not audit_path.exists():
        return Response(content='Audit log not found', media_type='text/plain', status_code=404)
    # Verify resolved path stays within audit directory (prevent traversal)
    if not str(audit_path.resolve()).startswith(str(audit_dir.resolve())):
        return Response(content='Invalid path', media_type='text/plain', status_code=403)
    with open(audit_path, 'rb') as f:
        content = f.read()
    return Response(content=content, media_type='application/json', headers={'Content-Disposition': 'attachment; filename="audit_log.jsonl"'})


# =============================================================================
# FIX 17: ENTERPRISE PDF REPORT GENERATION ENDPOINT
# =============================================================================

class PDFReportRequest(BaseModel):
    """Request payload for single-result enterprise PDF generation."""
    analysis_result: Dict[str, Any] = Field(..., description="Single analysis result to generate PDF for")
    include_raw_data: bool = Field(True, description="Include raw data in appendix")
    custom_filename: Optional[str] = Field(None, description="Custom filename (without extension)")


@router.post("/pdf")
async def generate_pdf_report_endpoint(request: PDFReportRequest) -> Dict[str, Any]:
    """Generate an enterprise-grade PDF report from a single analysis result.

    Produces a multi-page document with title page, table of contents,
    executive summary, AI interpretation, key findings, generated code,
    methodology, and optional raw-data appendix.

    Returns:
        Dict with ``success``, ``download_url``, and ``metadata`` on success;
        ``error`` and ``suggestion`` on failure.
    """
    logger.info("Generating enterprise PDF report")
    
    try:
        # Generate custom output path if filename provided
        output_path = None
        if request.custom_filename:
            reports_dir = settings.get_reports_path()
            
            # Sanitize filename
            safe_filename = "".join(c for c in request.custom_filename if c.isalnum() or c in (' ', '-', '_'))
            output_path = str(reports_dir / f"{safe_filename}.pdf")
        
        # Generate PDF using enterprise generator
        pdf_path = generate_pdf_report(
            analysis_result=request.analysis_result,
            output_path=output_path,
            include_raw_data=request.include_raw_data
        )
        
        # Extract filename
        filename = os.path.basename(pdf_path)
        
        logger.info("PDF report generated: %s", filename)
        
        return {
            "success": True,
            "message": "Enterprise PDF report generated successfully",
            "report_path": filename,
            "format": "pdf",
            "download_url": f"/api/report/download-report?filename={filename}",
            "features": [
                "Professional title page",
                "Table of contents",
                "Executive summary",
                "Query analysis",
                "AI interpretation",
                "Orchestrator reasoning",
                "Key findings",
                "Detailed results",
                "Data insights",
                "Generated code",
                "Visualizations",
                "Methodology",
                "Technical details",
                "Raw data appendix" if request.include_raw_data else "No appendix"
            ],
            "metadata": {
                "query": request.analysis_result.get('query', 'N/A')[:100],
                "model": request.analysis_result.get('model_used', 'Unknown'),
                "agent": request.analysis_result.get('agent', 'Unknown'),
                "pages": "Multiple",
                "file_size": f"{os.path.getsize(pdf_path) / 1024:.1f} KB"
            }
        }
        
    except Exception as e:
        logger.error("PDF generation failed: %s", e, exc_info=True)
        return {
            "success": False,
            "error": f"PDF generation failed: {str(e)}",
            "suggestion": "Check the analysis result format. Ensure it contains 'query' and result data."
        }


# =============================================================================
# Multi-Result Report Generation
# =============================================================================

@router.post("/")
async def generate_report(request: ReportGenerationRequest) -> Dict[str, Any]:
    """Generate a multi-result professional report (PDF / Excel / CSV)."""
    logger.info(
        "Generating %s report with %d results",
        request.format_type, len(request.results),
    )
    
    try:
        # Handle CSV export separately (simpler format)
        if request.format_type == 'csv':
            csv_path = _generate_csv_export(request.results)
            
            return {
                "success": True,
                "message": "CSV export generated successfully",
                "report_path": os.path.basename(csv_path),
                "format": "csv",
                "analysis_count": len(request.results)
            }
        
        # Create template with custom title
        template = ReportTemplate(title=request.title) if request.title else None
        
        # Generate report using enhanced report manager
        report_path = report_manager.generate_report(
            analysis_results=request.results,
            format_type=request.format_type,
            template=template
        )
        
        # Store the report path for download
        data_dir = settings.get_reports_path()
        # os.makedirs(data_dir, exist_ok=True) # settings.get_reports_path() already treats it as Path, but ensure it exists? settings.get_reports_path() creates it!
        
        # Copy to data directory for download
        if isinstance(report_path, list):
            # Multiple files generated
            stored_paths = []
            for path in report_path:
                filename = os.path.basename(path)
                stored_path = data_dir / filename
                
                # Copy file
                shutil.copy2(path, stored_path)
                stored_paths.append(filename)
            
            return {
                "success": True,
                "message": f"Reports generated successfully",
                "report_paths": stored_paths,
                "format": request.format_type,
                "analysis_count": len(request.results)
            }
        else:
            # Single file generated
            filename = os.path.basename(report_path)
            stored_path = data_dir / filename
            
            # Copy file
            shutil.copy2(report_path, stored_path)
            
            return {
                "success": True,
                "message": f"Report generated successfully",
                "report_path": filename,
                "format": request.format_type,
                "analysis_count": len(request.results)
            }
            
    except Exception as e:
        logger.error("Report generation failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": f"Report generation failed: {str(e)}",
                "suggestion": "Check the analysis results format and try again"
            },
        )


def _generate_csv_export(results: List[Dict[str, Any]]) -> str:
    """Convert analysis results to a CSV file in the reports directory.

    Args:
        results: List of analysis result dicts.

    Returns:
        Absolute path to the written CSV file.
    """
    try:
        # Extract key data from results
        csv_data = []
        for i, result in enumerate(results, 1):
            row = {
                'Analysis_ID': i,
                'Query': result.get('query', 'N/A'),
                'Filename': result.get('filename', 'N/A'),
                'Result': str(result.get('result', 'No result'))[:500],  # Truncate for CSV
                'Execution_Time_s': result.get('execution_time', 0),
                'Model': result.get('metadata', {}).get('model', 'N/A'),
                'Routing_Tier': result.get('metadata', {}).get('routing_tier', 'N/A'),
                'Cache_Hit': result.get('metadata', {}).get('cache_hit', False),
                'Success': result.get('success', True)
            }
            csv_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(csv_data)
        
        # Generate filename
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f"analysis_export_{timestamp}.csv"
        
        # Save to data/reports directory
        data_dir = settings.get_reports_path()
        csv_path = data_dir / csv_filename
        
        df.to_csv(csv_path, index=False)
        logger.info("CSV export generated: %s", csv_path)
        
        return csv_path
        
    except Exception as e:
        logger.error("CSV generation failed: %s", e, exc_info=True)
        raise e


@router.get('/download-report', response_model=None)
def download_report(filename: Optional[str] = None) -> Union[FileResponse, Response]:
    """Download the most recent generated report, or a specific file by name.

    Args:
        filename: Optional report filename. If omitted, serves the newest
            PDF/Excel/CSV in the reports directory.

    Returns:
        ``FileResponse`` with the appropriate content type.
    """
    try:
        # Look in reports directory
        reports_dir = settings.get_reports_path()
        
        report_file = None
        
        if filename:
            # Sanitize filename - prevent path traversal
            from werkzeug.utils import secure_filename
            safe_name = secure_filename(filename)
            if not safe_name:
                return Response(
                    content='Invalid filename',
                    media_type='text/plain',
                    status_code=400
                )
            potential_path = reports_dir / safe_name
            # Verify resolved path stays within reports directory
            if not str(potential_path.resolve()).startswith(str(reports_dir.resolve())):
                return Response(
                    content='Invalid file path',
                    media_type='text/plain',
                    status_code=403
                )
            if potential_path.exists():
                report_file = potential_path
        else:
            # Look for the most recent PDF/Excel/CSV report
            if reports_dir.exists():
                report_files = list(reports_dir.glob("*.pdf")) + \
                             list(reports_dir.glob("*.xlsx")) + \
                             list(reports_dir.glob("*.csv"))
                if report_files:
                    # Get the most recent file
                    report_file = max(report_files, key=lambda p: p.stat().st_mtime)
        
        if not report_file or not report_file.exists():
            return Response(
                content='No report found. Generate a report first.', 
                media_type='text/plain', 
                status_code=404
            )
        
        # Determine content type based on file extension
        content_type = "application/pdf"
        if report_file.suffix.lower() == ".xlsx":
            content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif report_file.suffix.lower() == ".csv":
            content_type = "text/csv"
        elif report_file.suffix.lower() == ".json":
            content_type = "application/json"
        
        logger.info("Serving report: %s", report_file.name)
            
        return FileResponse(
            path=str(report_file),
            media_type=content_type,
            filename=report_file.name
        )
        
    except Exception as e:
        logger.error("Report download failed: %s", e, exc_info=True)
        return Response(
            content=f'Report download failed: {str(e)}', 
            media_type='text/plain', 
            status_code=500
        )
