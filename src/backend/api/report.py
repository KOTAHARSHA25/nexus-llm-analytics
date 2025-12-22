# Enhanced endpoint for professional report generation and downloads
from fastapi import APIRouter, Response, Request, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import json
import logging
import pandas as pd

# Import the enhanced report manager
from backend.core.enhanced_reports import EnhancedReportManager, ReportTemplate

router = APIRouter()

class ReportGenerationRequest(BaseModel):
    """Request model for generating professional reports"""
    results: List[Dict[str, Any]] = Field(..., description="List of analysis results to include in report")
    format_type: str = Field("pdf", description="Report format: 'pdf', 'excel', 'csv', or 'both'")
    title: Optional[str] = Field(None, description="Custom report title")
    include_methodology: bool = Field(True, description="Include methodology section")
    include_raw_data: bool = Field(True, description="Include raw data in appendix")
    include_charts: bool = Field(True, description="Include visualizations")

# Initialize report manager
report_manager = EnhancedReportManager()


@router.get('/download-log')
@router.get('/download-log')
def download_log():
    from backend.core.config import settings
    log_path = settings.get_log_path()
    
    if not log_path or not log_path.exists():
        return Response(content='Log file not found', media_type='text/plain', status_code=404)
        
    with open(log_path, 'rb') as f:
        content = f.read()
    return Response(content=content, media_type='text/plain', headers={'Content-Disposition': f'attachment; filename="{log_path.name}"'})

@router.get('/download-audit')
def download_audit():
    # Audit log path could be added to settings, but for now we'll match the pattern
    audit_path = os.path.join(os.path.dirname(__file__), '../../data/audit/audit_log.jsonl')
    if not os.path.exists(audit_path):
        return Response(content='Audit log not found', media_type='text/plain', status_code=404)
    with open(audit_path, 'rb') as f:
        content = f.read()
    return Response(content=content, media_type='application/json', headers={'Content-Disposition': 'attachment; filename="audit_log.jsonl"'})



# Enhanced report generation endpoint
@router.post("/")
async def generate_report(request: ReportGenerationRequest):
    """
    Generate professional reports in multiple formats (PDF/Excel/CSV)
    
    Features:
    - Professional headers and footers with page numbers
    - Executive summary with key metrics
    - Key findings section
    - Statistical summary with confidence intervals
    - Data quality assessment
    - Methodology and glossary sections
    """
    logging.info(f"[REPORT] Generating {request.format_type} report with {len(request.results)} results")
    
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
        from backend.core.config import settings
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
                import shutil
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
            import shutil
            shutil.copy2(report_path, stored_path)
            
            return {
                "success": True,
                "message": f"Report generated successfully",
                "report_path": filename,
                "format": request.format_type,
                "analysis_count": len(request.results)
            }
            
    except Exception as e:
        logging.error(f"[REPORT] Generation failed: {e}")
        return {
            "success": False,
            "error": f"Report generation failed: {str(e)}",
            "suggestion": "Check the analysis results format and try again"
        }


def _generate_csv_export(results: List[Dict[str, Any]]) -> str:
    """Generate simple CSV export of analysis results"""
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
        from backend.core.config import settings
        data_dir = settings.get_reports_path()
        csv_path = data_dir / csv_filename
        
        df.to_csv(csv_path, index=False)
        logging.info(f"[CSV] Export generated: {csv_path}")
        
        return csv_path
        
    except Exception as e:
        logging.error(f"[CSV] Generation failed: {e}")
        raise e


# Download generated report
@router.get('/download-report')
def download_report(filename: Optional[str] = None):
    """
    Download the most recent generated report or a specific file
    """
    try:
        from pathlib import Path
        from backend.core.config import settings
        
        # Look in reports directory
        reports_dir = settings.get_reports_path()
        
        report_file = None
        
        if filename:
            # Look for specific filename
            potential_path = reports_dir / filename
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
        
        logging.info(f"[DOWNLOAD] Serving report: {report_file.name}")
            
        return FileResponse(
            path=str(report_file),
            media_type=content_type,
            filename=report_file.name
        )
        
    except Exception as e:
        logging.error(f"Report download failed: {e}")
        return Response(
            content=f'Report download failed: {str(e)}', 
            media_type='text/plain', 
            status_code=500
        )
