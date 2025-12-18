# Enhanced endpoint for professional report generation and downloads
from fastapi import APIRouter, Response, Request, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
import logging
from backend.core.enhanced_reports import EnhancedReportManager, ReportTemplate

router = APIRouter()

class ReportGenerationRequest(BaseModel):
    results: List[Dict[str, Any]]
    format_type: str = "pdf"  # "pdf", "excel", or "both"
    title: Optional[str] = None
    include_methodology: bool = True
    include_raw_data: bool = True

# Initialize enhanced report manager
report_manager = EnhancedReportManager()


@router.get('/download-log')
def download_log():
    log_path = os.path.join(os.path.dirname(__file__), '../../nexus.log')
    if not os.path.exists(log_path):
        return Response(content='Log file not found', media_type='text/plain', status_code=404)
    with open(log_path, 'rb') as f:
        content = f.read()
    return Response(content=content, media_type='text/plain', headers={'Content-Disposition': 'attachment; filename="nexus.log"'})

@router.get('/download-audit')
def download_audit():
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
    Generate professional reports in multiple formats
    """
    logging.info(f"[REPORT] Generating {request.format_type} report with {len(request.results)} analysis results")
    
    try:
        # Create custom template if title is provided
        template = None
        if request.title:
            template = ReportTemplate(title=request.title)
        
        # Generate report using enhanced report manager
        report_path = report_manager.generate_report(
            analysis_results=request.results,
            format_type=request.format_type,
            template=template
        )
        
        # Store the report path for download
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Copy to data directory for download
        if isinstance(report_path, list):
            # Multiple files generated
            stored_paths = []
            for path in report_path:
                filename = os.path.basename(path)
                stored_path = os.path.join(data_dir, filename)
                
                # Copy file
                import shutil
                shutil.copy2(path, stored_path)
                stored_paths.append(stored_path)
            
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
            stored_path = os.path.join(data_dir, filename)
            
            # Copy file
            import shutil
            shutil.copy2(report_path, stored_path)
            
            return {
                "success": True,
                "message": f"Report generated successfully",
                "report_path": stored_path,
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


# Download generated report
@router.get('/download-report')
def download_report(filename: Optional[str] = None):
    """
    Download the most recent generated report or a specific file
    """
    try:
        from pathlib import Path
        
        # Get the project root and data directories
        backend_dir = Path(__file__).parent.parent
        project_root = backend_dir.parent.parent
        
        # Look in multiple possible locations
        data_dirs = [
            project_root / "data" / "samples",
            project_root / "data", 
            backend_dir / "data",
            project_root / "reports"
        ]
        
        report_file = None
        
        if filename:
            # Look for specific filename
            for data_dir in data_dirs:
                potential_path = data_dir / filename
                if potential_path.exists():
                    report_file = potential_path
                    break
        else:
            # Look for the most recent PDF report
            for data_dir in data_dirs:
                if data_dir.exists():
                    pdf_files = list(data_dir.glob("*.pdf"))
                    if pdf_files:
                        # Get the most recent PDF file
                        report_file = max(pdf_files, key=lambda p: p.stat().st_mtime)
                        break
        
        if not report_file or not report_file.exists():
            return Response(
                content='No report found. Generate a report first.', 
                media_type='text/plain', 
                status_code=404
            )
        
        # Read and return the file
        with open(report_file, "rb") as f:
            content = f.read()
        
        # Determine content type based on file extension
        content_type = "application/pdf"
        if report_file.suffix.lower() == ".xlsx":
            content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif report_file.suffix.lower() == ".json":
            content_type = "application/json"
            
        return Response(
            content=content, 
            media_type=content_type, 
            headers={
                'Content-Disposition': f'attachment; filename="{report_file.name}"'
            }
        )
        
    except Exception as e:
        logging.error(f"Report download failed: {e}")
        return Response(
            content=f'Report download failed: {str(e)}', 
            media_type='text/plain', 
            status_code=500
        )
