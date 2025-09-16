# Enhanced endpoint for professional report generation and downloads
from fastapi import APIRouter, Response, Request, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
import logging
from core.enhanced_reports import EnhancedReportManager, ReportTemplate

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


# Download generated report stub
@router.get('/download-report')
def download_report():
    report_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'generated_report.pdf')
    if not os.path.exists(report_path):
        return Response(content='Report not found. Generate a report first.', media_type='text/plain', status_code=404)
    
    with open(report_path, "rb") as f:
        content = f.read()
        
    return Response(content=content, media_type='application/pdf', headers={'Content-Disposition': 'attachment; filename="generated_report.pdf"'})
