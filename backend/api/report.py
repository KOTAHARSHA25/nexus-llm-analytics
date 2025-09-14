# Endpoint to download logs and reports
from fastapi import APIRouter, Response, Request
import os


router = APIRouter()


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



# Handles /generate-report endpoint for report generation
@router.post("/")
async def generate_report(request: Request):
    # TODO: Compile results and return generated report file
    return {"message": "Report generation endpoint stub"}

# Download generated report stub
@router.get('/download-report')
def download_report():
    # TODO: Serve generated report file (PDF, Excel, etc.)
    return Response(content='Report download not implemented', media_type='text/plain', status_code=501)
