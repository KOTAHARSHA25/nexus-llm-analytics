# Endpoint to download logs and reports
from fastapi import APIRouter, Response, Request
import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO


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
    data = await request.json()
    analysis_results = data.get("results", [])

    # Generate PDF
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    # Add a default paragraph style if it doesn't exist
    if 'p' not in styles:
        styles.add(ParagraphStyle(name='p', parent=styles['Normal']))
    story = []

    story.append(Paragraph("Nexus-LLM-Analytics Report", styles['h1']))
    story.append(Spacer(1, 12))

    for result in analysis_results:
        story.append(Paragraph(result.get("title", "Analysis Result"), styles['h2']))
        story.append(Paragraph(f"Query: {result.get('query', 'N/A')}", styles['p']))
        story.append(Paragraph(f"Code: <code>{result.get('code', 'N/A')}</code>", styles['p']))
        story.append(Paragraph(f"Explanation: {result.get('explanation', 'N/A')}", styles['p']))
        
        # Handle different result types (preview, describe, etc.)
        if 'preview' in result:
            story.append(Paragraph(json.dumps(result['preview'], indent=2), styles['p']))
        if 'describe' in result:
            story.append(Paragraph(json.dumps(result['describe'], indent=2), styles['p']))
        if 'value_counts' in result:
            story.append(Paragraph(json.dumps(result['value_counts'], indent=2), styles['p']))
            
        story.append(Spacer(1, 12))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    # For simplicity, we'll save it to a temporary file.
    # In a real app, you might use a more robust storage solution.
    report_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'generated_report.pdf')
    with open(report_path, "wb") as f:
        f.write(pdf_bytes)

    return {"message": "Report generated successfully", "report_path": report_path}


# Download generated report stub
@router.get('/download-report')
def download_report():
    report_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'generated_report.pdf')
    if not os.path.exists(report_path):
        return Response(content='Report not found. Generate a report first.', media_type='text/plain', status_code=404)
    
    with open(report_path, "rb") as f:
        content = f.read()
        
    return Response(content=content, media_type='application/pdf', headers={'Content-Disposition': 'attachment; filename="generated_report.pdf"'})
