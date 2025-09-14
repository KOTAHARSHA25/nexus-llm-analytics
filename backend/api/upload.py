
from fastapi import APIRouter, UploadFile, File
import os
import io
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

router = APIRouter()

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)


# Handles /upload-documents endpoint for file uploads (CSV, JSON, PDF, TXT)
@router.post("/")
async def upload_document(file: UploadFile = File(...)):
    allowed = ('.csv', '.json', '.pdf', '.txt')
    if not file.filename.lower().endswith(allowed):
        return {"error": "Only CSV, JSON, PDF, and TXT files are supported."}
    file_path = os.path.join(DATA_DIR, file.filename)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Extract text if PDF or TXT
    extracted_text = None
    if file.filename.lower().endswith('.pdf'):
        if PyPDF2 is None:
            return {"error": "PyPDF2 is not installed. Please install it to process PDFs."}
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            extracted_text = "\n".join(page.extract_text() or '' for page in pdf_reader.pages)
        except Exception as e:
            return {"error": f"Failed to extract text from PDF: {e}"}
    elif file.filename.lower().endswith('.txt'):
        try:
            extracted_text = content.decode('utf-8')
        except Exception as e:
            return {"error": f"Failed to read TXT file: {e}"}

    # Save extracted text if available
    if extracted_text:
        text_path = file_path + '.extracted.txt'
        with open(text_path, 'w', encoding='utf-8') as tf:
            tf.write(extracted_text)

    return {"filename": file.filename, "message": "File uploaded successfully.", "extracted_text_path": text_path if extracted_text else None}
