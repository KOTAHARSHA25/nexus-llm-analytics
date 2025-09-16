from fastapi import APIRouter, UploadFile, File
import os
import io
from werkzeug.utils import secure_filename
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
    import logging
    allowed = ('.csv', '.json', '.pdf', '.txt')
    filename = secure_filename(file.filename)
    logging.info(f"[UPLOAD] Received file: {filename}")
    if not filename.lower().endswith(allowed):
        logging.warning(f"[UPLOAD] Rejected file type: {filename}")
        return {"error": "Only CSV, JSON, PDF, and TXT files are supported."}
    file_path = os.path.join(DATA_DIR, filename)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    logging.info(f"[UPLOAD] Saved file to: {file_path}")

    # Extract text if PDF or TXT
    extracted_text = None
    if filename.lower().endswith('.pdf'):
        if PyPDF2 is None:
            logging.error("[UPLOAD] PyPDF2 not installed for PDF extraction.")
            return {"error": "PyPDF2 is not installed. Please install it to process PDFs."}
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            extracted_text = "\n".join(page.extract_text() or '' for page in pdf_reader.pages)
            logging.info(f"[UPLOAD] Extracted text from PDF: {filename}")
        except Exception as e:
            logging.error(f"[UPLOAD] Failed to extract text from PDF: {e}")
            return {"error": f"Failed to extract text from PDF: {e}"}
    elif filename.lower().endswith('.txt'):
        try:
            extracted_text = content.decode('utf-8')
            logging.info(f"[UPLOAD] Extracted text from TXT: {filename}")
        except Exception as e:
            logging.error(f"[UPLOAD] Failed to read TXT file: {e}")
            return {"error": f"Failed to read TXT file: {e}"}

    # Save extracted text if available
    if extracted_text:
        text_path = file_path + '.extracted.txt'
        with open(text_path, 'w', encoding='utf-8') as tf:
            tf.write(extracted_text)
        logging.info(f"[UPLOAD] Saved extracted text to: {text_path}")

    logging.info(f"[UPLOAD] Upload complete for: {filename}")
    return {"filename": filename, "message": "File uploaded successfully.", "extracted_text_path": text_path if extracted_text else None}
