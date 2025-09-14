from fastapi import APIRouter, UploadFile, File
import os

router = APIRouter()

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Handles /upload-documents endpoint for file uploads
@router.post("/")
async def upload_document(file: UploadFile = File(...)):
    # Only allow CSV or JSON
    if not (file.filename.endswith('.csv') or file.filename.endswith('.json')):
        return {"error": "Only CSV and JSON files are supported."}
    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"filename": file.filename, "message": "File uploaded successfully."}
